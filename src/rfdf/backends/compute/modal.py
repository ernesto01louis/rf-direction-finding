"""Modal cloud compute backend.

Submits :class:`rfdf.hal.ComputeJob`s to `Modal <https://modal.com>`_ via the
``modal`` Python SDK.  The SDK is imported **lazily** — every method that needs
it calls :func:`_require_modal` — so this module is fully importable (and the
factory discoverable by ``rfdf compute list``) even when the ``modal`` package
is not installed.

Auth:
    Modal reads credentials from ``~/.modal.toml`` (written by ``modal token
    new``) or from the ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` environment
    variables.  A missing or empty token raises a :class:`RuntimeError` at
    first use.  The credential check happens inside :func:`_check_credentials`
    which is called at the start of every method that hits the Modal API.

Cost estimates:
    :meth:`ModalCompute.cost_estimate` uses a static ``_RATES`` table of
    indicative per-second GPU prices converted to USD/hr.  Modal bills per
    second, so actual costs will be lower for short jobs than the 1.5x padded
    estimate suggests.  These figures are *indicative only* — Modal's pricing
    changes; verify at `modal.com/pricing <https://modal.com/pricing>`_ and
    update ``_RATES`` when it diverges.

Modal execution model:
    Jobs are submitted as Modal :class:`modal.Sandbox` containers.  A Sandbox
    runs the job command directly without defining a ``modal.Function``; this
    avoids the need for a persistent ``modal.App`` deployment and is suited to
    ad-hoc compute jobs.  The Sandbox ID (``sandbox.object_id``) is stored in
    :attr:`JobHandle.job_id` so that :meth:`status`, :meth:`logs`, and
    :meth:`cancel` can look it up via :func:`modal.Sandbox.from_id`.

Usage::

    from rfdf.backends.compute.modal import create

    backend = create()
    async with backend:  # not strictly required — no cleanup needed
        handle = await backend.submit(job)
        status = await backend.status(handle)
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from rfdf.backends.compute._remote_storage import ModalVolumeStorage
from rfdf.hal.compute import ComputeJob, CostEstimate, JobHandle
from rfdf.hal.types import JobStatus

# ---------------------------------------------------------------------------
# Indicative hourly GPU rates (USD/hr) — update from modal.com/pricing.
# Modal bills per second; these are rough hourly equivalents for estimating.
# Keys match Modal's GPU type strings used in the ``gpu`` parameter.
# ``None`` is the fallback applied when no ``gpu_model`` is specified.
# ---------------------------------------------------------------------------

_RATES: dict[str | None, float] = {
    None: 0.36,  # fallback — cheapest available GPU (T4-equivalent)
    "T4": 0.36,
    "L4": 0.80,
    "A10G": 1.10,
    "A100": 3.70,
    "A100-80GB": 4.50,
    "H100": 4.98,
    "H100-80GB": 5.25,
    "H100-NVLINK": 6.20,
    "A10": 1.10,  # alias
    "L40S": 2.50,
}

# Modal Sandbox returncode mapping to JobStatus.
# A None returncode means the sandbox is still running.
_RETURNCODE_MAP: dict[int, JobStatus] = {
    0: JobStatus.COMPLETED,
}

# Default image used when no container_image is specified.
_DEFAULT_IMAGE_TAG = "python:3.11-slim"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_modal() -> Any:
    """Import and return the ``modal`` module, raising a clear error if absent.

    Returns:
        The ``modal`` module object.

    Raises:
        ImportError: With install instructions when the SDK is absent.
    """
    try:
        import modal  # lazy import — intentional

        return modal
    except ImportError as exc:
        raise ImportError(
            "The Modal backend requires the modal SDK; "
            "install it with:  pip install rfdf[compute-modal]"
        ) from exc


def _check_credentials() -> None:
    """Verify that Modal credentials are available in the environment.

    Modal reads credentials from ``~/.modal.toml`` (populated by
    ``modal token new``) or the ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``
    environment variables.  Both sources are probed via the SDK's config
    object.

    Raises:
        RuntimeError: If no credentials are found.
        ImportError: If the ``modal`` SDK is not installed.
    """
    modal = _require_modal()
    token_id = modal.config.config.get("token_id") or os.environ.get("MODAL_TOKEN_ID", "").strip()
    token_secret = (
        modal.config.config.get("token_secret") or os.environ.get("MODAL_TOKEN_SECRET", "").strip()
    )
    if not token_id or not token_secret:
        raise RuntimeError(
            "ModalCompute: Modal credentials are not configured. "
            "Run `modal token new` to set up credentials, or export "
            "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET before using this backend."
        )


def _pick_gpu_spec(job: ComputeJob) -> str | None:
    """Return a Modal GPU specification string for the given job.

    Modal's ``gpu`` parameter accepts a string such as ``"T4"`` or ``"A100"``
    (and optionally a count suffix like ``"A100:2"``).

    Args:
        job: The compute job whose GPU requirements drive the selection.

    Returns:
        A Modal GPU type string, or ``None`` for CPU-only jobs.
    """
    if job.gpu_count == 0:
        return None
    model = job.gpu_model or "T4"
    count = max(job.gpu_count, 1)
    if count > 1:
        return f"{model}:{count}"
    return model


def _build_command(job: ComputeJob) -> list[str]:
    """Build the command list for the Sandbox.

    When ``container_image`` is set the entry script is run directly inside
    the user's image.  When only ``pip_requirements`` is provided, a
    ``bash -c`` wrapper installs dependencies first.

    Args:
        job: The compute job.

    Returns:
        A list of command tokens suitable for ``modal.Sandbox.create(*args)``.
    """
    if job.container_image:
        # Entry script runs as-is inside the user's image.
        return ["python", job.entry_script]
    if job.pip_requirements:
        reqs = " ".join(f'"{r}"' for r in job.pip_requirements)
        pip_cmd = f"pip install -q {reqs}"
        inner = f"{pip_cmd} && python {job.entry_script}"
        return ["bash", "-c", inner]
    return ["bash", "-c", f"python {job.entry_script}"]


def _map_sandbox_status(returncode: int | None) -> JobStatus:
    """Map a Sandbox returncode to a :class:`JobStatus`.

    Args:
        returncode: The Sandbox returncode.  ``None`` means still running.

    Returns:
        The mapped :class:`JobStatus`.
    """
    if returncode is None:
        return JobStatus.RUNNING
    if returncode == 0:
        return JobStatus.COMPLETED
    return JobStatus.FAILED


async def _stream_sandbox_logs(handle: JobHandle) -> AsyncIterator[str]:
    """Async generator that streams stdout/stderr from a Modal Sandbox.

    Fetches the sandbox by ID, then reads stdout/stderr lines.  If the
    sandbox is not found or has already terminated, yields a final status
    line and returns.

    Args:
        handle: The job handle whose sandbox to stream.

    Yields:
        One line per log entry (stdout or stderr, interleaved by poll order).
    """
    modal = _require_modal()
    _check_credentials()
    try:
        sandbox = await asyncio.to_thread(modal.Sandbox.from_id, handle.job_id)
    except Exception as exc:
        yield f"[modal] sandbox {handle.job_id} lookup failed: {exc}"
        return

    # Drain stdout then stderr in a simple poll loop; Modal's StreamReader
    # supports async iteration for real-time streaming.
    poll_interval = 5.0
    while True:
        returncode = await asyncio.to_thread(lambda: sandbox.returncode)
        yield f"[modal] sandbox={handle.job_id} returncode={returncode}"
        if returncode is not None:
            # Collect any remaining log lines from stdout.
            try:
                out = await asyncio.to_thread(sandbox.stdout.read)
                if out:
                    for line in out.splitlines():
                        yield f"[modal/stdout] {line}"
                err = await asyncio.to_thread(sandbox.stderr.read)
                if err:
                    for line in err.splitlines():
                        yield f"[modal/stderr] {line}"
            except Exception as exc:
                yield f"[modal] log read error: {exc}"
            return
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# ModalCompute
# ---------------------------------------------------------------------------


class ModalCompute:
    """Implements :class:`rfdf.hal.ComputeBackend` for Modal GPU Sandboxes.

    Jobs are submitted as Modal :class:`modal.Sandbox` containers.  Each
    sandbox gets its own ephemeral image derived from ``_DEFAULT_IMAGE_TAG``
    (or the user-supplied ``container_image``).  The sandbox is terminated on
    :meth:`cancel`; Modal does not continue billing after termination.

    The SDK is imported lazily inside every method that uses it so that
    ``import rfdf.backends.compute.modal`` never pulls ``modal`` into
    ``sys.modules``.

    Attributes:
        storage: A :class:`~rfdf.backends.compute._remote_storage.ModalVolumeStorage`
            exposing the backend's persistent-storage interface.
    """

    name: str = "modal"

    def __init__(self, volume_name: str = "") -> None:
        """Initialise the backend.

        Args:
            volume_name: Optional Modal Volume name to attach to every submitted
                sandbox.  Leave empty to use ephemeral container storage only.
        """
        self._volume_name = volume_name
        # Map job_id → sandbox handle for in-process lookups.
        self._jobs: dict[str, Any] = {}
        self.storage = ModalVolumeStorage(volume_name=volume_name or "rfdf-default")

    @property
    def supports_gpu(self) -> bool:
        """Modal provides GPU instances; always ``True``."""
        return True

    @property
    def supports_persistent_storage(self) -> bool:
        """Modal Volumes persist across sandboxes; always ``True``."""
        return True

    async def submit(self, job: ComputeJob) -> JobHandle:
        """Create a Modal Sandbox for *job* and return a handle.

        The sandbox is started immediately.  The handle's ``extra`` dict carries
        the sandbox ID and submission metadata.  Modal Sandboxes do not require
        a pre-deployed App function; they accept arbitrary commands.

        Args:
            job: The compute job to submit.

        Returns:
            A :class:`JobHandle` with ``backend="modal"`` and ``job_id`` set to
            the Modal Sandbox object ID.

        Raises:
            RuntimeError: If Modal credentials are absent.
            ImportError: If the ``modal`` SDK is not installed.
        """
        modal = _require_modal()
        _check_credentials()

        image_tag = job.container_image or _DEFAULT_IMAGE_TAG
        gpu_spec = _pick_gpu_spec(job)
        command = _build_command(job)

        # Build the Modal Image from a registry tag.
        if job.container_image:
            image = modal.Image.from_registry(job.container_image)
        else:
            image = modal.Image.from_registry(_DEFAULT_IMAGE_TAG)
            if job.pip_requirements:
                image = image.pip_install(*job.pip_requirements)

        # Forward job env vars; add RFDF-specific context.
        env: dict[str, str] = dict(job.env)
        env["RFDF_JOB_TIMEOUT_S"] = str(int(job.timeout_s))
        if job.artifact_globs:
            env["RFDF_ARTIFACT_GLOBS"] = ",".join(job.artifact_globs)

        # App lookup for sandbox association.
        app_name = f"rfdf-job-{uuid.uuid4().hex[:8]}"

        # Build Sandbox kwargs.
        kwargs: dict[str, Any] = {
            "app": await asyncio.to_thread(modal.App.lookup, app_name, create_if_missing=True),
            "image": image,
            "env": env,
            "timeout": int(job.timeout_s),
        }
        if gpu_spec is not None:
            kwargs["gpu"] = gpu_spec

        # Attach a Modal Volume if one is configured.
        if self._volume_name:
            vol = await asyncio.to_thread(
                modal.Volume.from_name, self._volume_name, create_if_missing=True
            )
            kwargs["volumes"] = {"/modal-volume": vol}

        sandbox = await asyncio.to_thread(modal.Sandbox.create, *command, **kwargs)
        job_id: str = sandbox.object_id
        submitted_at = time.time()
        self._jobs[job_id] = sandbox
        return JobHandle(
            backend=self.name,
            job_id=job_id,
            submitted_at_s=submitted_at,
            extra={
                "app_name": app_name,
                "image_tag": image_tag,
                "gpu_spec": gpu_spec,
            },
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        """Return the current :class:`JobStatus` for *handle*.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            The current :class:`JobStatus`.

        Raises:
            RuntimeError: If Modal credentials are absent.
            ImportError: If the ``modal`` SDK is not installed.
        """
        modal = _require_modal()
        _check_credentials()
        sandbox = self._jobs.get(handle.job_id)
        if sandbox is None:
            # Re-hydrate from Modal by ID.
            try:
                sandbox = await asyncio.to_thread(modal.Sandbox.from_id, handle.job_id)
                self._jobs[handle.job_id] = sandbox
            except Exception:
                return JobStatus.CANCELLED
        # sandbox is guaranteed non-None here — the exception path returned above.
        assert sandbox is not None
        returncode = await asyncio.to_thread(lambda: sandbox.returncode)
        return _map_sandbox_status(returncode)

    def logs(self, handle: JobHandle) -> AsyncIterator[str]:
        """Yield log lines from the Modal Sandbox's stdout/stderr.

        Polls the sandbox status every few seconds and yields combined
        stdout + stderr lines until the sandbox exits.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            An :class:`AsyncIterator` that yields log lines (without trailing
            newline) until the sandbox terminates.
        """
        return _stream_sandbox_logs(handle)

    async def fetch_artifacts(self, handle: JobHandle, dest: Path) -> None:
        """Copy job artifacts from the Modal Sandbox into *dest*.

        In a full deployment this would use Modal's ``sandbox.filesystem`` or
        a shared Volume to transfer files.  For the current implementation
        this is a documented stub that operators extend with their preferred
        transfer mechanism (Modal Volume download, DVC pull, custom
        ``sandbox.exec``-based copy).

        Args:
            handle: A handle previously returned by :meth:`submit`.
            dest: Local destination directory (must already exist).

        Raises:
            FileNotFoundError: If *dest* does not exist.
            NotADirectoryError: If *dest* is not a directory.
        """
        _require_modal()
        if not dest.exists():
            raise FileNotFoundError(f"ModalCompute.fetch_artifacts: dest {dest} must exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"ModalCompute.fetch_artifacts: dest {dest} is not a directory"
            )
        # Artifact transfer from a Modal Sandbox requires one of:
        #   - A shared Modal Volume mounted to both the sandbox and the client
        #   - modal.sandbox.filesystem() for direct file access (beta API)
        #   - DVC remote pointing at cloud storage written by the job
        # Log the intent and return so calling code can degrade gracefully.
        import structlog

        log = structlog.get_logger(__name__)
        log.warning(
            "modal_fetch_artifacts_noop",
            job_id=handle.job_id,
            dest=str(dest),
            note=(
                "Artifact transfer requires a shared Modal Volume or "
                "sandbox.filesystem(); implement via Volume.read_file() "
                "or a custom sandbox.exec() copy."
            ),
        )

    async def cancel(self, handle: JobHandle) -> None:
        """Terminate the Modal Sandbox.  No-op when already terminated.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Raises:
            RuntimeError: If Modal credentials are absent.
            ImportError: If the ``modal`` SDK is not installed.
        """
        modal = _require_modal()
        _check_credentials()
        current = await self.status(handle)
        if current in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return
        sandbox = self._jobs.get(handle.job_id)
        if sandbox is None:
            try:
                sandbox = await asyncio.to_thread(modal.Sandbox.from_id, handle.job_id)
            except Exception:
                return  # already gone
        # sandbox is guaranteed non-None here — the exception path returned above.
        assert sandbox is not None
        await asyncio.to_thread(sandbox.terminate)

    async def cost_estimate(self, job: ComputeJob) -> CostEstimate:
        """Return a pre-submit cost estimate based on static rate data.

        Formula::

            runtime_h = job.timeout_s / 3600
            gpu_units  = max(job.gpu_count, 1)
            hourly_usd = _RATES.get(job.gpu_model, _RATES[None])

            estimated_usd = hourly_usd * 1.5 * runtime_h * gpu_units
            low_usd       = hourly_usd * 1.0 * runtime_h * gpu_units
            high_usd      = hourly_usd * 2.0 * runtime_h * gpu_units

        The 1.5x multiplier is the expected cost including sandbox startup overhead
        and modest over-run.  The 1.0x / 2.0x bounds form a 2x spread that covers
        typical variance in job duration.  Modal bills per second, so short jobs
        will be cheaper than the 1.5x estimate implies.

        Args:
            job: The compute job to estimate costs for.

        Returns:
            A :class:`CostEstimate` with backend ``"modal"``.

        Note:
            ``_RATES`` values are indicative and may drift as Modal changes its
            pricing.  Always verify against `modal.com/pricing
            <https://modal.com/pricing>`_ before committing to a budget.
        """
        hourly_rate = _RATES.get(job.gpu_model, _RATES[None])
        runtime_h = job.timeout_s / 3600.0
        gpu_units = max(job.gpu_count, 1)
        estimated = hourly_rate * 1.5 * runtime_h * gpu_units
        low = hourly_rate * 1.0 * runtime_h * gpu_units
        high = hourly_rate * 2.0 * runtime_h * gpu_units
        model_str = job.gpu_model or "default GPU (T4)"
        rationale = (
            f"${hourly_rate:.4f}/hr x 1.5 pad x {runtime_h:.2f}h x {gpu_units} GPU(s) "
            f"= ${estimated:.4f} (model={model_str}); "
            f"low=${low:.4f} (1.0x), high=${high:.4f} (2.0x); "
            f"rates are indicative -- verify at modal.com/pricing"
        )
        return CostEstimate(
            backend=self.name,
            estimated_usd=estimated,
            low_usd=low,
            high_usd=high,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Entry-point factory
# ---------------------------------------------------------------------------


def create(**kwargs: Any) -> ModalCompute:
    """Factory wired into the ``rfdf.backends.compute`` entry-point.

    Import-safe: does NOT import ``modal``.  The SDK is only touched when a
    method is called on the returned backend instance.

    Args:
        **kwargs: Optional keyword arguments forwarded to :class:`ModalCompute`.
            Currently supports ``volume_name: str``.

    Returns:
        Configured :class:`ModalCompute`.
    """
    return ModalCompute(**kwargs)


__all__ = ["ModalCompute", "create"]

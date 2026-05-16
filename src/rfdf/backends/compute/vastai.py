"""Vast.ai cloud compute backend.

Submits :class:`rfdf.hal.ComputeJob`s to `Vast.ai <https://vast.ai>`_ GPU
instances via the ``vastai`` Python SDK.  The SDK is imported **lazily** —
every method that needs it calls :func:`_require_vastai` — so this module is
fully importable (and the factory discoverable by ``rfdf compute list``) even
when the ``vastai`` package is not installed.

Auth:
    Vast.ai reads its API key from (in order):

    1. The ``VAST_API_KEY`` environment variable.
    2. ``$XDG_CONFIG_HOME/vastai/vast_api_key`` (typically
       ``~/.config/vastai/vast_api_key``).
    3. The legacy file ``~/.vast_api_key``.

    A missing key raises a :class:`RuntimeError` at first use via the SDK's
    own :func:`_resolve_api_key` helper.  No rfdf-specific config file is
    consulted.

Cost estimates:
    :meth:`VastAiCompute.cost_estimate` uses a static ``_RATES`` table of
    indicative hourly GPU prices in USD.  Vast.ai is a **marketplace** — spot
    prices fluctuate per host.  The table lists representative on-demand floor
    prices; actual charges may be lower (spot) or higher (premium regions).
    Operators should update ``_RATES`` from the Vast.ai console when it diverges.

Marketplace reliability:
    Unlike managed clouds, Vast.ai hosts are third-party machines.  Instances
    **can be interrupted** if a host goes offline or another renter outbids you.
    For resilient workloads, set ``cancel_unavail=True`` in the ``create_instance``
    call (already the default here) and implement retry logic at the campaign
    level.  High-reliability workloads should prefer a managed cloud backend
    (RunPod reserved pods or Modal) where uptime is contractually guaranteed.

Usage::

    from rfdf.backends.compute.vastai import create

    backend = create()
    async with backend:  # not strictly required — no cleanup needed
        handle = await backend.submit(job)
        status = await backend.status(handle)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from rfdf.backends.compute._remote_storage import VastAiStorage
from rfdf.hal.compute import ComputeJob, CostEstimate, JobHandle
from rfdf.hal.types import JobStatus

# ---------------------------------------------------------------------------
# Indicative hourly GPU rates (USD/hr) — update from vast.ai/console.
# Vast.ai is a marketplace; these are indicative floor prices for on-demand
# instances. Actual billed rates vary by host and bid type.
# Keys match Vast.ai gpu_name strings (case-sensitive as returned by the API).
# ``None`` is the fallback applied when no ``gpu_model`` is specified.
# ---------------------------------------------------------------------------

_RATES: dict[str | None, float] = {
    None: 0.20,  # fallback — cheapest available GPU
    "RTX 4090": 0.35,
    "RTX 4080": 0.25,
    "RTX 3090": 0.20,
    "RTX 3080": 0.14,
    "RTX 3070": 0.10,
    "RTX 3060": 0.08,
    "RTX A4000": 0.18,
    "RTX A5000": 0.28,
    "RTX A6000": 0.42,
    "A100 SXM4": 1.60,
    "A100 PCIe": 1.20,
    "A100 80GB": 1.80,
    "H100 SXM5": 2.80,
    "H100 PCIe": 2.50,
    "H100 NVL": 3.00,
    "A40": 0.40,
    "A30": 0.30,
    "V100": 0.35,
    "V100 SXM2": 0.50,
    "L40S": 0.90,
    "L40": 0.70,
    "Tesla T4": 0.15,
}

# Vast.ai instance actual_status strings → JobStatus mapping.
_STATUS_MAP: dict[str, JobStatus] = {
    "loading": JobStatus.PENDING,
    "provisioning": JobStatus.PENDING,
    "created": JobStatus.PENDING,
    "running": JobStatus.RUNNING,
    "exited": JobStatus.COMPLETED,
    "offline": JobStatus.FAILED,
    "destroyed": JobStatus.CANCELLED,
    "error": JobStatus.FAILED,
    "stopped": JobStatus.CANCELLED,
}

# Default PyTorch base image used when no container_image is specified.
_DEFAULT_IMAGE = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_vastai() -> Any:
    """Import and return the ``vastai`` module, raising a clear error if absent.

    Returns:
        The ``vastai`` module object.

    Raises:
        ImportError: With install instructions when the SDK is absent.
    """
    try:
        import vastai  # lazy import — intentional

        return vastai
    except ImportError as exc:
        raise ImportError(
            "The Vast.ai backend requires the vastai SDK; "
            "install it with:  pip install rfdf[compute-vastai]"
        ) from exc


def _make_client(vastai: Any) -> Any:
    """Construct a VastAI SDK client, raising a clear error if the key is missing.

    The SDK's ``_resolve_api_key`` reads from (in order) the ``VAST_API_KEY``
    environment variable, ``$XDG_CONFIG_HOME/vastai/vast_api_key``, and the
    legacy ``~/.vast_api_key`` file.  A missing key causes the SDK to raise
    ``RuntimeError``; we catch that and re-raise with a clearer message.

    Args:
        vastai: The imported ``vastai`` module.

    Returns:
        A configured :class:`vastai.VastAI` client instance.

    Raises:
        RuntimeError: If no API key is found in any of the three search locations.
    """
    # Allow the env var to override the file-based key explicitly.
    api_key: str | None = os.environ.get("VAST_API_KEY", "").strip() or None
    try:
        if api_key:
            return vastai.VastAI(api_key=api_key, raw=True, quiet=True)
        # Let the SDK resolve from ~/.vast_api_key / XDG config.
        return vastai.VastAI(raw=True, quiet=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "VastAiCompute: no Vast.ai API key found.  Set VAST_API_KEY in "
            "the environment, or save your key to ~/.config/vastai/vast_api_key "
            "(or the legacy ~/.vast_api_key)."
        ) from exc


def _build_search_query(job: ComputeJob) -> str:
    """Build a Vast.ai offer search query string for the given job.

    Args:
        job: The compute job whose GPU requirements drive the query.

    Returns:
        A query string accepted by :meth:`vastai.VastAI.search_offers`.
    """
    parts: list[str] = []
    if job.gpu_count and job.gpu_count > 0:
        parts.append(f"num_gpus={job.gpu_count}")
    else:
        parts.append("num_gpus>=1")
    if job.gpu_model:
        # Vast.ai uses the display name for gpu_name filtering.
        safe_name = job.gpu_model.replace(" ", "_")
        parts.append(f"gpu_name={safe_name}")
    if job.gpu_min_vram_gb:
        parts.append(f"gpu_ram>={job.gpu_min_vram_gb}")
    return " ".join(parts)


def _build_onstart_cmd(job: ComputeJob) -> str:
    """Build the onstart command that runs when the Vast.ai instance is ready.

    Vast.ai passes ``onstart`` as a shell command that is executed when the
    container starts.  We use a ``bash -c '<pip-install> && python <script>'``
    pattern when pip requirements are provided.

    Each requirement is double-quoted because the outer command is a single-
    quoted ``bash -c '...'`` in the JSON body — double quotes are safe inside
    single-quoted shell strings and don't break the outer delimiter.

    Args:
        job: The compute job.

    Returns:
        A shell command string suitable for the Vast.ai ``onstart`` parameter.
    """
    if job.container_image:
        # User's image already contains the entry point; run it directly.
        return f"python {job.entry_script}"
    if job.pip_requirements:
        reqs = " ".join(f'"{r}"' for r in job.pip_requirements)
        pip_cmd = f"pip install -q {reqs}"
        return f"bash -c '{pip_cmd} && python {job.entry_script}'"
    return f"bash -c 'python {job.entry_script}'"


def _map_instance_status(instance: dict[str, Any]) -> JobStatus:
    """Map a Vast.ai instance dict to a :class:`JobStatus`.

    Args:
        instance: Raw instance dict returned by the Vast.ai SDK.

    Returns:
        Mapped :class:`JobStatus`.
    """
    actual = (instance.get("actual_status") or "").lower()
    return _STATUS_MAP.get(actual, JobStatus.PENDING)


async def _poll_instance_logs(handle: JobHandle) -> AsyncIterator[str]:
    """Async generator that polls a Vast.ai instance and yields log lines.

    Args:
        handle: The job handle whose instance to poll.

    Yields:
        One status line per poll until the instance terminates.
    """
    vastai = _require_vastai()
    client = _make_client(vastai)
    instance_id = int(handle.job_id)
    poll_interval = 5.0
    while True:
        try:
            instance: dict[str, Any] = await asyncio.to_thread(client.show_instance, instance_id)
        except Exception as exc:
            yield f"[vastai] instance {instance_id} lookup failed: {exc}"
            return
        actual = (instance.get("actual_status") or "unknown").lower()
        yield f"[vastai] instance={instance_id} status={actual}"
        if actual in {"exited", "offline", "destroyed", "error", "stopped"}:
            # Attempt to fetch instance logs before returning.
            try:
                log_text: str = await asyncio.to_thread(client.logs, instance_id, tail="100")
                for line in (log_text or "").splitlines():
                    yield f"[vastai/logs] {line}"
            except Exception:
                pass
            return
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# VastAiCompute
# ---------------------------------------------------------------------------


class VastAiCompute:
    """Implements :class:`rfdf.hal.ComputeBackend` for Vast.ai GPU instances.

    Vast.ai is a **GPU marketplace**: :meth:`submit` searches for the cheapest
    offer matching the job's GPU requirements (via ``search_offers``), creates
    an instance on the winning offer, and runs the entry script via the
    ``onstart`` command.  :meth:`status`, :meth:`logs`, :meth:`cancel`, and
    :meth:`fetch_artifacts` map to the corresponding SDK methods.

    **Marketplace reliability tradeoff:** Vast.ai hosts are third-party
    machines and instances **can be interrupted** if a host goes offline or
    another tenant outbids you.  High-reliability workloads should prefer
    managed cloud backends (RunPod reserved, Modal).  For batch/research
    workloads the lower price floor is typically worth the interruption risk.

    The SDK is imported lazily inside every method that uses it so that
    ``import rfdf.backends.compute.vastai`` never pulls ``vastai`` into
    ``sys.modules``.

    Attributes:
        storage: A :class:`~rfdf.backends.compute._remote_storage.VastAiStorage`
            exposing the backend's per-instance storage interface.
    """

    name: str = "vastai"

    def __init__(self) -> None:
        """Initialise the backend."""
        # Map job_id (str instance ID) → raw instance dict for in-process lookups.
        self._jobs: dict[str, dict[str, Any]] = {}
        self.storage = VastAiStorage()

    @property
    def supports_gpu(self) -> bool:
        """Vast.ai provides GPU instances; always ``True``."""
        return True

    @property
    def supports_persistent_storage(self) -> bool:
        """Vast.ai instances use per-instance local disk (not a shared volume).

        Network volumes exist on Vast.ai but require a separate booking.  This
        backend does not auto-provision a shared volume, so persistent storage
        between separate instance bookings is not guaranteed.
        """
        return False

    async def submit(self, job: ComputeJob) -> JobHandle:
        """Search for the cheapest Vast.ai offer and create an instance for *job*.

        Searches the Vast.ai marketplace for the cheapest on-demand offer that
        satisfies ``job.gpu_model`` / ``job.gpu_count`` / ``job.gpu_min_vram_gb``,
        then calls ``create_instance`` on the top result.  The ``onstart``
        command installs pip requirements (if any) and runs the entry script.

        Args:
            job: The compute job to submit.

        Returns:
            A :class:`JobHandle` with ``backend="vastai"`` and ``job_id`` set to
            the Vast.ai instance ID (as a string).

        Raises:
            RuntimeError: If no Vast.ai API key is available, or if no
                suitable offer is found on the marketplace.
            ImportError: If the ``vastai`` SDK is not installed.
        """
        vastai = _require_vastai()
        client = _make_client(vastai)

        query = _build_search_query(job)
        image = job.container_image or _DEFAULT_IMAGE
        onstart_cmd = _build_onstart_cmd(job)

        # Env vars forwarded to the instance.
        env: dict[str, str] = dict(job.env)
        env["RFDF_JOB_TIMEOUT_S"] = str(int(job.timeout_s))
        if job.artifact_globs:
            env["RFDF_ARTIFACT_GLOBS"] = ",".join(job.artifact_globs)

        # Search for the cheapest matching offer.
        offers: list[dict[str, Any]] = await asyncio.to_thread(
            client.search_offers,
            query=query,
            type="on-demand",
            order="dph_total",
            limit=5,
        )
        if not offers:
            raise RuntimeError(
                f"VastAiCompute: no Vast.ai offers found for query {query!r}. "
                "Relax gpu_model / gpu_min_vram_gb constraints or check "
                "vast.ai availability."
            )

        # Take the cheapest offer (first result with order=dph_total).
        best_offer = offers[0]
        offer_id: int = best_offer["id"]

        label = f"rfdf-{uuid.uuid4().hex[:8]}"
        result: dict[str, Any] = await asyncio.to_thread(
            client.create_instance,
            offer_id,
            image=image,
            disk=10.0,
            env=env,
            onstart_cmd=onstart_cmd,
            label=label,
            cancel_unavail=True,
        )
        # The create_instance response carries the new_contract (instance) ID.
        instance_id: int = result.get("new_contract", 0) or result.get("id", 0)
        job_id = str(instance_id)
        submitted_at = time.time()
        self._jobs[job_id] = result
        return JobHandle(
            backend=self.name,
            job_id=job_id,
            submitted_at_s=submitted_at,
            extra={
                "offer_id": offer_id,
                "label": label,
                "image": image,
                "query": query,
            },
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        """Return the current :class:`JobStatus` for *handle*.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            The current :class:`JobStatus`.

        Raises:
            RuntimeError: If no Vast.ai API key is available.
            ImportError: If the ``vastai`` SDK is not installed.
        """
        vastai = _require_vastai()
        client = _make_client(vastai)
        try:
            instance: dict[str, Any] = await asyncio.to_thread(
                client.show_instance, int(handle.job_id)
            )
        except Exception:
            return JobStatus.CANCELLED
        self._jobs[handle.job_id] = instance
        return _map_instance_status(instance)

    def logs(self, handle: JobHandle) -> AsyncIterator[str]:
        """Yield log lines from the Vast.ai instance.

        Polls the instance status every few seconds and yields progress lines
        until the instance terminates.  When the instance exits, attempts to
        fetch its log output and yields those lines as well.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            An :class:`AsyncIterator` that yields log lines (without trailing
            newline) until the instance terminates.
        """
        return _poll_instance_logs(handle)

    async def fetch_artifacts(self, handle: JobHandle, dest: Path) -> None:
        """Copy job artifacts from the Vast.ai instance into *dest*.

        In a full deployment this would use ``vastai copy`` (SCP-backed) to
        transfer files from the instance.  For the current implementation this
        is a documented stub that operators extend with their preferred transfer
        mechanism (``vastai copy instance_id:/path /local/path``, rsync over
        SSH, DVC pull from cloud storage written by the job).

        Args:
            handle: A handle previously returned by :meth:`submit`.
            dest: Local destination directory (must already exist).

        Raises:
            FileNotFoundError: If *dest* does not exist.
            NotADirectoryError: If *dest* is not a directory.
        """
        _require_vastai()
        if not dest.exists():
            raise FileNotFoundError(f"VastAiCompute.fetch_artifacts: dest {dest} must exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"VastAiCompute.fetch_artifacts: dest {dest} is not a directory"
            )
        # Artifact transfer from a Vast.ai instance requires one of:
        #   - `vastai copy <instance_id>:/path /local/path` (SCP-backed)
        #   - rsync over SSH using the instance's SSH endpoint
        #   - DVC remote pointing at cloud storage written by the job
        # Log the intent and return so calling code can degrade gracefully.
        import structlog

        log = structlog.get_logger(__name__)
        log.warning(
            "vastai_fetch_artifacts_noop",
            job_id=handle.job_id,
            dest=str(dest),
            note=(
                "Artifact transfer requires SSH access to the Vast.ai instance; "
                "use `vastai copy <id>:/path /local/path` or rsync/SCP against "
                "the instance's SSH endpoint."
            ),
        )

    async def cancel(self, handle: JobHandle) -> None:
        """Destroy the Vast.ai instance.  No-op when already destroyed.

        Destruction (not stop) is used to end billing immediately.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Raises:
            RuntimeError: If no Vast.ai API key is available.
            ImportError: If the ``vastai`` SDK is not installed.
        """
        vastai = _require_vastai()
        client = _make_client(vastai)
        current = await self.status(handle)
        if current in {JobStatus.CANCELLED}:
            return
        with contextlib.suppress(Exception):
            await asyncio.to_thread(client.destroy_instance, int(handle.job_id))

    async def cost_estimate(self, job: ComputeJob) -> CostEstimate:
        """Return a pre-submit cost estimate based on static rate data.

        This method is **SDK-free** — it uses only the static ``_RATES`` table
        and never imports the ``vastai`` module.  The contract test suite calls
        it without the SDK installed.

        Formula::

            runtime_h = job.timeout_s / 3600
            gpu_units  = max(job.gpu_count, 1)
            hourly_usd = _RATES.get(job.gpu_model, _RATES[None])

            estimated_usd = hourly_usd * 1.5 * runtime_h * gpu_units
            low_usd       = hourly_usd * 1.0 * runtime_h * gpu_units
            high_usd      = hourly_usd * 2.0 * runtime_h * gpu_units

        The 1.5x multiplier is the expected cost including instance startup
        overhead and modest over-run.  The 1.0x / 2.0x bounds form a 2x spread
        that covers typical variance in job duration and marketplace spot-price
        fluctuations.

        Args:
            job: The compute job to estimate costs for.

        Returns:
            A :class:`CostEstimate` with backend ``"vastai"``.

        Note:
            ``_RATES`` values are indicative marketplace floor prices and may
            drift.  Verify current pricing at `vast.ai
            <https://vast.ai/console/create/>`_ before committing to a budget.
        """
        hourly_rate = _RATES.get(job.gpu_model, _RATES[None])
        runtime_h = job.timeout_s / 3600.0
        gpu_units = max(job.gpu_count, 1)
        estimated = hourly_rate * 1.5 * runtime_h * gpu_units
        low = hourly_rate * 1.0 * runtime_h * gpu_units
        high = hourly_rate * 2.0 * runtime_h * gpu_units
        model_str = job.gpu_model or "default GPU"
        rationale = (
            f"${hourly_rate:.4f}/hr x 1.5 pad x {runtime_h:.2f}h x {gpu_units} GPU(s) "
            f"= ${estimated:.4f} (model={model_str}); "
            f"low=${low:.4f} (1.0x), high=${high:.4f} (2.0x); "
            f"rates are indicative marketplace floors -- verify at vast.ai"
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


def create(**kwargs: Any) -> VastAiCompute:
    """Factory wired into the ``rfdf.backends.compute`` entry-point.

    Import-safe: does NOT import ``vastai``.  The SDK is only touched when a
    method is called on the returned backend instance.

    Args:
        **kwargs: Optional keyword arguments forwarded to :class:`VastAiCompute`.
            Currently no constructor kwargs are defined; reserved for future
            extensions (e.g. ``server_url``).

    Returns:
        Configured :class:`VastAiCompute`.
    """
    return VastAiCompute(**kwargs)


__all__ = ["VastAiCompute", "create"]

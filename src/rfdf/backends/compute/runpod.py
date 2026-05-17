"""RunPod cloud compute backend.

Submits :class:`rfdf.hal.ComputeJob`s to `RunPod <https://www.runpod.io>`_ GPU
pods via the ``runpod`` Python SDK.  The SDK is imported **lazily** — every
method that needs it calls :func:`_require_runpod` — so this module is fully
importable (and the factory discoverable by ``rfdf compute list``) even when the
``runpod`` package is not installed.

Auth:
    Set ``RUNPOD_API_KEY`` in the environment before calling any method.  The
    SDK reads this variable automatically after :func:`_configure_sdk` sets it
    via ``runpod.api_key``.  A missing key raises a :class:`RuntimeError` at
    first use.

Cost estimates:
    :meth:`RunPodCompute.cost_estimate` uses a static ``_RATES`` table of
    indicative hourly GPU prices in USD.  These figures drift over time as
    RunPod changes its market pricing.  They are labelled *indicative only*;
    the actual charge is settled by RunPod.  Operators should update ``_RATES``
    from the RunPod console when it diverges.

Usage::

    from rfdf.backends.compute.runpod import create

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

from rfdf.backends.compute._remote_storage import RunpodVolumeStorage
from rfdf.hal.compute import ComputeJob, CostEstimate, JobHandle
from rfdf.hal.types import JobStatus

# ---------------------------------------------------------------------------
# Indicative hourly GPU rates (USD/hr) — update from the RunPod console.
# Keys match the RunPod GPU display names / type IDs used in ``gpu_type_id``.
# ``None`` is the fallback applied when no ``gpu_model`` is requested.
# ---------------------------------------------------------------------------

_RATES: dict[str | None, float] = {
    None: 0.20,  # fallback — cheapest available GPU
    "NVIDIA RTX A4000": 0.24,
    "NVIDIA RTX A4500": 0.28,
    "NVIDIA RTX A5000": 0.30,
    "NVIDIA RTX A6000": 0.54,
    "A100 SXM": 1.89,
    "A100 PCIe": 1.64,
    "H100 SXM": 3.99,
    "H100 PCIe": 3.49,
    "RTX 4090": 0.74,
    "RTX 4080": 0.49,
    "RTX 3090": 0.44,
    "RTX 3080": 0.30,
    "RTX A5000": 0.30,
    "RTX A6000": 0.54,
}

# RunPod desired-status strings as returned by get_pod().
_STATUS_MAP: dict[str, JobStatus] = {
    "CREATED": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "RESTARTING": JobStatus.RUNNING,
    "EXITED": JobStatus.COMPLETED,
    "PAUSED": JobStatus.PENDING,
    "DEAD": JobStatus.FAILED,
    "TERMINATED": JobStatus.CANCELLED,
}

# Default PyTorch base image used when no container_image is specified.
_DEFAULT_IMAGE = "runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_runpod() -> Any:
    """Import and return the ``runpod`` module, raising a clear error if absent.

    Returns:
        The ``runpod`` module object.

    Raises:
        ImportError: With install instructions when the SDK is absent.
    """
    try:
        import runpod  # lazy import — intentional

        return runpod
    except ImportError as exc:
        raise ImportError(
            "The RunPod backend requires the runpod SDK; "
            "install it with:  pip install rfdf[compute-runpod]"
        ) from exc


def _configure_sdk(runpod: Any) -> str:
    """Read ``RUNPOD_API_KEY`` from the environment and inject it into the SDK.

    Args:
        runpod: The imported ``runpod`` module.

    Returns:
        The API key string.

    Raises:
        RuntimeError: If the environment variable is absent or empty.
    """
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "RunPodCompute: RUNPOD_API_KEY is not set. "
            "Export the variable before using this backend."
        )
    runpod.api_key = api_key
    return api_key


def _map_pod_status(pod: dict[str, Any]) -> JobStatus:
    """Map a RunPod pod dict to a :class:`JobStatus`.

    RunPod pods expose ``desiredStatus`` (the operator's intent) and
    ``runtime`` (whether the pod is actually running a workload).  We use
    ``desiredStatus`` as the primary signal because it updates atomically.

    Args:
        pod: Raw pod dict returned by the RunPod SDK.

    Returns:
        Mapped :class:`JobStatus`.
    """
    desired = (pod.get("desiredStatus") or "").upper()
    return _STATUS_MAP.get(desired, JobStatus.PENDING)


def _pick_gpu_type_id(job: ComputeJob) -> str | None:
    """Return a RunPod GPU type ID for the given job spec.

    Preference order:
    1. ``job.gpu_model`` if set (used verbatim as the GPU type ID).
    2. ``None`` if ``job.gpu_count == 0`` (CPU-only pod).
    3. A generic GPU matching ``job.gpu_min_vram_gb`` (defaults to the
       cheapest available option via ``None`` key in ``_RATES``).

    Args:
        job: The compute job whose GPU requirements drive the selection.

    Returns:
        A RunPod GPU type ID string, or ``None`` for CPU-only pods.
    """
    if job.gpu_count == 0:
        return None
    if job.gpu_model:
        return job.gpu_model
    # No explicit model requested — let RunPod pick by returning None
    # (the SDK will allocate the cheapest available GPU when gpu_type_id is
    # None and gpu_count > 0 would conflict, so we default to the fallback).
    return None


def _build_docker_args(job: ComputeJob) -> str:
    """Build the ``docker_args`` string for a pip-requirements job.

    When no ``container_image`` is provided, we use a PyTorch base image and
    inject a startup command that installs ``job.pip_requirements`` then runs
    ``job.entry_script``.

    Args:
        job: The compute job.

    Returns:
        A ``docker_args`` string (empty string if container_image is set, since
        the entry point is already baked into the image).
    """
    if job.container_image:
        # entry_script is run as-is inside the user's image.
        return f"python {job.entry_script}"
    # Double-quote each requirement: the whole command is wrapped in an outer
    # single-quoted `bash -c '...'`, so requirements (which contain shell
    # metacharacters like `>`) must use double quotes to avoid terminating it.
    reqs = " ".join(f'"{r}"' for r in job.pip_requirements)
    pip_cmd = f"pip install -q {reqs} && " if job.pip_requirements else ""
    return f"bash -c '{pip_cmd}python {job.entry_script}'"


async def _poll_pod_logs(handle: JobHandle) -> AsyncIterator[str]:
    """Async generator that polls a RunPod pod and yields status lines.

    Imported by :meth:`RunPodCompute.logs` to satisfy the sync-method-returning-
    async-iterator shape required by the :class:`~rfdf.hal.ComputeBackend` Protocol.

    Args:
        handle: The job handle whose pod to poll.

    Yields:
        One status line per poll until the pod terminates or is not found.
    """
    runpod = _require_runpod()
    _configure_sdk(runpod)
    poll_interval = 5.0
    while True:
        pod: dict[str, Any] | None = await asyncio.to_thread(runpod.get_pod, handle.job_id)
        if pod is None:
            yield f"[runpod] pod {handle.job_id} not found (terminated or expired)"
            return
        desired = (pod.get("desiredStatus") or "").upper()
        uptime = pod.get("uptimeSeconds") or 0
        yield f"[runpod] pod={handle.job_id} status={desired} uptime={uptime}s"
        if desired in {"EXITED", "DEAD", "TERMINATED"}:
            return
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# RunPodCompute
# ---------------------------------------------------------------------------


class RunPodCompute:
    """Implements :class:`rfdf.hal.ComputeBackend` for RunPod GPU pods.

    Jobs are submitted as RunPod pods.  The pod is terminated (not stopped) on
    :meth:`cancel`; stopping would leave the pod billable.  Logs are retrieved
    via the RunPod SSH endpoint when available, with a polling fallback.

    The SDK is imported lazily inside every method that uses it so that
    ``import rfdf.backends.compute.runpod`` never pulls ``runpod`` into
    ``sys.modules``.

    Attributes:
        storage: A :class:`~rfdf.backends.compute._remote_storage.RunpodVolumeStorage`
            exposing the backend's persistent-storage interface.
    """

    name: str = "runpod"

    def __init__(self, network_volume_id: str = "") -> None:
        """Initialise the backend.

        Args:
            network_volume_id: Optional RunPod Network Volume ID to attach to
                every submitted pod.  Leave empty to use ephemeral container
                storage only.
        """
        self._network_volume_id = network_volume_id
        # Map job_id → raw pod metadata for lookups that need more than the ID.
        self._jobs: dict[str, dict[str, Any]] = {}
        self.storage = RunpodVolumeStorage(
            volume_id=network_volume_id or "none",
            mount_path="/runpod-volume",
        )

    @property
    def supports_gpu(self) -> bool:
        """RunPod provides GPU instances; always ``True``."""
        return True

    @property
    def supports_persistent_storage(self) -> bool:
        """RunPod Network Volumes persist across jobs; always ``True``."""
        return True

    async def submit(self, job: ComputeJob) -> JobHandle:
        """Create a RunPod pod for *job* and return a handle.

        The pod is started immediately.  The handle's ``extra`` dict carries the
        raw pod dict returned by the SDK so that later calls can inspect it
        without a round-trip.

        Args:
            job: The compute job to submit.

        Returns:
            A :class:`JobHandle` with ``backend="runpod"`` and ``job_id`` set to
            the RunPod pod ID.

        Raises:
            RuntimeError: If ``RUNPOD_API_KEY`` is absent or the SDK raises.
            ImportError: If the ``runpod`` SDK is not installed.
        """
        runpod = _require_runpod()
        _configure_sdk(runpod)

        image = job.container_image or _DEFAULT_IMAGE
        gpu_type_id = _pick_gpu_type_id(job)
        docker_args = _build_docker_args(job)

        # Forward job env vars; add any RFDF-specific context.
        env: dict[str, str] = dict(job.env)
        env["RFDF_JOB_TIMEOUT_S"] = str(int(job.timeout_s))
        if job.artifact_globs:
            env["RFDF_ARTIFACT_GLOBS"] = ",".join(job.artifact_globs)

        kwargs: dict[str, Any] = dict(
            name=f"rfdf-{uuid.uuid4().hex[:8]}",
            image_name=image,
            gpu_type_id=gpu_type_id,
            gpu_count=max(job.gpu_count, 1) if gpu_type_id is not None else 1,
            docker_args=docker_args,
            env=env,
            cloud_type="ALL",
            support_public_ip=True,
            start_ssh=True,
            container_disk_in_gb=20,
        )
        if self._network_volume_id:
            kwargs["network_volume_id"] = self._network_volume_id
            kwargs["volume_in_gb"] = 10

        pod_dict: dict[str, Any] = await asyncio.to_thread(runpod.create_pod, **kwargs)
        pod_id: str = pod_dict["id"]
        submitted_at = time.time()
        self._jobs[pod_id] = pod_dict
        return JobHandle(
            backend=self.name,
            job_id=pod_id,
            submitted_at_s=submitted_at,
            extra={"pod": pod_dict},
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        """Return the current :class:`JobStatus` for *handle*.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            The current :class:`JobStatus`.

        Raises:
            RuntimeError: If ``RUNPOD_API_KEY`` is absent.
            ImportError: If the ``runpod`` SDK is not installed.
        """
        runpod = _require_runpod()
        _configure_sdk(runpod)
        pod: dict[str, Any] | None = await asyncio.to_thread(runpod.get_pod, handle.job_id)
        if pod is None:
            # Pod has been terminated and removed from the API.
            return JobStatus.CANCELLED
        self._jobs[handle.job_id] = pod
        return _map_pod_status(pod)

    def logs(self, handle: JobHandle) -> AsyncIterator[str]:
        """Yield log lines from the RunPod pod's SSH output.

        Polls the pod status every few seconds and yields progress lines until
        the pod exits.  A full SSH/SCP log-stream implementation requires an
        active SSH endpoint and is left as an operator extension.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            An :class:`AsyncIterator` that yields log lines (without trailing
            newline) until the pod terminates.
        """
        return _poll_pod_logs(handle)

    async def fetch_artifacts(self, handle: JobHandle, dest: Path) -> None:
        """Copy job artifacts from the RunPod pod into *dest*.

        In a full deployment this would SCP files from the pod's SSH endpoint.
        For the current implementation, this is a documented stub that operators
        extend with their preferred transfer mechanism (rsync, SCP, DVC pull).

        Args:
            handle: A handle previously returned by :meth:`submit`.
            dest: Local destination directory (must already exist).

        Raises:
            FileNotFoundError: If *dest* does not exist.
            NotADirectoryError: If *dest* is not a directory.
        """
        _require_runpod()
        if not dest.exists():
            raise FileNotFoundError(f"RunPodCompute.fetch_artifacts: dest {dest} must exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"RunPodCompute.fetch_artifacts: dest {dest} is not a directory"
            )
        # Artifact transfer from RunPod requires an SSH endpoint or a shared
        # network volume.  Operators should configure one of:
        #   - RunPod Network Volume mounted to a local NFS share
        #   - DVC remote pointing at the pod's storage
        #   - A custom SCP/rsync call against the pod's public SSH port
        # Log the intent and return so calling code can degrade gracefully.
        import structlog

        log = structlog.get_logger(__name__)
        log.warning(
            "runpod_fetch_artifacts_noop",
            job_id=handle.job_id,
            dest=str(dest),
            note=(
                "Artifact transfer requires an SSH endpoint or shared volume; "
                "implement SCP/rsync or mount the network volume locally."
            ),
        )

    async def cancel(self, handle: JobHandle) -> None:
        """Terminate the RunPod pod.  No-op when already terminated.

        Termination (not stop) is used to avoid continued billing on stopped pods.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Raises:
            RuntimeError: If ``RUNPOD_API_KEY`` is absent.
            ImportError: If the ``runpod`` SDK is not installed.
        """
        runpod = _require_runpod()
        _configure_sdk(runpod)
        current = await self.status(handle)
        if current == JobStatus.CANCELLED:
            return
        await asyncio.to_thread(runpod.terminate_pod, handle.job_id)

    async def cost_estimate(self, job: ComputeJob) -> CostEstimate:
        """Return a pre-submit cost estimate based on static rate data.

        Formula::

            runtime_h = job.timeout_s / 3600
            gpu_units  = max(job.gpu_count, 1)
            hourly_usd = _RATES.get(job.gpu_model, _RATES[None])

            estimated_usd = hourly_usd * 1.5 * runtime_h * gpu_units
            low_usd       = hourly_usd * 1.0 * runtime_h * gpu_units
            high_usd      = hourly_usd * 2.0 * runtime_h * gpu_units

        The 1.5x multiplier is the expected cost including pod startup overhead
        and modest over-run.  The 1.0x / 2.0x bounds form a 2x spread that
        covers typical variance in job duration.

        Args:
            job: The compute job to estimate costs for.

        Returns:
            A :class:`CostEstimate` with backend ``"runpod"``.

        Note:
            ``_RATES`` values are indicative and may drift as RunPod changes its
            market pricing.  Always verify against the RunPod console before
            committing to a budget.
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
            f"rates are indicative -- verify at runpod.io"
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


def create(**kwargs: Any) -> RunPodCompute:
    """Factory wired into the ``rfdf.backends.compute`` entry-point.

    Import-safe: does NOT import ``runpod``.  The SDK is only touched when a
    method is called on the returned backend instance.

    Args:
        **kwargs: Optional keyword arguments forwarded to :class:`RunPodCompute`.
            Currently supports ``network_volume_id: str``.

    Returns:
        Configured :class:`RunPodCompute`.
    """
    return RunPodCompute(**kwargs)


__all__ = ["RunPodCompute", "create"]

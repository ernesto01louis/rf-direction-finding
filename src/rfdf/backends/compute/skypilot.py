"""SkyPilot multi-cloud compute backend.

Submits :class:`rfdf.hal.ComputeJob`s to the cheapest available cloud via
`SkyPilot <https://skypilot.readthedocs.io>`_.  The SDK (``skypilot`` / imported
as ``sky``) is imported **lazily** — every method that needs it calls
:func:`_require_sky` — so this module is fully importable (and the factory
discoverable by ``rfdf compute list``) even when the ``skypilot`` package is not
installed.

Auth:
    SkyPilot uses each cloud's native credential mechanism (AWS ``~/.aws/``, GCP
    service-account JSON, Lambda Cloud token, RunPod API key, etc.).  Credentials
    are configured **outside** this backend via ``sky check``.  If SkyPilot
    reports no enabled clouds, the backend raises a :class:`RuntimeError` at
    first use.

Multi-cloud cheapest-pick:
    SkyPilot's ``optimize_target=OptimizeTarget.COST`` (the default) scans all
    ``sky check``-enabled providers and launches the task on the cheapest
    available resource that satisfies the ``sky.Resources`` spec.  The operator
    never has to choose a cloud manually; adding a new cloud credential (e.g.
    ``sky check aws``) automatically widens the pool.

Cost estimates:
    :meth:`SkyPilotCompute.cost_estimate` uses a static ``_RATES`` table of
    representative GPU hourly prices across the cloud providers SkyPilot
    supports (AWS, GCP, Lambda, RunPod, Vast.ai, etc.).  SkyPilot itself picks
    the cheapest available price at launch time; these figures are *indicative
    ceilings* used for pre-launch budget checks.  Operators should update
    ``_RATES`` when cloud pricing changes.

Usage::

    from rfdf.backends.compute.skypilot import create

    backend = create()
    async with backend:  # not strictly required — no cleanup needed
        handle = await backend.submit(job)
        status = await backend.status(handle)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from rfdf.backends.compute._remote_storage import SkyPilotBucketStorage
from rfdf.hal.compute import ComputeJob, CostEstimate, JobHandle
from rfdf.hal.types import JobStatus

# ---------------------------------------------------------------------------
# Indicative hourly GPU rates (USD/hr) — update when cloud pricing changes.
# SkyPilot optimises to the cheapest available option across providers; these
# are representative ceiling prices for pre-launch estimation only.
# Keys match the accelerator display names used in sky.Resources(accelerators=).
# ``None`` is the fallback applied when no ``gpu_model`` is requested.
# ---------------------------------------------------------------------------

_RATES: dict[str | None, float] = {
    None: 0.20,  # fallback — cheapest available GPU across providers
    # AWS P-series / G-series (spot-equivalent floor)
    "V100": 0.90,
    "T4": 0.35,
    "A10G": 1.10,
    "A100": 3.70,
    "A100-80GB": 4.50,
    "H100": 4.98,
    # GCP (on-demand representative)
    "K80": 0.45,
    "P100": 1.46,
    "P4": 0.60,
    # Lambda Cloud (fixed hourly)
    "A10": 0.60,
    "A6000": 1.10,
    "A100-80GB-SXM": 2.00,
    "H100-SXM": 2.49,
    # RunPod (community cloud)
    "RTX 3090": 0.44,
    "RTX 4090": 0.74,
    "RTX A6000": 0.54,
}

# SkyPilot sky.JobStatus → rfdf JobStatus mapping.
_JOB_STATUS_MAP: dict[str, JobStatus] = {
    "INIT": JobStatus.PENDING,
    "PENDING": JobStatus.PENDING,
    "SETTING_UP": JobStatus.RUNNING,
    "RUNNING": JobStatus.RUNNING,
    "SUCCEEDED": JobStatus.COMPLETED,
    "FAILED": JobStatus.FAILED,
    "FAILED_SETUP": JobStatus.FAILED,
    "FAILED_DRIVER": JobStatus.FAILED,
    "CANCELLED": JobStatus.CANCELLED,
}

# SkyPilot sky.ClusterStatus → rfdf JobStatus mapping (used when job queue is empty).
_CLUSTER_STATUS_MAP: dict[str, JobStatus] = {
    "INIT": JobStatus.PENDING,
    "PENDING": JobStatus.PENDING,
    "UP": JobStatus.RUNNING,
    "STOPPED": JobStatus.CANCELLED,
    "AUTOSTOPPING": JobStatus.RUNNING,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_sky() -> Any:
    """Import and return the ``sky`` module, raising a clear error if absent.

    Returns:
        The ``sky`` module object.

    Raises:
        ImportError: With install instructions when the SDK is absent.
    """
    try:
        import sky  # lazy import — intentional

        return sky
    except ImportError as exc:
        raise ImportError(
            "The SkyPilot backend requires the skypilot SDK; "
            "install it with:  pip install rfdf[compute-skypilot]"
        ) from exc


def _check_clouds(sky: Any) -> None:
    """Verify that at least one cloud provider is enabled via ``sky check``.

    SkyPilot lazily discovers which clouds are available by inspecting cached
    credential checks.  This function calls the lightweight
    ``check_capabilities`` probe to verify at least one compute-capable cloud
    is registered before we try to launch.

    Args:
        sky: The imported ``sky`` module.

    Raises:
        RuntimeError: If no cloud credentials are configured.  Operator action:
            configure at least one cloud credential (e.g. ``sky check aws``)
            and rerun ``sky check`` to validate.
        ImportError: If the ``sky`` SDK is not installed.
    """
    from sky.clouds.cloud import CloudCapability

    try:
        enabled = sky.check.get_cached_enabled_clouds_or_refresh(
            capability=CloudCapability.COMPUTE,
            raise_if_no_cloud_access=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "SkyPilotCompute: unable to query enabled clouds. "
            "Run `sky check` to configure at least one cloud provider."
        ) from exc
    if not enabled:
        raise RuntimeError(
            "SkyPilotCompute: no cloud credentials configured. "
            "Run `sky check` to set up at least one provider "
            "(e.g. `sky check aws` or `sky check gcp`)."
        )


def _build_sky_task(job: ComputeJob, cluster_name: str, sky: Any) -> Any:
    """Build a ``sky.Task`` from a :class:`ComputeJob`.

    The task's ``run`` command either executes the entry script directly (when
    a ``container_image`` is given) or installs pip requirements first via a
    ``bash -c`` wrapper.

    Args:
        job: The compute job to convert.
        cluster_name: Cluster name label (stored as a task tag).
        sky: The imported ``sky`` module.

    Returns:
        A configured ``sky.Task`` instance.
    """
    envs: dict[str, str] = dict(job.env)
    envs["RFDF_JOB_TIMEOUT_S"] = str(int(job.timeout_s))
    if job.artifact_globs:
        envs["RFDF_ARTIFACT_GLOBS"] = ",".join(job.artifact_globs)

    run_cmd = _build_run_command(job)

    # SkyPilot setup installs pip requirements before the run command.
    setup_cmd: str | None = None
    if job.pip_requirements and not job.container_image:
        reqs = " ".join(f'"{r}"' for r in job.pip_requirements)
        setup_cmd = f"pip install -q {reqs}"

    resources = _build_sky_resources(job, sky)

    task = sky.Task(
        name=cluster_name,
        run=run_cmd,
        setup=setup_cmd,
        envs=envs,
        resources=resources,
    )
    if job.container_image:
        task.docker_image = job.container_image
    return task


def _build_run_command(job: ComputeJob) -> str:
    """Return the shell command that runs the job entry script.

    When ``container_image`` is set the entry script runs directly.
    Otherwise a plain ``python <script>`` suffices (pip setup runs separately
    in the SkyPilot ``setup`` step).

    Args:
        job: The compute job.

    Returns:
        A shell command string.
    """
    return f"python {job.entry_script}"


def _build_sky_resources(job: ComputeJob, sky: Any) -> Any:
    """Build a ``sky.Resources`` spec from a :class:`ComputeJob`.

    Translates ``gpu_count`` / ``gpu_model`` / ``gpu_min_vram_gb`` to the
    SkyPilot ``accelerators`` parameter.  CPU-only jobs (``gpu_count == 0``)
    request no accelerators.

    Args:
        job: The compute job.
        sky: The imported ``sky`` module.

    Returns:
        A configured ``sky.Resources`` instance.
    """
    accelerators: str | dict[str, int] | None = None
    if job.gpu_count > 0:
        model = job.gpu_model or "T4"
        count = max(job.gpu_count, 1)
        accelerators = {model: count}

    disk_size: int = 50  # GiB — sensible default for ML workloads
    return sky.Resources(
        accelerators=accelerators,
        disk_size=disk_size,
        use_spot=False,  # prefer on-demand for predictability
    )


def _map_sky_job_status(sky_status: Any) -> JobStatus:
    """Map a ``sky.JobStatus`` enum value to a :class:`JobStatus`.

    Args:
        sky_status: A ``sky.JobStatus`` enum instance or its string value.

    Returns:
        The mapped :class:`JobStatus`.
    """
    key = sky_status.value if hasattr(sky_status, "value") else str(sky_status)
    return _JOB_STATUS_MAP.get(key, JobStatus.PENDING)


def _map_sky_cluster_status(sky_status: Any) -> JobStatus:
    """Map a ``sky.ClusterStatus`` enum value to a :class:`JobStatus`.

    Args:
        sky_status: A ``sky.ClusterStatus`` enum instance or its string value.

    Returns:
        The mapped :class:`JobStatus`.
    """
    key = sky_status.value if hasattr(sky_status, "value") else str(sky_status)
    return _CLUSTER_STATUS_MAP.get(key, JobStatus.PENDING)


async def _stream_sky_logs(handle: JobHandle) -> AsyncIterator[str]:
    """Async generator that polls a SkyPilot cluster and yields log lines.

    Uses ``sky.queue`` to find the active job on the cluster and polls its
    status, yielding human-readable lines.  When the job is complete or the
    cluster disappears, ``sky.tail_logs`` drains any remaining output.

    Args:
        handle: The job handle (``job_id`` is the cluster name, ``extra`` carries
            the SkyPilot job ID).

    Yields:
        One status or log line per iteration until the job terminates.
    """
    sky = _require_sky()
    cluster_name = handle.job_id
    sky_job_id: int | None = handle.extra.get("sky_job_id")
    poll_interval = 5.0

    while True:
        try:
            queue_req = await asyncio.to_thread(sky.queue, cluster_name)
            jobs_list = sky.stream_and_get(queue_req)
        except Exception as exc:
            yield f"[skypilot] queue lookup failed for {cluster_name}: {exc}"
            return

        if not jobs_list:
            yield f"[skypilot] cluster={cluster_name} no active jobs (completed or not started)"
            return

        # Find our job or take the most recent.
        target_job = None
        for j in jobs_list:
            if sky_job_id is not None and j.job_id == sky_job_id:
                target_job = j
                break
        if target_job is None:
            target_job = jobs_list[-1]

        status_val = (
            target_job.status.value
            if hasattr(target_job.status, "value")
            else str(target_job.status)
        )
        yield f"[skypilot] cluster={cluster_name} job={target_job.job_id} status={status_val}"

        if status_val in {"SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED"}:
            # Drain final logs.
            try:
                logs_req = await asyncio.to_thread(
                    sky.tail_logs,
                    cluster_name,
                    job_id=target_job.job_id,
                    follow=False,
                )
                if isinstance(logs_req, str):
                    for line in (logs_req or "").splitlines():
                        yield f"[skypilot/log] {line}"
            except Exception:
                pass
            return

        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# SkyPilotCompute
# ---------------------------------------------------------------------------


class SkyPilotCompute:
    """Implements :class:`rfdf.hal.ComputeBackend` for SkyPilot multi-cloud.

    SkyPilot abstracts over multiple cloud providers (AWS, GCP, Lambda Cloud,
    RunPod, Vast.ai, and others) and picks the **cheapest available** resource
    that satisfies the :class:`ComputeJob` spec at the time of launch.  The
    operator configures which clouds are available via ``sky check``; adding a
    new credential widens the pool without any code change.

    **Execution model:** Each :meth:`submit` call launches a named SkyPilot
    cluster (``rfdf-<uuid8>``), runs the entry script as a ``sky.Task``, and
    immediately tears the cluster down after the task completes (``down=True``
    in ``sky.launch``).  The :attr:`JobHandle.job_id` is the cluster name.
    The numeric SkyPilot job ID (for ``sky.queue`` / ``sky.tail_logs`` lookups)
    is stored in ``handle.extra["sky_job_id"]``.

    **Auth:** SkyPilot reads credentials from each cloud's standard location
    (``~/.aws/``, GCP service-account JSON, Lambda ``~/.lambda_cloud/``, etc.).
    Run ``sky check`` once to validate the setup before using this backend.

    **Cheapest-pick behaviour:** SkyPilot's ``optimize_target=COST`` scans all
    ``sky check``-enabled providers.  If RunPod and AWS are both enabled and
    AWS happens to be cheaper for the requested GPU at that moment, SkyPilot
    picks AWS automatically.

    The SDK is imported lazily inside every method that uses it so that
    ``import rfdf.backends.compute.skypilot`` never pulls ``sky`` into
    ``sys.modules``.

    Attributes:
        storage: A :class:`~rfdf.backends.compute._remote_storage.SkyPilotBucketStorage`
            exposing the backend's cloud object-storage interface.
    """

    name: str = "skypilot"

    def __init__(self, bucket_name: str = "") -> None:
        """Initialise the backend.

        Args:
            bucket_name: Optional SkyPilot-managed bucket name to attach to
                every submitted task.  Leave empty to use ephemeral container
                storage only.
        """
        self._bucket_name = bucket_name
        # Map cluster_name → launch metadata for in-process lookups.
        self._jobs: dict[str, dict[str, Any]] = {}
        self.storage = SkyPilotBucketStorage(bucket_name=bucket_name or "rfdf-skypilot")

    @property
    def supports_gpu(self) -> bool:
        """SkyPilot spans GPU-capable clouds; always ``True``."""
        return True

    @property
    def supports_persistent_storage(self) -> bool:
        """SkyPilot supports cloud object storage (S3, GCS, etc.); always ``True``."""
        return True

    async def submit(self, job: ComputeJob) -> JobHandle:
        """Launch a SkyPilot task and return a handle.

        Creates a ``sky.Task`` from *job*, calls ``sky.launch`` with
        ``optimize_target=COST`` (picks cheapest cloud), and stores the cluster
        name in the :class:`JobHandle`.  The cluster is auto-torn-down after the
        task completes (``down=True``).

        Args:
            job: The compute job to submit.

        Returns:
            A :class:`JobHandle` with ``backend="skypilot"`` and ``job_id`` set
            to the SkyPilot cluster name (``rfdf-<uuid8>``).

        Raises:
            RuntimeError: If no cloud credentials are configured (``sky check``
                returns an empty list).
            ImportError: If the ``sky`` SDK is not installed.
        """
        sky = _require_sky()
        _check_clouds(sky)

        cluster_name = f"rfdf-{uuid.uuid4().hex[:8]}"
        task = await asyncio.to_thread(_build_sky_task, job, cluster_name, sky)

        launch_req = await asyncio.to_thread(
            sky.launch,
            task,
            cluster_name=cluster_name,
            retry_until_up=False,
            down=True,  # auto-teardown after task completes
            optimize_target=sky.OptimizeTarget.COST,
        )
        result = sky.stream_and_get(launch_req)

        sky_job_id: int | None = None
        # result is (job_id, resource_handle) — job_id may be None
        if result is not None and isinstance(result, (list, tuple)) and len(result) >= 1:
            raw_id = result[0]
            sky_job_id = int(raw_id) if raw_id is not None else None

        submitted_at = time.time()
        self._jobs[cluster_name] = {
            "cluster_name": cluster_name,
            "sky_job_id": sky_job_id,
            "submitted_at": submitted_at,
        }
        return JobHandle(
            backend=self.name,
            job_id=cluster_name,
            submitted_at_s=submitted_at,
            extra={
                "cluster_name": cluster_name,
                "sky_job_id": sky_job_id,
            },
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        """Return the current :class:`JobStatus` for *handle*.

        Probes ``sky.queue`` for job-level status on the cluster.  If the
        cluster has already been torn down (auto-down after completion),
        ``sky.status`` is consulted instead.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            The current :class:`JobStatus`.

        Raises:
            RuntimeError: If no cloud credentials are configured.
            ImportError: If the ``sky`` SDK is not installed.
        """
        sky = _require_sky()
        cluster_name = handle.job_id
        sky_job_id: int | None = handle.extra.get("sky_job_id")

        # Try job-level status first via sky.queue.
        try:
            queue_req = await asyncio.to_thread(sky.queue, cluster_name)
            jobs_list = sky.stream_and_get(queue_req)
            if jobs_list:
                target = None
                for j in jobs_list:
                    if sky_job_id is not None and j.job_id == sky_job_id:
                        target = j
                        break
                if target is None:
                    target = jobs_list[-1]
                return _map_sky_job_status(target.status)
        except Exception:
            pass  # cluster may be torn down; fall through to cluster status

        # Fall back to cluster-level status.
        try:
            status_req = await asyncio.to_thread(sky.status, cluster_names=[cluster_name])
            clusters = sky.stream_and_get(status_req)
            if not clusters:
                # Cluster is gone — task completed and was torn down.
                return JobStatus.COMPLETED
            return _map_sky_cluster_status(clusters[0].status)
        except Exception:
            return JobStatus.CANCELLED

    def logs(self, handle: JobHandle) -> AsyncIterator[str]:
        """Yield log lines from the SkyPilot cluster.

        Polls ``sky.queue`` every few seconds and yields status lines until
        the job terminates.  When the job is complete, drains remaining output
        via ``sky.tail_logs``.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Returns:
            An :class:`AsyncIterator` that yields log lines (without trailing
            newline) until the job terminates.
        """
        return _stream_sky_logs(handle)

    async def fetch_artifacts(self, handle: JobHandle, dest: Path) -> None:
        """Copy job artifacts from the SkyPilot cluster into *dest*.

        In a full deployment this would use ``sky.Storage`` or ``sky.exec`` to
        transfer files from the cluster.  For the current implementation this
        is a documented stub that operators extend with their preferred transfer
        mechanism (``sky.Storage`` copy to/from S3/GCS, rsync via
        ``sky exec <cluster> 'cp artifacts/ /mounted/bucket/'``, or DVC pull
        from cloud storage written by the job).

        Args:
            handle: A handle previously returned by :meth:`submit`.
            dest: Local destination directory (must already exist).

        Raises:
            FileNotFoundError: If *dest* does not exist.
            NotADirectoryError: If *dest* is not a directory.
        """
        _require_sky()
        if not dest.exists():
            raise FileNotFoundError(f"SkyPilotCompute.fetch_artifacts: dest {dest} must exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"SkyPilotCompute.fetch_artifacts: dest {dest} is not a directory"
            )
        # Artifact transfer from a SkyPilot cluster requires one of:
        #   - A sky.Storage bucket mounted to the cluster (write by job, read here)
        #   - sky.exec on a live cluster followed by a local rsync
        #   - DVC remote pointing at cloud storage written by the job
        # Log the intent and return so calling code can degrade gracefully.
        import structlog

        log = structlog.get_logger(__name__)
        log.warning(
            "skypilot_fetch_artifacts_noop",
            job_id=handle.job_id,
            dest=str(dest),
            note=(
                "Artifact transfer from a SkyPilot cluster requires a shared "
                "sky.Storage bucket or rsync via sky exec; implement by mounting "
                "cloud storage (S3/GCS) in the sky.Task and pulling the bucket "
                "locally after the task completes."
            ),
        )

    async def cancel(self, handle: JobHandle) -> None:
        """Cancel the SkyPilot job and tear down the cluster.

        Calls ``sky.cancel`` to abort the running job, then ``sky.down`` to
        release cluster resources and stop billing.  No-op when the cluster
        has already been torn down.

        Args:
            handle: A handle previously returned by :meth:`submit`.

        Raises:
            RuntimeError: If no cloud credentials are configured.
            ImportError: If the ``sky`` SDK is not installed.
        """
        sky = _require_sky()
        cluster_name = handle.job_id
        sky_job_id: int | None = handle.extra.get("sky_job_id")

        current = await self.status(handle)
        if current in {JobStatus.CANCELLED, JobStatus.COMPLETED}:
            return

        # Cancel the job first.
        try:
            job_ids_arg = [sky_job_id] if sky_job_id is not None else None
            cancel_req = await asyncio.to_thread(
                sky.cancel,
                cluster_name,
                all=job_ids_arg is None,
                job_ids=job_ids_arg,
            )
            sky.stream_and_get(cancel_req)
        except Exception:
            pass  # best-effort

        # Tear down the cluster to stop billing.
        try:
            down_req = await asyncio.to_thread(sky.down, cluster_name)
            sky.stream_and_get(down_req)
        except Exception:
            pass  # cluster may already be gone

    async def cost_estimate(self, job: ComputeJob) -> CostEstimate:
        """Return a pre-submit cost estimate based on static rate data.

        This method is **SDK-free** — it uses only the static ``_RATES`` table
        and never imports the ``sky`` module.  The contract test suite calls it
        without the SDK installed.

        Formula::

            runtime_h = job.timeout_s / 3600
            gpu_units  = max(job.gpu_count, 1)
            hourly_usd = _RATES.get(job.gpu_model, _RATES[None])

            estimated_usd = hourly_usd * 1.5 * runtime_h * gpu_units
            low_usd       = hourly_usd * 1.0 * runtime_h * gpu_units
            high_usd      = hourly_usd * 2.0 * runtime_h * gpu_units

        The 1.5x multiplier is the expected cost including cluster startup
        overhead and modest over-run.  The 1.0x / 2.0x bounds form a 2x spread
        that covers typical variance in job duration and cloud spot-price
        fluctuations.  The static ``_RATES`` table reflects representative
        on-demand ceiling prices; SkyPilot's actual selected rate may be lower
        (spot, cheaper cloud) or higher (premium region, GPU shortage).

        Args:
            job: The compute job to estimate costs for.

        Returns:
            A :class:`CostEstimate` with backend ``"skypilot"``.

        Note:
            ``_RATES`` values are indicative ceiling prices.  SkyPilot's
            multi-cloud optimiser will often beat these estimates, especially
            when spot instances are enabled.  Verify actual billed costs in
            your cloud console after the first run.
        """
        hourly_rate = _RATES.get(job.gpu_model, _RATES[None])
        runtime_h = job.timeout_s / 3600.0
        gpu_units = max(job.gpu_count, 1)
        estimated = hourly_rate * 1.5 * runtime_h * gpu_units
        low = hourly_rate * 1.0 * runtime_h * gpu_units
        high = hourly_rate * 2.0 * runtime_h * gpu_units
        model_str = job.gpu_model or "default GPU (cheapest available)"
        rationale = (
            f"${hourly_rate:.4f}/hr x 1.5 pad x {runtime_h:.2f}h x {gpu_units} GPU(s) "
            f"= ${estimated:.4f} (model={model_str}); "
            f"low=${low:.4f} (1.0x), high=${high:.4f} (2.0x); "
            f"rates are indicative ceilings -- SkyPilot picks cheapest available cloud"
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


def create(**kwargs: Any) -> SkyPilotCompute:
    """Factory wired into the ``rfdf.backends.compute`` entry-point.

    Import-safe: does NOT import ``sky``.  The SDK is only touched when a
    method is called on the returned backend instance.

    Args:
        **kwargs: Optional keyword arguments forwarded to :class:`SkyPilotCompute`.
            Currently supports ``bucket_name: str``.

    Returns:
        Configured :class:`SkyPilotCompute`.
    """
    return SkyPilotCompute(**kwargs)


__all__ = ["SkyPilotCompute", "create"]

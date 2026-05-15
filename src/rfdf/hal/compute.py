"""``ComputeBackend`` Protocol for ML / batch job dispatch.

A compute backend submits a job (container image OR pip requirements + entry script
plus a working directory + env + GPU spec), reports status, streams logs, and
fetches artifacts on completion. Implementations: ``local`` (Stage 2), ``runpod`` /
``modal`` / ``vastai`` / ``skypilot`` (Stage 4+). All implementations share this
single interface so the training recipe is portable across providers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from rfdf.hal.types import JobStatus


class ComputeJob(BaseModel):
    """Payload submitted to a ``ComputeBackend``.

    Either ``container_image`` (preferred for reproducibility) OR
    ``pip_requirements`` + ``entry_script`` is supplied — never both. Backends
    validate this at submit time and raise ``ValueError`` on conflict.

    Attributes:
        entry_script: Relative path of the script to run inside ``working_dir``.
        container_image: OCI image reference (e.g. ``"python:3.11-slim"``); when
            set, the backend ignores ``pip_requirements``.
        pip_requirements: PEP 440 requirement strings installed into a fresh
            virtual environment before running ``entry_script``.
        working_dir: Local directory uploaded to the backend's execution host.
        env: Environment variables exported to the job process.
        gpu_count: Number of GPUs requested; ``0`` means CPU-only.
        gpu_min_vram_gb: Minimum per-GPU VRAM in GB.
        gpu_model: Optional model preference (e.g. ``"A6000"``).
        timeout_s: Soft wall-clock timeout in seconds.
        priority: Backend-specific priority hint (``"low" | "normal" | "high"``).
        artifact_globs: Glob patterns relative to ``working_dir`` to fetch on
            completion (e.g. ``["models/*.pt", "*.json"]``).
    """

    entry_script: str
    container_image: str | None = None
    pip_requirements: list[str] = Field(default_factory=list)
    working_dir: Path
    env: dict[str, str] = Field(default_factory=dict)
    gpu_count: int = 0
    gpu_min_vram_gb: float = 0.0
    gpu_model: str | None = None
    timeout_s: float = 3600.0
    priority: str = "normal"
    artifact_globs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_image_xor_requirements(self) -> ComputeJob:
        if self.container_image and self.pip_requirements:
            raise ValueError("ComputeJob: provide container_image OR pip_requirements, not both")
        return self


class JobHandle(BaseModel):
    """Opaque identifier returned by ``ComputeBackend.submit``.

    Backends embed whatever data they need to resume status/log/cancel later;
    callers should treat the handle as a black box.

    Attributes:
        backend: Name of the backend that owns this handle.
        job_id: Backend-internal identifier (PID, container ID, provider job ID).
        submitted_at_s: Submission timestamp in epoch seconds.
        extra: Backend-private state (avoid relying on schema).
    """

    backend: str
    job_id: str
    submitted_at_s: float
    extra: dict[str, Any] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    """Pre-submit cost estimate for a ``ComputeJob``.

    Attributes:
        backend: Backend name.
        estimated_usd: Mean expected cost in USD.
        low_usd: 10th-percentile cost (best case).
        high_usd: 90th-percentile cost (worst case).
        rationale: Short human-readable explanation.
    """

    backend: str
    estimated_usd: float
    low_usd: float
    high_usd: float
    rationale: str = ""


@runtime_checkable
class ComputeBackend(Protocol):
    """Submits + monitors compute jobs.

    All methods are async because remote providers (RunPod, Modal, Vast.ai,
    SkyPilot) require network I/O. The local backend implements the same shape
    via ``asyncio.to_thread`` for subprocess management.
    """

    @property
    def name(self) -> str:
        """Stable identifier of this backend (e.g. ``"local"``, ``"runpod"``)."""
        ...

    @property
    def supports_gpu(self) -> bool:
        """True if the backend can satisfy ``gpu_count > 0`` jobs."""
        ...

    @property
    def supports_persistent_storage(self) -> bool:
        """True if artifacts persist beyond the job's lifetime on the backend host."""
        ...

    async def submit(self, job: ComputeJob) -> JobHandle:
        """Submit ``job`` and return a handle for later polling."""
        ...

    async def status(self, handle: JobHandle) -> JobStatus:
        """Return the current ``JobStatus`` for ``handle``."""
        ...

    def logs(self, handle: JobHandle) -> AsyncIterator[str]:
        """Yield log lines until the job terminates.

        Implementations should make this an ``async def`` generator so callers
        can break out cleanly.
        """
        ...

    async def fetch_artifacts(self, handle: JobHandle, dest: Path) -> None:
        """Copy job artifacts matching ``job.artifact_globs`` into ``dest``.

        ``dest`` MUST exist; the backend writes files directly inside it.
        """
        ...

    async def cancel(self, handle: JobHandle) -> None:
        """Cancel the job; no-op if already terminated."""
        ...

    async def cost_estimate(self, job: ComputeJob) -> CostEstimate:
        """Return a pre-submit cost estimate."""
        ...

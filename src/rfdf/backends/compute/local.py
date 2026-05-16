"""Local compute backend (in-process / subprocess).

Runs :class:`ComputeJob`s on the host the caller is running on. The execution
mode is picked automatically by the presence of ``container_image``:

* ``container_image`` set → ``docker run`` subprocess: ``job.working_dir`` is
  bind-mounted at ``/workspace`` and the entry script runs inside the image.
* otherwise → ``python entry_script`` directly in the current interpreter (the
  interpreter is assumed to already carry the job's dependencies; the local
  backend does not build a per-job ``venv``).

GPU detection is best-effort and never imports torch at module load. The local
backend reports ``supports_gpu`` truthy when ``nvidia-smi`` exits 0; the
``cost_estimate`` returns a $0 ledger entry — local compute is "free" against
the budget tracker.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from rfdf.hal.compute import ComputeJob, CostEstimate, JobHandle
from rfdf.hal.types import JobStatus


def _detect_gpu_present() -> bool:
    """Return True when ``nvidia-smi`` exists AND exits 0 quickly."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _build_docker_command(job: ComputeJob) -> list[str]:
    """Build the ``docker run`` argv for a container-image job.

    ``job.working_dir`` is bind-mounted at ``/workspace`` (the container's
    working directory), the job's ``env`` is forwarded with ``-e`` flags, and
    ``--gpus all`` is added when the job requests a GPU. The entry script is
    invoked as ``python <entry_script>`` relative to ``/workspace``.

    Args:
        job: The compute job; ``job.container_image`` must be set.

    Returns:
        The ``docker run`` argument vector.
    """
    assert job.container_image is not None, "_build_docker_command needs a container_image"
    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{job.working_dir.resolve()}:/workspace",
        "-w",
        "/workspace",
    ]
    for key, value in job.env.items():
        cmd += ["-e", f"{key}={value}"]
    if job.gpu_count > 0:
        cmd += ["--gpus", "all"]
    cmd += [job.container_image, "python", job.entry_script]
    return cmd


class LocalCompute:
    """Implements :class:`rfdf.hal.ComputeBackend` for the local machine.

    State for in-flight jobs lives in-memory; restart drops it. That's
    acceptable for development; Stage 4 remote backends persist via the
    provider's API.
    """

    name: str = "local"

    def __init__(self) -> None:
        """Initialise an empty job registry."""
        self._jobs: dict[str, _LocalJobRecord] = {}
        self._gpu = _detect_gpu_present()

    @property
    def supports_gpu(self) -> bool:
        """True when nvidia-smi reported a GPU at backend construction."""
        return self._gpu

    @property
    def supports_persistent_storage(self) -> bool:
        """Local backend writes artifacts to the caller's filesystem."""
        return True

    async def submit(self, job: ComputeJob) -> JobHandle:
        """Spawn a subprocess to run ``job.entry_script`` from ``job.working_dir``.

        When ``job.container_image`` is set the script runs inside a ``docker
        run`` container; otherwise it runs directly in the current interpreter.
        """
        if job.container_image is not None and shutil.which("docker") is None:
            raise RuntimeError(
                "LocalCompute: container_image execution requires the docker CLI "
                "on PATH; none was found"
            )
        if job.gpu_count > 0 and not self._gpu:
            raise RuntimeError(
                f"LocalCompute: job requests {job.gpu_count} GPU(s) but none detected"
            )
        if not job.working_dir.is_dir():
            raise FileNotFoundError(f"LocalCompute: working_dir {job.working_dir} missing")
        entry = job.working_dir / job.entry_script
        if not entry.is_file():
            raise FileNotFoundError(f"LocalCompute: entry_script {entry} missing")
        job_id = uuid.uuid4().hex
        record = _LocalJobRecord(job=job, entry=entry)
        await record.start()
        self._jobs[job_id] = record
        return JobHandle(
            backend=self.name,
            job_id=job_id,
            submitted_at_s=record.started_at_s,
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        """Return the current :class:`JobStatus` for ``handle``."""
        record = self._lookup(handle)
        await record.poll()
        return record.status

    def logs(self, handle: JobHandle) -> AsyncIterator[str]:
        """Yield captured stdout lines until the job terminates."""
        record = self._lookup(handle)
        return record.stream_logs()

    async def fetch_artifacts(self, handle: JobHandle, dest: Path) -> None:
        """Copy artifacts matching ``job.artifact_globs`` from working_dir into dest.

        With no globs (a no-op job), the destination is left untouched.
        """
        record = self._lookup(handle)
        await record.poll()
        if not dest.exists():
            raise FileNotFoundError(f"LocalCompute: dest {dest} must exist")
        if not dest.is_dir():
            raise NotADirectoryError(f"LocalCompute: dest {dest} is not a directory")
        for pattern in record.job.artifact_globs:
            for src in record.job.working_dir.glob(pattern):
                if src.is_file():
                    shutil.copy2(src, dest / src.name)

    async def cancel(self, handle: JobHandle) -> None:
        """Terminate the subprocess (no-op when already finished)."""
        record = self._lookup(handle)
        await record.cancel()

    async def cost_estimate(self, job: ComputeJob) -> CostEstimate:
        """Local compute is free against the budget."""
        return CostEstimate(
            backend=self.name,
            estimated_usd=0.0,
            low_usd=0.0,
            high_usd=0.0,
            rationale="local compute consumes no metered cloud resources",
        )

    def _lookup(self, handle: JobHandle) -> _LocalJobRecord:
        if handle.backend != self.name:
            raise ValueError(f"LocalCompute: handle backend {handle.backend!r} != {self.name!r}")
        record = self._jobs.get(handle.job_id)
        if record is None:
            raise KeyError(f"LocalCompute: unknown job_id {handle.job_id!r}")
        return record


class _LocalJobRecord:
    """In-memory tracker for one subprocess job."""

    def __init__(self, *, job: ComputeJob, entry: Path) -> None:
        """Capture the job spec and a resolved entry-script path."""
        self.job = job
        self.entry = entry
        self.proc: asyncio.subprocess.Process | None = None
        self.status: JobStatus = JobStatus.PENDING
        self.started_at_s: float = 0.0
        self.finished_at_s: float | None = None
        self._stdout_lines: list[str] = []

    async def start(self) -> None:
        """Launch the subprocess (``docker run`` or direct) and switch to RUNNING."""
        import os
        import sys

        if self.job.container_image is None:
            cmd = [sys.executable, str(self.entry)]
            env = {**self.job.env}
        else:
            cmd = _build_docker_command(self.job)
            # The docker CLI needs the caller's PATH / DOCKER_HOST; the job's
            # own env is forwarded into the container via -e flags instead.
            env = {**os.environ}
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.job.working_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.started_at_s = time.time()
        self.status = JobStatus.RUNNING

    async def poll(self) -> None:
        """Update status from the subprocess returncode."""
        if self.proc is None or self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return
        if self.proc.returncode is None:
            # Drain any pending stdout without blocking.
            try:
                line = await asyncio.wait_for(
                    self.proc.stdout.readline() if self.proc.stdout else _empty(),
                    timeout=0.01,
                )
            except TimeoutError:
                return
            if line:
                self._stdout_lines.append(line.decode().rstrip())
            return
        self.finished_at_s = time.time()
        self.status = JobStatus.COMPLETED if self.proc.returncode == 0 else JobStatus.FAILED

    async def cancel(self) -> None:
        """Terminate the process if still running."""
        if self.proc is None or self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=5.0)
        except TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        self.status = JobStatus.CANCELLED

    async def stream_logs(self) -> AsyncIterator[str]:
        """Yield subprocess stdout lines until EOF."""
        if self.proc is None:
            return
        if self.proc.stdout is None:
            return
        # Replay anything we've already buffered.
        for buffered_line in self._stdout_lines:
            yield buffered_line
        # Then stream remaining output.
        while True:
            raw_line = await self.proc.stdout.readline()
            if not raw_line:
                break
            yield raw_line.decode().rstrip()
        await self.proc.wait()
        self.finished_at_s = time.time()
        self.status = JobStatus.COMPLETED if self.proc.returncode == 0 else JobStatus.FAILED


async def _empty() -> bytes:
    """Helper returning empty bytes to short-circuit a missing stdout."""
    return b""


def create(**_: Any) -> LocalCompute:
    """Factory wired into the ``rfdf.backends.compute`` entry-point.

    Returns:
        Configured :class:`LocalCompute`.
    """
    return LocalCompute()


__all__ = ["LocalCompute", "create"]

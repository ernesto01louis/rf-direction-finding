"""Unit tests for the local compute backend."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rfdf.backends.compute.local import create as create_local
from rfdf.hal import ComputeBackend, ComputeJob, JobStatus


def _make_no_op_job(working_dir: Path, *, script_name: str = "run.py") -> ComputeJob:
    """Write a trivial entry script that prints success and exits 0."""
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / script_name).write_text("print('hello from local compute')\n")
    return ComputeJob(entry_script=script_name, working_dir=working_dir)


def test_protocol_conformance() -> None:
    """LocalCompute is a structural ComputeBackend."""
    backend = create_local()
    assert isinstance(backend, ComputeBackend)
    assert backend.name == "local"
    assert backend.supports_persistent_storage is True


def test_submit_runs_subprocess_to_completion(tmp_path: Path) -> None:
    """A simple Python entry script completes with JobStatus.COMPLETED."""
    backend = create_local()
    job = _make_no_op_job(tmp_path)

    async def go() -> JobStatus:
        handle = await backend.submit(job)
        # Poll until the job finishes.
        for _ in range(200):
            status = await backend.status(handle)
            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                return status
            await asyncio.sleep(0.02)
        return await backend.status(handle)

    assert asyncio.run(go()) == JobStatus.COMPLETED


def test_submit_propagates_nonzero_exit_as_failed(tmp_path: Path) -> None:
    """A subprocess that exits non-zero becomes JobStatus.FAILED."""
    backend = create_local()
    (tmp_path / "boom.py").write_text("import sys; sys.exit(2)\n")
    job = ComputeJob(entry_script="boom.py", working_dir=tmp_path)

    async def go() -> JobStatus:
        handle = await backend.submit(job)
        for _ in range(200):
            status = await backend.status(handle)
            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                return status
            await asyncio.sleep(0.02)
        return await backend.status(handle)

    assert asyncio.run(go()) == JobStatus.FAILED


def test_fetch_artifacts_noop_leaves_dest_empty(tmp_path: Path) -> None:
    """A no-op job with no artifact_globs leaves dest empty after fetch."""
    backend = create_local()
    job = _make_no_op_job(tmp_path / "work")
    dest = tmp_path / "out"
    dest.mkdir()

    async def go() -> list[Path]:
        handle = await backend.submit(job)
        for _ in range(200):
            status = await backend.status(handle)
            if status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.02)
        await backend.fetch_artifacts(handle, dest)
        return sorted(dest.iterdir())

    assert asyncio.run(go()) == []


def test_fetch_artifacts_copies_matching_files(tmp_path: Path) -> None:
    """artifact_globs files end up in dest after fetch_artifacts."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "produce.py").write_text(
        "from pathlib import Path\nPath('result.json').write_text('{\"ok\": true}')\n"
    )
    backend = create_local()
    job = ComputeJob(
        entry_script="produce.py",
        working_dir=work,
        artifact_globs=["*.json"],
    )
    dest = tmp_path / "out"
    dest.mkdir()

    async def go() -> list[str]:
        handle = await backend.submit(job)
        for _ in range(200):
            if (await backend.status(handle)) == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.02)
        await backend.fetch_artifacts(handle, dest)
        return sorted(p.name for p in dest.iterdir())

    assert asyncio.run(go()) == ["result.json"]


def test_fetch_artifacts_requires_existing_dir(tmp_path: Path) -> None:
    """fetch_artifacts raises FileNotFoundError when dest is missing."""
    backend = create_local()
    job = _make_no_op_job(tmp_path / "work")

    async def go() -> None:
        handle = await backend.submit(job)
        await asyncio.sleep(0.2)
        await backend.fetch_artifacts(handle, tmp_path / "no-such-dir")

    with pytest.raises(FileNotFoundError):
        asyncio.run(go())


def test_cost_estimate_is_zero(tmp_path: Path) -> None:
    """Local compute costs nothing against the budget tracker."""
    backend = create_local()
    job = _make_no_op_job(tmp_path)
    est = asyncio.run(backend.cost_estimate(job))
    assert est.estimated_usd == pytest.approx(0.0)
    assert est.backend == "local"


def test_missing_entry_script_raises(tmp_path: Path) -> None:
    """A submit with a missing entry_script raises FileNotFoundError."""
    backend = create_local()
    job = ComputeJob(entry_script="nope.py", working_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="entry_script"):
        asyncio.run(backend.submit(job))


def test_handle_backend_mismatch_raises(tmp_path: Path) -> None:
    """Passing a handle for another backend raises ValueError."""
    from rfdf.hal import JobHandle

    backend = create_local()
    bogus = JobHandle(backend="not-local", job_id="x", submitted_at_s=0.0)
    with pytest.raises(ValueError, match="backend"):
        asyncio.run(backend.status(bogus))


def test_compute_job_rejects_image_and_requirements(tmp_path: Path) -> None:
    """ComputeJob raises ValueError when both container + pip_requirements set."""
    with pytest.raises(ValueError, match="container_image"):
        ComputeJob(
            entry_script="run.py",
            working_dir=tmp_path,
            container_image="python:3.11",
            pip_requirements=["numpy"],
        )

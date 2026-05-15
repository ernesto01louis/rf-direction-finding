"""Property-based contract for :class:`rfdf.hal.ComputeBackend`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfdf.hal import ComputeBackend, ComputeJob, JobStatus


def test_isinstance_protocol(compute_factories: list[tuple[str, Callable[..., Any]]]) -> None:
    """Every compute backend is a structural ComputeBackend."""
    assert compute_factories, "no compute backends discovered"
    for name, factory in compute_factories:
        backend = factory()
        assert isinstance(backend, ComputeBackend), name


def test_name_is_nonempty_string(compute_factories: list[tuple[str, Callable[..., Any]]]) -> None:
    """backend.name is a non-empty string identifier."""
    for name, factory in compute_factories:
        backend = factory()
        assert isinstance(backend.name, str) and backend.name, name


def test_no_op_job_reaches_completed(
    compute_factories: list[tuple[str, Callable[..., Any]]],
    tmp_path: Path,
) -> None:
    """A trivial Python script eventually reports JobStatus.COMPLETED."""
    for name, factory in compute_factories:
        backend = factory()
        work = tmp_path / f"work-{name}"
        work.mkdir(exist_ok=True)
        (work / "noop.py").write_text("print('contract ok')\n")
        job = ComputeJob(entry_script="noop.py", working_dir=work)

        async def run(b: Any = backend, j: Any = job) -> JobStatus:
            handle = await b.submit(j)
            for _ in range(200):
                status = await b.status(handle)
                if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                    return status  # type: ignore[no-any-return]
                await asyncio.sleep(0.02)
            return await b.status(handle)  # type: ignore[no-any-return]

        assert asyncio.run(run()) == JobStatus.COMPLETED, name


def test_fetch_artifacts_no_op_leaves_dest_empty(
    compute_factories: list[tuple[str, Callable[..., Any]]],
    tmp_path: Path,
) -> None:
    """A no-op job with no artifact_globs leaves dest empty after fetch."""
    for name, factory in compute_factories:
        backend = factory()
        work = tmp_path / f"work-{name}-noop"
        work.mkdir()
        (work / "run.py").write_text("print('no artifacts')\n")
        job = ComputeJob(entry_script="run.py", working_dir=work)
        dest = tmp_path / f"out-{name}"
        dest.mkdir()

        async def run(b: Any = backend, j: Any = job, d: Path = dest) -> list[str]:
            handle = await b.submit(j)
            for _ in range(200):
                status = await b.status(handle)
                if status == JobStatus.COMPLETED:
                    break
                await asyncio.sleep(0.02)
            await b.fetch_artifacts(handle, d)
            return sorted(p.name for p in d.iterdir())

        assert asyncio.run(run()) == [], name


def test_cost_estimate_is_well_formed(
    compute_factories: list[tuple[str, Callable[..., Any]]],
    tmp_path: Path,
) -> None:
    """cost_estimate returns a CostEstimate with the backend's name + sane bounds."""
    for name, factory in compute_factories:
        backend = factory()
        work = tmp_path / f"work-{name}-cost"
        work.mkdir()
        (work / "run.py").write_text("pass\n")
        job = ComputeJob(entry_script="run.py", working_dir=work)
        est = asyncio.run(backend.cost_estimate(job))
        assert est.backend, name
        assert est.estimated_usd >= 0.0, (name, est.estimated_usd)
        assert est.low_usd <= est.high_usd, name


def _unused_pytest_helper(_p: pytest.MonkeyPatch) -> None:  # pragma: no cover
    pass

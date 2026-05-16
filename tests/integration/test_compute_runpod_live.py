"""Live RunPod connectivity integration tests.

These tests make REAL API calls to RunPod and will incur charges if a pod is
accidentally left running.  They are skipped automatically in CI because the
``RUNPOD_API_KEY`` secret is not exported to PR runners.

Run locally with::

    RUNPOD_API_KEY=<your-key> pytest tests/integration/test_compute_runpod_live.py -v -m integration

.. warning::
    Each test that submits a pod WILL be billed by RunPod.  Prefer running
    :func:`test_list_gpus_returns_nonempty` (read-only) for connectivity checks.
"""

from __future__ import annotations

import asyncio
import os

import pytest

# Skip the entire module when the SDK is absent.
runpod = pytest.importorskip("runpod", reason="runpod SDK not installed")

_HAS_KEY = bool(os.environ.get("RUNPOD_API_KEY", "").strip())


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="RUNPOD_API_KEY not set — live tests skipped")
def test_list_gpus_returns_nonempty() -> None:
    """get_gpus() returns at least one GPU type without error."""
    api_key = os.environ["RUNPOD_API_KEY"]
    runpod.api_key = api_key
    gpus = runpod.get_gpus()
    assert isinstance(gpus, list)
    assert len(gpus) > 0
    # Each entry should have an id field.
    assert all("id" in g for g in gpus), f"Unexpected GPU shape: {gpus[0]}"


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="RUNPOD_API_KEY not set — live tests skipped")
def test_create_and_terminate_pod(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[name-defined]
    """Submit a minimal CPU pod, verify it appears, then terminate it.

    This is the cheapest real round-trip: a CPU-only pod with the minimum
    specs.  The test terminates the pod immediately to minimise billing.
    """
    from rfdf.backends.compute.runpod import create as create_backend
    from rfdf.hal.compute import ComputeJob
    from rfdf.hal.types import JobStatus

    backend = create_backend()

    # Write a trivial entry script.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        wd = Path(tmpdir)
        (wd / "noop.py").write_text("print('live test noop')\n")

        job = ComputeJob(
            entry_script="noop.py",
            working_dir=wd,
            gpu_count=0,  # CPU-only — cheapest available
            timeout_s=300.0,
        )

        async def go() -> None:
            handle = await backend.submit(job)
            assert handle.backend == "runpod"
            assert handle.job_id
            # Immediately cancel to avoid charges.
            await backend.cancel(handle)
            status = await backend.status(handle)
            assert status in {JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.RUNNING}

        asyncio.run(go())

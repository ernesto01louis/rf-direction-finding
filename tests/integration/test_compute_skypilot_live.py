"""Live SkyPilot connectivity integration tests.

These tests make REAL API calls to whatever cloud providers are configured via
``sky check`` and WILL incur charges if a cluster is accidentally left running.
They are skipped automatically in CI because the ``SKYPILOT_TEST_LIVE``
environment variable is not exported to PR runners.

Run locally with::

    SKYPILOT_TEST_LIVE=1 pytest tests/integration/test_compute_skypilot_live.py -v -m integration

.. warning::
    Each test that launches a cluster WILL be billed by the selected cloud
    provider.  Prefer :func:`test_list_enabled_clouds` (read-only) for
    connectivity checks; set a very short ``timeout_s`` and use
    ``gpu_count=0`` (CPU-only) for the cheapest possible round-trip.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

# Skip the entire module when the SDK is absent.
sky = pytest.importorskip("sky", reason="skypilot SDK not installed")

_HAS_LIVE = bool(os.environ.get("SKYPILOT_TEST_LIVE", "").strip())


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_LIVE, reason="SKYPILOT_TEST_LIVE not set — live tests skipped")
def test_list_enabled_clouds() -> None:
    """get_cached_enabled_clouds_or_refresh() returns at least one cloud without error."""
    import sky.check as sky_check
    from sky.clouds.cloud import CloudCapability

    enabled = sky_check.get_cached_enabled_clouds_or_refresh(
        capability=CloudCapability.COMPUTE,
        raise_if_no_cloud_access=False,
    )
    assert isinstance(enabled, list)
    assert len(enabled) > 0, (
        "No enabled clouds found — run `sky check` to configure at least one provider."
    )


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_LIVE, reason="SKYPILOT_TEST_LIVE not set — live tests skipped")
def test_cost_estimate_without_credentials(tmp_path: Path) -> None:  # type: ignore[name-defined]
    """cost_estimate() works without any network call — pure static arithmetic."""
    import tempfile
    from pathlib import Path

    from rfdf.backends.compute.skypilot import create as create_backend
    from rfdf.hal.compute import ComputeJob

    backend = create_backend()

    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        (wd / "noop.py").write_text("print('cost estimate test')\n")
        job = ComputeJob(
            entry_script="noop.py",
            working_dir=wd,
            gpu_count=1,
            gpu_model="T4",
            timeout_s=3600.0,
        )
        est = asyncio.run(backend.cost_estimate(job))
        assert est.backend == "skypilot"
        assert est.estimated_usd >= 0.0
        assert est.low_usd <= est.estimated_usd <= est.high_usd
        assert "skypilot" in est.rationale.lower() or "cheapest" in est.rationale.lower()


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_LIVE, reason="SKYPILOT_TEST_LIVE not set — live tests skipped")
def test_launch_cpu_only_job_and_cancel(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[name-defined]
    """Submit a minimal CPU-only job, verify handle, then cancel immediately.

    This is the cheapest real round-trip: a CPU-only task with the minimum
    specs.  The test cancels the cluster immediately after launch to minimise
    billing.
    """
    import tempfile
    from pathlib import Path

    from rfdf.backends.compute.skypilot import create as create_backend
    from rfdf.hal.compute import ComputeJob
    from rfdf.hal.types import JobStatus

    backend = create_backend()

    with tempfile.TemporaryDirectory() as tmpdir:
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
            assert handle.backend == "skypilot"
            assert handle.job_id.startswith("rfdf-")
            # Cancel immediately to minimise billing.
            await backend.cancel(handle)
            status = await backend.status(handle)
            assert status in {
                JobStatus.CANCELLED,
                JobStatus.COMPLETED,
                JobStatus.RUNNING,
                JobStatus.PENDING,
            }

        asyncio.run(go())

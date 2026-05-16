"""Live Vast.ai connectivity integration tests.

These tests make REAL API calls to Vast.ai and will incur charges if an
instance is accidentally left running.  They are skipped automatically in CI
because the ``VAST_API_KEY`` secret is not exported to PR runners.

Run locally with::

    VAST_API_KEY=<your-key> pytest tests/integration/test_compute_vastai_live.py -v -m integration

.. warning::
    Each test that creates an instance WILL be billed by Vast.ai.  Prefer
    running :func:`test_search_offers_returns_nonempty` (read-only) for
    connectivity checks.  All instance-creating tests call
    ``backend.cancel(handle)`` immediately after submission to minimise cost.
"""

from __future__ import annotations

import asyncio
import os

import pytest

# Skip the entire module when the SDK is absent.
vastai = pytest.importorskip("vastai", reason="vastai SDK not installed")

_HAS_KEY = bool(os.environ.get("VAST_API_KEY", "").strip())


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="VAST_API_KEY not set — live tests skipped")
def test_search_offers_returns_nonempty() -> None:
    """search_offers() returns at least one GPU offer without error."""
    client = vastai.VastAI(raw=True, quiet=True)
    offers = client.search_offers(
        query="num_gpus>=1",
        type="on-demand",
        order="dph_total",
        limit=5,
    )
    assert isinstance(offers, list), f"Expected list, got {type(offers)}"
    assert len(offers) > 0, "No offers returned — Vast.ai marketplace may be empty"
    # Each offer should have an id and dph_total.
    assert all("id" in o for o in offers), f"Unexpected offer shape: {offers[0]}"


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="VAST_API_KEY not set — live tests skipped")
def test_cost_estimate_sdk_free() -> None:
    """cost_estimate works without touching the Vast.ai API (pure arithmetic)."""
    import tempfile
    from pathlib import Path

    from rfdf.backends.compute.vastai import create as create_backend
    from rfdf.hal.compute import ComputeJob

    backend = create_backend()

    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        (wd / "noop.py").write_text("print('live test noop')\n")

        job = ComputeJob(
            entry_script="noop.py",
            working_dir=wd,
            gpu_count=1,
            gpu_model="RTX 3090",
            timeout_s=300.0,
        )

        est = asyncio.run(backend.cost_estimate(job))
        assert est.backend == "vastai"
        assert est.low_usd <= est.estimated_usd <= est.high_usd
        assert est.rationale


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="VAST_API_KEY not set — live tests skipped")
def test_create_and_destroy_instance() -> None:
    """Submit a minimal instance, verify it appears, then destroy it immediately.

    This is the cheapest real round-trip: a single-GPU instance with the
    minimum specs and an immediate cancel.  The test destroys the instance
    immediately to minimise billing.
    """
    import tempfile
    from pathlib import Path

    from rfdf.backends.compute.vastai import create as create_backend
    from rfdf.hal.compute import ComputeJob
    from rfdf.hal.types import JobStatus

    backend = create_backend()

    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        (wd / "noop.py").write_text("print('live test noop')\n")

        job = ComputeJob(
            entry_script="noop.py",
            working_dir=wd,
            gpu_count=1,
            timeout_s=300.0,
        )

        async def go() -> None:
            handle = await backend.submit(job)
            assert handle.backend == "vastai"
            assert handle.job_id
            # Immediately cancel to avoid charges.
            await backend.cancel(handle)
            status = await backend.status(handle)
            assert status in {
                JobStatus.CANCELLED,
                JobStatus.COMPLETED,
                JobStatus.RUNNING,
                JobStatus.PENDING,
            }

        asyncio.run(go())

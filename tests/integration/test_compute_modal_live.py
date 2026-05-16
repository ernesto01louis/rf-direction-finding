"""Live Modal connectivity integration tests.

These tests make REAL API calls to Modal and may incur charges if a sandbox is
accidentally left running.  They are skipped automatically in CI because the
``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` secrets are not exported to PR runners.

Run locally with::

    MODAL_TOKEN_ID=<id> MODAL_TOKEN_SECRET=<secret> \
        pytest tests/integration/test_compute_modal_live.py -v -m integration

.. warning::
    Each test that creates a sandbox WILL be billed by Modal.  Prefer running
    connectivity-only checks (list environments / check config) when available.
    Each sandbox is terminated immediately after the status check to minimise
    billing.
"""

from __future__ import annotations

import asyncio
import os

import pytest

# Skip the entire module when the SDK is absent.
modal = pytest.importorskip("modal", reason="modal SDK not installed")

_HAS_CREDS = bool(
    os.environ.get("MODAL_TOKEN_ID", "").strip()
    and os.environ.get("MODAL_TOKEN_SECRET", "").strip()
)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_CREDS, reason="MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set")
def test_credentials_are_readable() -> None:
    """Modal SDK can read the configured credentials without error."""
    from modal.config import config as modal_cfg

    token_id = modal_cfg.get("token_id") or os.environ.get("MODAL_TOKEN_ID", "")
    token_secret = modal_cfg.get("token_secret") or os.environ.get("MODAL_TOKEN_SECRET", "")
    assert token_id, "MODAL_TOKEN_ID not visible to the SDK config"
    assert token_secret, "MODAL_TOKEN_SECRET not visible to the SDK config"


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_CREDS, reason="MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set")
def test_create_and_terminate_sandbox(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[name-defined]
    """Submit a minimal CPU sandbox, verify it appears, then terminate it.

    This is the cheapest real round-trip: a CPU-only sandbox running a trivial
    echo command.  The test terminates the sandbox immediately to minimise
    billing.
    """
    from rfdf.backends.compute.modal import create as create_backend
    from rfdf.hal.compute import ComputeJob
    from rfdf.hal.types import JobStatus

    backend = create_backend()

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
            assert handle.backend == "modal"
            assert handle.job_id
            # Immediately cancel to avoid charges.
            await backend.cancel(handle)
            status = await backend.status(handle)
            assert status in {
                JobStatus.CANCELLED,
                JobStatus.COMPLETED,
                JobStatus.RUNNING,
                JobStatus.FAILED,
            }

        asyncio.run(go())

"""Shared pytest fixtures for rfdf tests.

Per-domain fixtures (mock SDR, mock geometry, calibration factories) land
alongside the suites that need them. This file holds the session-scoped
spine: anything that's expensive to build and immutable per session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def tiny_sigmf(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate a 0.1-second / 256-sample single-channel SigMF pair.

    Used by file-replay tests + the demo-no-hardware pipeline smoke test. The
    binary lives under a session-scoped tmp directory so the repo never ships
    sample data (audit-lesson: ``check-added-large-files`` keeps the tree lean).

    Returns:
        ``(meta_path, data_path)`` tuple. Both files exist and are valid.
    """
    out_dir = tmp_path_factory.mktemp("sigmf")
    sample_rate = 2_560.0
    duration_s = 0.1
    num_samples = round(sample_rate * duration_s)
    # CW emitter at +200 Hz; amplitude 0.5.
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    iq = (0.5 * np.exp(1j * 2 * np.pi * 200.0 * t)).astype(np.complex64)
    data_path = out_dir / "tiny.sigmf-data"
    meta_path = out_dir / "tiny.sigmf-meta"
    iq.tofile(data_path)
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:version": "1.0.0",
            "core:num_channels": 1,
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": 868e6,
            }
        ],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path, data_path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.hardware`` tests unless ``RFDF_HARDWARE=1`` is set.

    Stage 5 adds tests that need a physical B210 / AntRunner / GRBL controller.
    They are skipped on every developer machine and in the standard CI jobs;
    the self-hosted hardware runner (see ``.github/workflows/``) sets
    ``RFDF_HARDWARE=1`` so they execute against the attached devices.
    """
    import os

    if os.environ.get("RFDF_HARDWARE") == "1":
        return
    skip_hardware = pytest.mark.skip(reason="needs physical hardware — set RFDF_HARDWARE=1 to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)

"""Hardware-in-the-loop tests for the B210 SDR backend.

Marked ``@pytest.mark.hardware`` — skipped on every machine that does not set
``RFDF_HARDWARE=1`` (see ``tests/conftest.py``). The self-hosted hardware runner
sets that variable and provides the B210 serials via ``RFDF_B210_SERIALS`` (a
comma-separated list). Site-specific serials are never committed to the repo.

These tests exercise the real UHD device lifecycle: open the multi-device
session, verify clock/time lock, capture coherent IQ, and measure pilot-tone
phase repeatability across retunes — the single most important real-hardware
metric for coherent DOA.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

from rfdf.backends.sdr.b210 import create
from rfdf.hal import SdrConfig, SdrSource

pytestmark = pytest.mark.hardware


def _serials() -> list[str]:
    raw = os.environ.get("RFDF_B210_SERIALS", "").strip()
    if not raw:
        pytest.skip("set RFDF_B210_SERIALS to a comma-separated list of B210 serials")
    return [s.strip() for s in raw.split(",") if s.strip()]


def test_b210_is_structural_sdr_source() -> None:
    """The constructed backend satisfies the SdrSource Protocol."""
    sdr = create(serial_numbers=_serials())
    assert isinstance(sdr, SdrSource)


def test_b210_configure_locks_and_calibrates() -> None:
    """configure() opens the session, locks clocks, and recalibrates the pilot."""
    sdr = create(serial_numbers=_serials())
    cfg = SdrConfig(center_freq_hz=868e6, sample_rate_hz=2e6, coherent=True)

    async def run() -> None:
        await sdr.configure(cfg)
        status = await sdr.status()
        assert status["reachable"] is True
        assert status["ref_locked"] is True
        assert status["calibrated"] is True
        await sdr.close()

    asyncio.run(run())


def test_b210_coherent_capture_stream() -> None:
    """A short coherent stream yields well-formed multi-channel blocks."""
    sdr = create(serial_numbers=_serials())
    cfg = SdrConfig(center_freq_hz=868e6, sample_rate_hz=2e6, coherent=True)

    async def run() -> None:
        await sdr.configure(cfg)
        await sdr.start()
        collected = 0
        async for block in sdr.stream():
            assert block.iq.shape[0] == sdr.num_channels
            assert block.iq.dtype == np.complex64
            assert np.all(np.isfinite(block.iq))
            collected += 1
            if collected >= 5:
                await sdr.stop()
                break
        await sdr.close()
        assert collected >= 5

    asyncio.run(run())


def test_b210_pilot_phase_repeatability_over_retunes() -> None:
    """Pilot-tone per-channel phase std stays small across repeated retunes.

    This is the load-bearing real-hardware metric: if the fractional-N PLL
    retune phase is not being calibrated out, this std blows up. The Stage 5
    acceptance target is std < ~1 degree; the operator runs the full 100-retune
    sweep via `rfdf hw selftest`.
    """
    sdr = create(serial_numbers=_serials())
    cfg = SdrConfig(center_freq_hz=868e6, sample_rate_hz=2e6, coherent=True)

    async def run() -> None:
        phases: list[np.ndarray] = []
        for _ in range(10):
            await sdr.configure(cfg)  # each configure() retunes + recalibrates
            calibration = sdr.calibration
            assert calibration is not None
            phases.append(np.angle(calibration.channel_gains))
        await sdr.close()
        # After calibration each retune's correction should be consistent.
        stacked = np.unwrap(np.asarray(phases), axis=0)
        std_deg = float(np.max(np.std(np.degrees(stacked), axis=0)))
        assert std_deg < 5.0, f"pilot phase std {std_deg:.2f} deg too high"

    asyncio.run(run())

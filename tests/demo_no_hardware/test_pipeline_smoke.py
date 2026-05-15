"""Demo-no-hardware pipeline smoke test.

Replaces the Stage 1 placeholder. Wires the four HAL Protocols end-to-end:

* :class:`StaticGeometry` (pentagonal 5-antenna array at lambda/2 for 868 MHz)
* :class:`MockRotator` parked at (0, 0)
* :class:`MockSdr` with three CW emitters at known bearings
* :class:`LocalCompute` for a no-op job

The smoke test runs a 0.05 s capture (100 000 samples at 2 Msps), asserts
shape / dtype / finite-ness on the synthesised IQ, and runs MUSIC on the
streamed block, asserting it recovers the three emitter bearings.

**This test MUST stay green for the rest of the project's life.** It's the
load-bearing principle ("every algorithm must work on synthetic data") made
executable.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from rfdf.backends.geometry.static import create as create_static_geometry
from rfdf.backends.rotator.mock import create as create_mock_rotator
from rfdf.backends.sdr.mock import (
    CWEmitter,
    MockSdrScenario,
)
from rfdf.backends.sdr.mock import (
    create as create_mock_sdr,
)
from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.doa.music import music
from rfdf.dsp.steering import build_manifold
from rfdf.hal import SdrConfig


def test_demo_no_hardware_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Compose all four HAL Protocols, capture, run stub DOA, verify shapes."""
    monkeypatch.chdir(tmp_path)

    # 1. Geometry: pentagonal 5-antenna LPDA cluster at lambda/2 for 868 MHz.
    geometry = create_static_geometry()
    assert geometry.num_antennas == 5

    # 2. Rotator: parked at (0, 0).
    rotator = create_mock_rotator(
        max_speed_deg_per_s=720.0,
        positioning_accuracy_deg=0.0,
        seed=0,
    )
    asyncio.run(rotator.goto(0.0, 0.0))
    az, el = asyncio.run(rotator.position())
    assert az == pytest.approx(0.0)
    assert el == pytest.approx(0.0)

    # 3. SDR: three in-plane CW emitters at azimuths 30, 90, -45 degrees. Each has a
    #    distinct frequency offset so they are temporally uncorrelated — a rank-3
    #    covariance MUSIC can resolve. Elevation is 0 (the cross is a planar array).
    scenario = MockSdrScenario(
        emitters=[
            CWEmitter(azimuth_deg=30.0, elevation_deg=0.0, power_dbm=0.0),
            CWEmitter(azimuth_deg=90.0, elevation_deg=0.0, power_dbm=0.0, freq_offset_hz=10e3),
            CWEmitter(azimuth_deg=-45.0, elevation_deg=0.0, power_dbm=0.0, freq_offset_hz=-15e3),
        ],
        snr_db=20.0,
    )
    sdr = create_mock_sdr(geometry=geometry, scenario=scenario, block_samples=8192, seed=1)
    cfg = SdrConfig(center_freq_hz=868e6, sample_rate_hz=2e6, rx_gain_db=30.0)
    asyncio.run(sdr.configure(cfg))

    # 4. Capture 0.05 s of IQ -> 100 000 samples at 2 Msps.
    recording = asyncio.run(sdr.capture(0.05))
    assert recording.sigmf_meta_path.exists()
    assert recording.sigmf_data_path.exists()
    assert recording.num_samples == 100_000
    assert recording.duration_s == pytest.approx(0.05)

    # 5. Stream a single block to exercise the streaming path too.
    asyncio.run(sdr.start())

    async def grab_block() -> np.ndarray:
        async for block in sdr.stream():
            await sdr.stop()
            return block.iq
        raise AssertionError("no block")

    iq = asyncio.run(grab_block())
    assert iq.shape == (5, 8192)
    assert iq.dtype == np.complex64
    assert np.all(np.isfinite(iq))

    # 6. Real MUSIC over the streamed IQ — the Stage 3 upgrade of the load-bearing
    #    gate. MUSIC must recover all three emitter azimuths from synthetic data.
    positions = asyncio.run(geometry.positions())
    covariance = sample_covariance(iq)
    manifold = build_manifold(positions, np.arange(-180.0, 180.0, 0.5), np.array([0.0]), 868e6)
    estimate = music(covariance, manifold, num_signals=3)
    recovered = sorted(estimate.azimuth_deg)
    for expected, actual in zip([-45.0, 30.0, 90.0], recovered, strict=True):
        assert abs(actual - expected) <= 2.0, f"MUSIC recovered {actual}, expected {expected}"


def test_demo_no_hardware_geometry_change_changes_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing geometry between captures changes the synthesised IQ.

    This is the load-bearing claim that the mock SDR's signal model is
    physically grounded — if changing geometry didn't change IQ, the
    array-factor would be a lie. Stage 3 DOA algorithms depend on this.
    """
    monkeypatch.chdir(tmp_path)
    cfg = SdrConfig(center_freq_hz=868e6, sample_rate_hz=2e6, rx_gain_db=30.0)
    scenario = MockSdrScenario(
        emitters=[CWEmitter(azimuth_deg=45.0, elevation_deg=0.0, power_dbm=0.0)],
        snr_db=30.0,
    )

    # Two distinct geometries — same number of antennas, different positions.
    tight_array = create_static_geometry(
        antennas=[
            (0.0, 0.0, 0.0),
            (0.08, 0.0, 0.0),
            (0.0, 0.08, 0.0),
            (-0.08, 0.0, 0.0),
            (0.0, -0.08, 0.0),
        ]
    )
    loose_array = create_static_geometry(
        antennas=[
            (0.0, 0.0, 0.0),
            (0.17, 0.0, 0.0),
            (0.0, 0.17, 0.0),
            (-0.17, 0.0, 0.0),
            (0.0, -0.17, 0.0),
        ]
    )

    def grab(geometry: object) -> np.ndarray:
        sdr = create_mock_sdr(
            geometry=geometry,  # type: ignore[arg-type]
            scenario=scenario,
            block_samples=1024,
            seed=0,
        )

        async def go() -> np.ndarray:
            await sdr.configure(cfg)
            await sdr.start()
            async for block in sdr.stream():
                await sdr.stop()
                return block.iq.astype(np.complex128)
            raise AssertionError("no block")

        return asyncio.run(go())

    iq_tight = grab(tight_array)
    iq_loose = grab(loose_array)
    assert not np.allclose(iq_tight, iq_loose), (
        "geometry change did not affect IQ — array-factor math is broken"
    )

"""Unit tests for the Doa orchestration class."""

from __future__ import annotations

import asyncio

import pytest

from rfdf.backends.geometry.static import create as create_static_geometry
from rfdf.backends.sdr.mock import CWEmitter, MockSdrScenario
from rfdf.backends.sdr.mock import create as create_mock_sdr
from rfdf.dsp.calibration import load_simulated
from rfdf.dsp.doa.orchestration import Doa
from rfdf.dsp.doa.result import Algorithm, DoaEstimate
from rfdf.dsp.geometry_presets import half_wavelength_spacing, ula
from rfdf.hal import SdrConfig

FREQ = 2.4e9
SPACING = half_wavelength_spacing(FREQ)


def _mock_setup(
    *, azimuths: list[float], snr_db: float = 20.0, num_channels: int = 8, seed: int = 1
) -> tuple[object, object]:
    """Build a configured mock SDR on a ULA plus its geometry controller."""
    positions = ula(num_channels, SPACING)
    geometry = create_static_geometry(antennas=positions.tolist())
    emitters = [
        CWEmitter(azimuth_deg=az, elevation_deg=0.0, power_dbm=0.0, freq_offset_hz=5e3 * index)
        for index, az in enumerate(azimuths)
    ]
    scenario = MockSdrScenario(emitters=emitters, snr_db=snr_db)
    sdr = create_mock_sdr(geometry=geometry, scenario=scenario, block_samples=4096, seed=seed)
    asyncio.run(
        sdr.configure(
            SdrConfig(center_freq_hz=FREQ, sample_rate_hz=2e6, channels=list(range(num_channels)))
        )
    )
    return sdr, geometry


def test_doa_run_recovers_a_bearing() -> None:
    """Doa.run captures from the mock SDR and recovers the emitter bearing."""
    sdr, geometry = _mock_setup(azimuths=[65.0])
    doa = Doa(sdr, geometry, algorithm=Algorithm.MUSIC)  # type: ignore[arg-type]
    estimate = asyncio.run(doa.run(duration_s=0.02, num_signals=1))
    assert isinstance(estimate, DoaEstimate)
    assert estimate.azimuth_deg[0] == pytest.approx(65.0, abs=1.5)


def test_doa_run_auto_estimates_the_source_count() -> None:
    """With num_signals left None, Doa.run estimates the source count via MDL."""
    sdr, geometry = _mock_setup(azimuths=[50.0, 110.0])
    doa = Doa(sdr, geometry, algorithm=Algorithm.MUSIC)  # type: ignore[arg-type]
    estimate = asyncio.run(doa.run(duration_s=0.05))
    assert estimate.num_signals == 2
    recovered = sorted(estimate.azimuth_deg)
    assert recovered[0] == pytest.approx(50.0, abs=2.0)
    assert recovered[1] == pytest.approx(110.0, abs=2.0)


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_doa_run_with_each_algorithm(algorithm: Algorithm) -> None:
    """Every algorithm choice recovers the bearing through Doa.run on a ULA."""
    sdr, geometry = _mock_setup(azimuths=[72.0])
    doa = Doa(sdr, geometry, algorithm=algorithm)  # type: ignore[arg-type]
    estimate = asyncio.run(doa.run(duration_s=0.02, num_signals=1))
    assert estimate.azimuth_deg[0] == pytest.approx(72.0, abs=2.0)


def test_doa_run_applies_a_calibration() -> None:
    """Doa.run applies a supplied calibration to the captured IQ."""
    sdr, geometry = _mock_setup(azimuths=[40.0])
    calibration = load_simulated(ula(8, SPACING), freq_hz=FREQ)
    doa = Doa(sdr, geometry, calibration=calibration, algorithm=Algorithm.MUSIC)  # type: ignore[arg-type]
    estimate = asyncio.run(doa.run(duration_s=0.02, num_signals=1))
    assert estimate.azimuth_deg[0] == pytest.approx(40.0, abs=1.5)

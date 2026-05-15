"""Unit tests for the position-domain synthetic aperture."""

from __future__ import annotations

import numpy as np
import pytest

from rfdf.dsp.calibration import load_simulated
from rfdf.dsp.doa.result import DoaEstimate
from rfdf.dsp.doa.synthetic_aperture import StationCapture, synthetic_aperture_doa
from rfdf.dsp.geometry_presets import half_wavelength_spacing, ula
from rfdf.dsp.steering import steering_vector

FREQ = 5.8e9
SPACING = half_wavelength_spacing(FREQ)
AZ_GRID = np.arange(0.0, 180.0, 0.25)


def _build_captures(
    *,
    true_az: float,
    num_stations: int = 6,
    snr_db: float = 25.0,
    snapshots: int = 2000,
    seed: int = 0,
    rail_span_m: float = 1.5,
) -> list[StationCapture]:
    """Build a morphing-array capture set: one 5-element ULA per rail station."""
    rng = np.random.default_rng(seed)
    base = ula(5, SPACING)
    rail = np.linspace(0.0, rail_span_m, num_stations)
    source = (rng.standard_normal(snapshots) + 1j * rng.standard_normal(snapshots)) / np.sqrt(2.0)
    noise_std = np.sqrt(1.0 / (10.0 ** (snr_db / 10.0)) / 2.0)
    captures: list[StationCapture] = []
    for station in range(num_stations):
        positions = base + np.array([rail[station], 0.0, 0.0])
        response = steering_vector(positions, az_deg=true_az, el_deg=0.0, freq_hz=FREQ)
        pilot_phase = float(rng.uniform(-np.pi, np.pi))
        signal = np.outer(response, source) * np.exp(1j * pilot_phase)
        noise = noise_std * (
            rng.standard_normal((5, snapshots)) + 1j * rng.standard_normal((5, snapshots))
        )
        captures.append(
            StationCapture(positions=positions, iq=signal + noise, pilot_phase_rad=pilot_phase)
        )
    return captures


def test_coherent_fusion_recovers_bearing() -> None:
    """Coherent fusion of 6 stations recovers the source bearing."""
    captures = _build_captures(true_az=72.0, seed=1)
    calibration = load_simulated(ula(5, SPACING), freq_hz=FREQ)
    estimate = synthetic_aperture_doa(
        captures,
        calibration=calibration,
        freq_hz=FREQ,
        az_grid_deg=AZ_GRID,
        algorithm="music",
        fusion="coherent",
        num_signals=1,
    )
    assert isinstance(estimate, DoaEstimate)
    assert estimate.azimuth_deg[0] == pytest.approx(72.0, abs=1.0)


def test_incoherent_fusion_recovers_bearing() -> None:
    """Incoherent fusion recovers the source bearing."""
    captures = _build_captures(true_az=48.0, seed=2)
    calibration = load_simulated(ula(5, SPACING), freq_hz=FREQ)
    estimate = synthetic_aperture_doa(
        captures,
        calibration=calibration,
        freq_hz=FREQ,
        az_grid_deg=AZ_GRID,
        fusion="incoherent",
        num_signals=1,
    )
    assert estimate.azimuth_deg[0] == pytest.approx(48.0, abs=2.0)


def test_block_diagonal_fusion_recovers_bearing() -> None:
    """Block-diagonal fusion recovers the source bearing."""
    captures = _build_captures(true_az=105.0, seed=3)
    calibration = load_simulated(ula(5, SPACING), freq_hz=FREQ)
    estimate = synthetic_aperture_doa(
        captures,
        calibration=calibration,
        freq_hz=FREQ,
        az_grid_deg=AZ_GRID,
        fusion="block-diagonal",
        num_signals=1,
    )
    assert estimate.azimuth_deg[0] == pytest.approx(105.0, abs=2.0)


def test_coherent_fusion_beats_a_single_station() -> None:
    """The 30-element synthetic aperture estimates more precisely than one station."""
    truth = 80.0
    calibration = load_simulated(ula(5, SPACING), freq_hz=FREQ)
    fused_errors: list[float] = []
    single_errors: list[float] = []
    for seed in range(20):
        captures = _build_captures(true_az=truth, snr_db=5.0, snapshots=400, seed=seed)
        fused = synthetic_aperture_doa(
            captures,
            calibration=calibration,
            freq_hz=FREQ,
            az_grid_deg=AZ_GRID,
            fusion="coherent",
            num_signals=1,
        )
        single = synthetic_aperture_doa(
            captures[:1],
            calibration=calibration,
            freq_hz=FREQ,
            az_grid_deg=AZ_GRID,
            fusion="coherent",
            num_signals=1,
        )
        fused_errors.append(fused.azimuth_deg[0] - truth)
        single_errors.append(single.azimuth_deg[0] - truth)
    assert float(np.std(fused_errors)) < float(np.std(single_errors))


def test_synthetic_aperture_rejects_unknown_fusion() -> None:
    """An unknown fusion mode is rejected."""
    captures = _build_captures(true_az=60.0, seed=5)
    calibration = load_simulated(ula(5, SPACING), freq_hz=FREQ)
    with pytest.raises(ValueError, match="fusion"):
        synthetic_aperture_doa(
            captures, calibration=calibration, freq_hz=FREQ, az_grid_deg=AZ_GRID, fusion="bogus"
        )


def test_synthetic_aperture_rejects_empty_capture_list() -> None:
    """At least one station capture is required."""
    calibration = load_simulated(ula(5, SPACING), freq_hz=FREQ)
    with pytest.raises(ValueError, match="StationCapture"):
        synthetic_aperture_doa([], calibration=calibration, freq_hz=FREQ, az_grid_deg=AZ_GRID)

"""Unit tests for the narrowband DOA estimators.

Each estimator is held to the Cramer-Rao bound: its empirical RMSE over a fixed-seed
Monte-Carlo sweep must stay within 3x sqrt(CRLB). Bartlett and MVDR are additionally
cross-checked against pyArgus; MUSIC is cross-checked analytically and against the
parametric estimators (pyArgus's DOA_MUSIC is broken on modern NumPy).
"""

from __future__ import annotations

import numpy as np
import pytest

from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.crlb import compute_crlb
from rfdf.dsp.doa import DoaEstimate
from rfdf.dsp.doa.bartlett import bartlett
from rfdf.dsp.doa.esprit import esprit, unitary_esprit
from rfdf.dsp.doa.music import music
from rfdf.dsp.doa.mvdr import mvdr
from rfdf.dsp.doa.root_music import root_music
from rfdf.dsp.errors import NotULAError, SourceCountError
from rfdf.dsp.geometry_presets import half_wavelength_spacing, planar_cross, ula
from rfdf.dsp.steering import build_manifold, steering_vector

FREQ = 2.4e9
SPACING = half_wavelength_spacing(FREQ)


def _simulate_covariance(
    positions: np.ndarray,  # type: ignore[type-arg]
    *,
    azimuths_deg: list[float],
    snr_db: float,
    snapshots: int,
    seed: int,
    elevations_deg: list[float] | None = None,
) -> np.ndarray:  # type: ignore[type-arg]
    """Synthesise a sample covariance for unit-power uncorrelated Gaussian sources."""
    rng = np.random.default_rng(seed)
    num_channels = positions.shape[0]
    num_sources = len(azimuths_deg)
    elevations = elevations_deg if elevations_deg is not None else [0.0] * num_sources
    manifold = np.stack(
        [
            steering_vector(positions, az_deg=azimuths_deg[i], el_deg=elevations[i], freq_hz=FREQ)
            for i in range(num_sources)
        ],
        axis=1,
    )
    sources = (
        rng.standard_normal((num_sources, snapshots))
        + 1j * rng.standard_normal((num_sources, snapshots))
    ) / np.sqrt(2.0)
    signal = manifold @ sources
    noise_std = np.sqrt(1.0 / (10.0 ** (snr_db / 10.0)) / 2.0)
    noise = noise_std * (
        rng.standard_normal((num_channels, snapshots))
        + 1j * rng.standard_normal((num_channels, snapshots))
    )
    return sample_covariance(signal + noise)


def _fine_manifold(positions: np.ndarray) -> object:  # type: ignore[type-arg]
    """A 0.1-degree azimuth manifold over (0, 180) for grid estimators."""
    return build_manifold(positions, np.arange(0.0, 180.0, 0.1), np.array([0.0]), FREQ)


# --------------------------------------------------------------------------------------
# Basic single-source recovery
# --------------------------------------------------------------------------------------


def test_bartlett_finds_single_source() -> None:
    """Bartlett peaks at a single emitter's bearing."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[70.0], snr_db=20.0, snapshots=4000, seed=1)
    est = bartlett(cov, _fine_manifold(positions), num_signals=1)
    assert isinstance(est, DoaEstimate)
    assert est.azimuth_deg[0] == pytest.approx(70.0, abs=1.0)


def test_mvdr_finds_single_source() -> None:
    """MVDR peaks at a single emitter's bearing."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[110.0], snr_db=20.0, snapshots=4000, seed=2)
    est = mvdr(cov, _fine_manifold(positions), num_signals=1)
    assert est.azimuth_deg[0] == pytest.approx(110.0, abs=1.0)


def test_music_finds_single_source() -> None:
    """MUSIC peaks at a single emitter's bearing."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[55.0], snr_db=20.0, snapshots=4000, seed=3)
    est = music(cov, _fine_manifold(positions), num_signals=1)
    assert est.azimuth_deg[0] == pytest.approx(55.0, abs=0.5)
    assert est.pseudospectrum_db is not None


def test_root_music_finds_single_source() -> None:
    """Root-MUSIC recovers a single emitter's bearing."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[63.0], snr_db=20.0, snapshots=4000, seed=4)
    est = root_music(cov, positions=positions, freq_hz=FREQ, num_signals=1)
    assert est.azimuth_deg[0] == pytest.approx(63.0, abs=0.5)


def test_esprit_finds_single_source() -> None:
    """ESPRIT recovers a single emitter's bearing."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[120.0], snr_db=20.0, snapshots=4000, seed=5)
    est = esprit(cov, positions=positions, freq_hz=FREQ, num_signals=1)
    assert est.azimuth_deg[0] == pytest.approx(120.0, abs=0.5)


def test_unitary_esprit_finds_single_source() -> None:
    """Unitary ESPRIT recovers a single emitter's bearing."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[85.0], snr_db=20.0, snapshots=4000, seed=6)
    est = unitary_esprit(cov, positions=positions, freq_hz=FREQ, num_signals=1)
    assert est.azimuth_deg[0] == pytest.approx(85.0, abs=0.5)


# --------------------------------------------------------------------------------------
# Multi-source resolution
# --------------------------------------------------------------------------------------


def test_music_resolves_two_sources() -> None:
    """MUSIC resolves two well-separated emitters."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(
        positions, azimuths_deg=[60.0, 100.0], snr_db=20.0, snapshots=4000, seed=7
    )
    est = music(cov, _fine_manifold(positions), num_signals=2)
    recovered = sorted(est.azimuth_deg)
    assert recovered[0] == pytest.approx(60.0, abs=1.0)
    assert recovered[1] == pytest.approx(100.0, abs=1.0)


def test_root_music_resolves_two_sources() -> None:
    """Root-MUSIC resolves two well-separated emitters."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(
        positions, azimuths_deg=[65.0, 115.0], snr_db=20.0, snapshots=4000, seed=8
    )
    est = root_music(cov, positions=positions, freq_hz=FREQ, num_signals=2)
    recovered = sorted(est.azimuth_deg)
    assert recovered[0] == pytest.approx(65.0, abs=1.0)
    assert recovered[1] == pytest.approx(115.0, abs=1.0)


# --------------------------------------------------------------------------------------
# CRLB-bounded accuracy (the mathematical-integrity gate)
# --------------------------------------------------------------------------------------


def _rmse_over_trials(estimator: object, positions: np.ndarray, truth_az: float) -> float:  # type: ignore[type-arg]
    """Run an estimator over 120 fixed-seed noise realisations; return the error std."""
    errors = []
    manifold = _fine_manifold(positions)
    for seed in range(120):
        cov = _simulate_covariance(
            positions, azimuths_deg=[truth_az], snr_db=12.0, snapshots=2000, seed=seed
        )
        if estimator in (bartlett, mvdr, music):
            est = estimator(cov, manifold, num_signals=1)  # type: ignore[operator]
        else:
            est = estimator(cov, positions=positions, freq_hz=FREQ, num_signals=1)  # type: ignore[operator]
        errors.append(est.azimuth_deg[0] - truth_az)
    return float(np.std(errors))


@pytest.mark.parametrize("estimator", [bartlett, mvdr, music, root_music, esprit, unitary_esprit])
def test_estimator_meets_crlb_on_ula(estimator: object) -> None:
    """Every estimator's RMSE on a ULA stays within 3x sqrt(CRLB)."""
    positions = ula(7, SPACING)
    truth_az = 65.0
    crlb = compute_crlb(
        positions, freq_hz=FREQ, snr_db=12.0, snapshots=2000, direction_deg=truth_az
    )
    rmse = _rmse_over_trials(estimator, positions, truth_az)
    assert rmse < 3.0 * np.sqrt(crlb), (
        f"RMSE {rmse:.4f} exceeds 3*sqrt(CRLB) {3 * np.sqrt(crlb):.4f}"
    )


def test_music_meets_crlb_on_planar_cross() -> None:
    """MUSIC on the planar cross stays within 3x sqrt(CRLB) in azimuth."""
    positions = planar_cross(SPACING)
    truth_az = 40.0
    crlb = compute_crlb(
        positions, freq_hz=FREQ, snr_db=12.0, snapshots=2000, direction_deg=truth_az
    )
    manifold = build_manifold(positions, np.arange(-180.0, 180.0, 0.1), np.array([0.0]), FREQ)
    errors = []
    for seed in range(120):
        cov = _simulate_covariance(
            positions, azimuths_deg=[truth_az], snr_db=12.0, snapshots=2000, seed=seed
        )
        est = music(cov, manifold, num_signals=1)
        errors.append(est.azimuth_deg[0] - truth_az)
    assert float(np.std(errors)) < 3.0 * np.sqrt(crlb)


# --------------------------------------------------------------------------------------
# Analytic + cross-algorithm verification
# --------------------------------------------------------------------------------------


def test_music_nulls_at_truth_for_a_noiseless_covariance() -> None:
    """A noiseless single-source covariance gives a MUSIC peak exactly at the truth."""
    positions = ula(8, SPACING)
    truth_az = 72.0
    steer = steering_vector(positions, az_deg=truth_az, el_deg=0.0, freq_hz=FREQ)
    cov = np.outer(steer, steer.conj()) + 1e-9 * np.eye(8)
    est = music(cov, _fine_manifold(positions), num_signals=1)
    assert est.azimuth_deg[0] == pytest.approx(truth_az, abs=0.05)


def test_music_root_music_and_esprit_agree() -> None:
    """The three subspace estimators agree on the same data within 0.5 degrees."""
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[78.0], snr_db=20.0, snapshots=8000, seed=9)
    az_music = music(cov, _fine_manifold(positions), num_signals=1).azimuth_deg[0]
    az_root = root_music(cov, positions=positions, freq_hz=FREQ, num_signals=1).azimuth_deg[0]
    az_esprit = esprit(cov, positions=positions, freq_hz=FREQ, num_signals=1).azimuth_deg[0]
    assert abs(az_music - az_root) < 0.5
    assert abs(az_music - az_esprit) < 0.5


def test_bartlett_matches_pyargus() -> None:
    """rfdf Bartlett and pyArgus DOA_Bartlett peak at the same grid index."""
    de = pytest.importorskip("pyargus.directionEstimation")
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[70.0], snr_db=20.0, snapshots=4000, seed=11)
    manifold = build_manifold(positions, np.arange(0.0, 180.0, 0.5), np.array([0.0]), FREQ)
    mine = bartlett(cov, manifold, num_signals=1)
    theirs = np.abs(np.asarray(de.DOA_Bartlett(cov, manifold.matrix.T)))
    assert mine.pseudospectrum_db is not None
    assert int(np.argmax(mine.pseudospectrum_db)) == int(np.argmax(theirs))


def test_mvdr_matches_pyargus() -> None:
    """rfdf MVDR and pyArgus DOA_Capon peak at the same grid index."""
    de = pytest.importorskip("pyargus.directionEstimation")
    positions = ula(8, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[95.0], snr_db=20.0, snapshots=4000, seed=12)
    manifold = build_manifold(positions, np.arange(0.0, 180.0, 0.5), np.array([0.0]), FREQ)
    mine = mvdr(cov, manifold, num_signals=1)
    theirs = np.abs(np.asarray(de.DOA_Capon(cov, manifold.matrix.T)))
    assert mine.pseudospectrum_db is not None
    assert int(np.argmax(mine.pseudospectrum_db)) == int(np.argmax(theirs))


# --------------------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------------------


def test_root_music_rejects_non_ula() -> None:
    """Root-MUSIC raises on a non-ULA geometry."""
    positions = planar_cross(SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[40.0], snr_db=20.0, snapshots=2000, seed=13)
    with pytest.raises(NotULAError):
        root_music(cov, positions=positions, freq_hz=FREQ, num_signals=1)


def test_esprit_rejects_non_ula() -> None:
    """ESPRIT raises on a non-ULA geometry."""
    positions = planar_cross(SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[40.0], snr_db=20.0, snapshots=2000, seed=14)
    with pytest.raises(NotULAError):
        esprit(cov, positions=positions, freq_hz=FREQ, num_signals=1)


def test_music_rejects_too_many_signals() -> None:
    """MUSIC needs fewer signals than channels."""
    positions = ula(5, SPACING)
    cov = _simulate_covariance(positions, azimuths_deg=[60.0], snr_db=20.0, snapshots=2000, seed=15)
    with pytest.raises(SourceCountError):
        music(cov, _fine_manifold(positions), num_signals=5)


def test_music_rejects_malformed_covariance() -> None:
    """A non-square covariance is rejected before any eigendecomposition."""
    positions = ula(8, SPACING)
    with pytest.raises(Exception, match="square"):
        music(np.zeros((8, 3), dtype=np.complex128), _fine_manifold(positions), num_signals=1)

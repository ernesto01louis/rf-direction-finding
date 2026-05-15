"""Unit tests for 2-D MUSIC, wideband DOA, and coherent-source smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from rfdf.dsp.coherent import (
    forward_backward_smoothing,
    forward_spatial_smoothing,
    toeplitz_rectify,
)
from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.doa.music import music
from rfdf.dsp.doa.music_2d import music_2d
from rfdf.dsp.doa.result import Doa2DResult
from rfdf.dsp.doa.wideband import cssm, incoherent_wideband_music
from rfdf.dsp.errors import InvalidCovarianceError
from rfdf.dsp.geometry_presets import half_wavelength_spacing, planar_cross, ula
from rfdf.dsp.steering import build_manifold, steering_vector

FREQ = 2.4e9
SPACING = half_wavelength_spacing(FREQ)


def _covariance(
    positions: np.ndarray,  # type: ignore[type-arg]
    *,
    azel_deg: list[tuple[float, float]],
    snr_db: float,
    snapshots: int,
    seed: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """Sample covariance for unit-power uncorrelated Gaussian sources at (az, el)."""
    rng = np.random.default_rng(seed)
    num_channels = positions.shape[0]
    manifold = np.stack(
        [steering_vector(positions, az_deg=az, el_deg=el, freq_hz=FREQ) for az, el in azel_deg],
        axis=1,
    )
    sources = (
        rng.standard_normal((len(azel_deg), snapshots))
        + 1j * rng.standard_normal((len(azel_deg), snapshots))
    ) / np.sqrt(2.0)
    signal = manifold @ sources
    noise_std = np.sqrt(1.0 / (10.0 ** (snr_db / 10.0)) / 2.0)
    noise = noise_std * (
        rng.standard_normal((num_channels, snapshots))
        + 1j * rng.standard_normal((num_channels, snapshots))
    )
    return sample_covariance(signal + noise)


def _coherent_iq(
    positions: np.ndarray,  # type: ignore[type-arg]
    *,
    azimuths_deg: list[float],
    snr_db: float,
    snapshots: int,
    seed: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """IQ for two perfectly-coherent sources (each a fixed multiple of one waveform)."""
    rng = np.random.default_rng(seed)
    num_channels = positions.shape[0]
    manifold = np.stack(
        [steering_vector(positions, az_deg=az, el_deg=0.0, freq_hz=FREQ) for az in azimuths_deg],
        axis=1,
    )
    base = (rng.standard_normal(snapshots) + 1j * rng.standard_normal(snapshots)) / np.sqrt(2.0)
    coefficients = np.array([1.0 + 0.0j, 0.8 * np.exp(1j * 1.1)])
    sources = np.outer(coefficients, base)
    signal = manifold @ sources
    noise_std = np.sqrt(1.0 / (10.0 ** (snr_db / 10.0)) / 2.0)
    noise = noise_std * (
        rng.standard_normal((num_channels, snapshots))
        + 1j * rng.standard_normal((num_channels, snapshots))
    )
    return signal + noise


def _wideband_iq(
    positions: np.ndarray,  # type: ignore[type-arg]
    *,
    azimuths_deg: list[float],
    snr_db: float,
    snapshots: int,
    seed: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """IQ for wideband (spectrally white) sources."""
    rng = np.random.default_rng(seed)
    num_channels = positions.shape[0]
    manifold = np.stack(
        [steering_vector(positions, az_deg=az, el_deg=0.0, freq_hz=FREQ) for az in azimuths_deg],
        axis=1,
    )
    sources = (
        rng.standard_normal((len(azimuths_deg), snapshots))
        + 1j * rng.standard_normal((len(azimuths_deg), snapshots))
    ) / np.sqrt(2.0)
    signal = manifold @ sources
    noise_std = np.sqrt(1.0 / (10.0 ** (snr_db / 10.0)) / 2.0)
    noise = noise_std * (
        rng.standard_normal((num_channels, snapshots))
        + 1j * rng.standard_normal((num_channels, snapshots))
    )
    return signal + noise


# --------------------------------------------------------------------------------------
# 2-D MUSIC
# --------------------------------------------------------------------------------------


def test_music_2d_recovers_an_elevated_source() -> None:
    """2-D MUSIC on the planar cross recovers an out-of-plane source's (az, el)."""
    positions = planar_cross(SPACING)
    cov = _covariance(positions, azel_deg=[(35.0, 25.0)], snr_db=30.0, snapshots=8000, seed=1)
    manifold = build_manifold(
        positions, np.arange(-180.0, 180.0, 1.0), np.arange(0.0, 61.0, 1.0), FREQ
    )
    result = music_2d(cov, manifold, num_signals=1)
    assert isinstance(result, Doa2DResult)
    assert result.azimuth_deg[0] == pytest.approx(35.0, abs=2.0)
    assert result.elevation_deg[0] == pytest.approx(25.0, abs=5.0)


def test_music_2d_recovers_an_in_plane_source() -> None:
    """2-D MUSIC recovers an in-plane source at the elevation grid's edge."""
    positions = planar_cross(SPACING)
    cov = _covariance(positions, azel_deg=[(60.0, 0.0)], snr_db=30.0, snapshots=8000, seed=2)
    manifold = build_manifold(
        positions, np.arange(-180.0, 180.0, 1.0), np.arange(0.0, 31.0, 1.0), FREQ
    )
    result = music_2d(cov, manifold, num_signals=1)
    assert result.azimuth_deg[0] == pytest.approx(60.0, abs=2.0)
    assert result.elevation_deg[0] == pytest.approx(0.0, abs=3.0)


def test_music_2d_pseudospectrum_is_a_surface() -> None:
    """The 2-D pseudospectrum has the shape of the (azimuth, elevation) grid."""
    positions = planar_cross(SPACING)
    cov = _covariance(positions, azel_deg=[(20.0, 15.0)], snr_db=30.0, snapshots=4000, seed=3)
    az_grid = np.arange(-180.0, 180.0, 2.0)
    el_grid = np.arange(0.0, 46.0, 2.0)
    result = music_2d(cov, build_manifold(positions, az_grid, el_grid, FREQ), num_signals=1)
    assert result.pseudospectrum_db.shape == (az_grid.size, el_grid.size)


def test_music_2d_resolves_two_sources() -> None:
    """2-D MUSIC resolves two well-separated sources in azimuth."""
    positions = planar_cross(SPACING)
    cov = _covariance(
        positions, azel_deg=[(20.0, 10.0), (110.0, 30.0)], snr_db=30.0, snapshots=8000, seed=4
    )
    manifold = build_manifold(
        positions, np.arange(-180.0, 180.0, 1.0), np.arange(0.0, 61.0, 1.0), FREQ
    )
    result = music_2d(cov, manifold, num_signals=2)
    azimuths = sorted(result.azimuth_deg)
    assert azimuths[0] == pytest.approx(20.0, abs=3.0)
    assert azimuths[1] == pytest.approx(110.0, abs=3.0)


# --------------------------------------------------------------------------------------
# Coherent-source smoothing
# --------------------------------------------------------------------------------------


def test_plain_music_fails_on_coherent_sources() -> None:
    """Plain MUSIC cannot resolve two perfectly-coherent sources."""
    positions = ula(8, SPACING)
    cov = sample_covariance(
        _coherent_iq(positions, azimuths_deg=[70.0, 100.0], snr_db=25.0, snapshots=4000, seed=1)
    )
    manifold = build_manifold(positions, np.arange(0.0, 180.0, 0.5), np.array([0.0]), FREQ)
    azimuths = sorted(music(cov, manifold, num_signals=2).azimuth_deg)
    resolved = abs(azimuths[0] - 70.0) < 2.0 and abs(azimuths[1] - 100.0) < 2.0
    assert not resolved


def test_forward_smoothing_resolves_coherent_sources() -> None:
    """Forward spatial smoothing lets MUSIC resolve a coherent pair."""
    positions = ula(8, SPACING)
    cov = sample_covariance(
        _coherent_iq(positions, azimuths_deg=[70.0, 100.0], snr_db=25.0, snapshots=4000, seed=1)
    )
    smoothed = forward_spatial_smoothing(cov, subarray_size=5)
    sub_manifold = build_manifold(
        ula(5, SPACING), np.arange(0.0, 180.0, 0.5), np.array([0.0]), FREQ
    )
    azimuths = sorted(music(smoothed, sub_manifold, num_signals=2).azimuth_deg)
    assert azimuths[0] == pytest.approx(70.0, abs=2.0)
    assert azimuths[1] == pytest.approx(100.0, abs=2.0)


def test_forward_backward_smoothing_resolves_coherent_sources() -> None:
    """Forward-backward smoothing resolves a coherent pair with a larger subarray."""
    positions = ula(8, SPACING)
    cov = sample_covariance(
        _coherent_iq(positions, azimuths_deg=[65.0, 110.0], snr_db=25.0, snapshots=4000, seed=2)
    )
    smoothed = forward_backward_smoothing(cov, subarray_size=6)
    sub_manifold = build_manifold(
        ula(6, SPACING), np.arange(0.0, 180.0, 0.5), np.array([0.0]), FREQ
    )
    azimuths = sorted(music(smoothed, sub_manifold, num_signals=2).azimuth_deg)
    assert azimuths[0] == pytest.approx(65.0, abs=2.0)
    assert azimuths[1] == pytest.approx(110.0, abs=2.0)


def test_toeplitz_rectify_resolves_coherent_sources() -> None:
    """Toeplitz reconstruction restores the rank a coherent pair destroys."""
    positions = ula(8, SPACING)
    cov = sample_covariance(
        _coherent_iq(positions, azimuths_deg=[72.0, 105.0], snr_db=25.0, snapshots=4000, seed=3)
    )
    rectified = toeplitz_rectify(cov)
    manifold = build_manifold(positions, np.arange(0.0, 180.0, 0.5), np.array([0.0]), FREQ)
    azimuths = sorted(music(rectified, manifold, num_signals=2).azimuth_deg)
    assert azimuths[0] == pytest.approx(72.0, abs=2.5)
    assert azimuths[1] == pytest.approx(105.0, abs=2.5)


def test_forward_smoothing_rejects_oversized_subarray() -> None:
    """A subarray larger than the array is rejected."""
    positions = ula(8, SPACING)
    cov = _covariance(positions, azel_deg=[(70.0, 0.0)], snr_db=20.0, snapshots=2000, seed=4)
    with pytest.raises(InvalidCovarianceError, match="subarray"):
        forward_spatial_smoothing(cov, subarray_size=9)


# --------------------------------------------------------------------------------------
# Wideband DOA
# --------------------------------------------------------------------------------------


def test_incoherent_wideband_music_recovers_source() -> None:
    """Incoherent wideband MUSIC recovers a wideband emitter's bearing."""
    positions = ula(8, SPACING)
    iq = _wideband_iq(positions, azimuths_deg=[75.0], snr_db=20.0, snapshots=8192, seed=1)
    estimate = incoherent_wideband_music(
        iq,
        positions=positions,
        num_signals=1,
        sample_rate_hz=2e6,
        center_freq_hz=FREQ,
        az_grid_deg=np.arange(0.0, 180.0, 0.5),
        num_bins=16,
    )
    assert estimate.azimuth_deg[0] == pytest.approx(75.0, abs=1.5)


def test_cssm_recovers_source() -> None:
    """CSSM (coherent signal-subspace focusing) recovers a wideband emitter."""
    positions = ula(8, SPACING)
    iq = _wideband_iq(positions, azimuths_deg=[95.0], snr_db=20.0, snapshots=8192, seed=2)
    estimate = cssm(
        iq,
        positions=positions,
        num_signals=1,
        sample_rate_hz=2e6,
        center_freq_hz=FREQ,
        az_grid_deg=np.arange(0.0, 180.0, 0.5),
        num_bins=16,
    )
    assert estimate.azimuth_deg[0] == pytest.approx(95.0, abs=1.5)


def test_incoherent_wideband_music_resolves_two_sources() -> None:
    """Incoherent wideband MUSIC resolves two wideband emitters."""
    positions = ula(8, SPACING)
    iq = _wideband_iq(positions, azimuths_deg=[60.0, 115.0], snr_db=20.0, snapshots=8192, seed=3)
    estimate = incoherent_wideband_music(
        iq,
        positions=positions,
        num_signals=2,
        sample_rate_hz=2e6,
        center_freq_hz=FREQ,
        az_grid_deg=np.arange(0.0, 180.0, 0.5),
        num_bins=16,
    )
    azimuths = sorted(estimate.azimuth_deg)
    assert azimuths[0] == pytest.approx(60.0, abs=2.0)
    assert azimuths[1] == pytest.approx(115.0, abs=2.0)

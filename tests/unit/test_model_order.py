"""Unit tests for rfdf.dsp.model_order."""

from __future__ import annotations

import numpy as np
import pytest

from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.errors import SourceCountError
from rfdf.dsp.geometry_presets import half_wavelength_spacing, ula
from rfdf.dsp.model_order import aic, estimate_num_signals, mdl, sorte
from rfdf.dsp.steering import steering_vector

FREQ = 2.4e9
SPACING = half_wavelength_spacing(FREQ)


def _covariance(
    positions: np.ndarray,  # type: ignore[type-arg]
    *,
    azimuths_deg: list[float],
    snr_db: float,
    snapshots: int,
    seed: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """Sample covariance for unit-power uncorrelated Gaussian sources."""
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
    return sample_covariance(signal + noise)


@pytest.mark.parametrize("estimator", [aic, mdl])
def test_information_criterion_recovers_three_sources(estimator: object) -> None:
    """AIC and MDL both recover a 3-source count at high SNR."""
    cov = _covariance(
        ula(8, SPACING), azimuths_deg=[40.0, 75.0, 120.0], snr_db=20.0, snapshots=2000, seed=1
    )
    assert estimator(cov, snapshots=2000) == 3  # type: ignore[operator]


def test_mdl_recovers_a_single_source() -> None:
    """MDL recovers a 1-source count."""
    cov = _covariance(ula(8, SPACING), azimuths_deg=[65.0], snr_db=20.0, snapshots=2000, seed=2)
    assert mdl(cov, snapshots=2000) == 1


def test_sorte_recovers_three_sources() -> None:
    """SORTE recovers a 3-source count from the eigenvalue gaps."""
    cov = _covariance(
        ula(8, SPACING), azimuths_deg=[40.0, 75.0, 120.0], snr_db=20.0, snapshots=2000, seed=3
    )
    assert sorte(cov) == 3


def test_mdl_reports_zero_sources_on_noise() -> None:
    """MDL reports zero sources for a noise-only covariance."""
    rng = np.random.default_rng(4)
    noise = (rng.standard_normal((8, 2000)) + 1j * rng.standard_normal((8, 2000))) / np.sqrt(2.0)
    assert mdl(sample_covariance(noise), snapshots=2000) == 0


def test_estimate_num_signals_dispatches_by_method() -> None:
    """estimate_num_signals routes to each named criterion."""
    cov = _covariance(
        ula(8, SPACING), azimuths_deg=[50.0, 100.0], snr_db=20.0, snapshots=2000, seed=5
    )
    assert estimate_num_signals(cov, snapshots=2000, method="mdl") == 2
    assert estimate_num_signals(cov, snapshots=2000, method="aic") == 2
    assert estimate_num_signals(cov, snapshots=2000, method="sorte") == 2


def test_estimate_num_signals_rejects_unknown_method() -> None:
    """An unknown method name is rejected."""
    cov = _covariance(ula(8, SPACING), azimuths_deg=[50.0], snr_db=20.0, snapshots=1000, seed=6)
    with pytest.raises(ValueError, match="method"):
        estimate_num_signals(cov, snapshots=1000, method="bogus")


def test_aic_rejects_nonpositive_snapshots() -> None:
    """AIC needs a positive snapshot count."""
    cov = _covariance(ula(8, SPACING), azimuths_deg=[50.0], snr_db=20.0, snapshots=1000, seed=7)
    with pytest.raises(ValueError, match="snapshots"):
        aic(cov, snapshots=0)


def test_sorte_rejects_tiny_arrays() -> None:
    """SORTE needs at least four channels."""
    cov = _covariance(ula(3, SPACING), azimuths_deg=[50.0], snr_db=20.0, snapshots=1000, seed=8)
    with pytest.raises(SourceCountError, match="SORTE"):
        sorte(cov)

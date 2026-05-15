"""Unit tests for rfdf.dsp.covariance."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rfdf.dsp.covariance import diagonal_load, sample_covariance
from rfdf.dsp.errors import InvalidCovarianceError


def _rank_one_iq(num_channels: int = 5, num_snapshots: int = 2048, seed: int = 0) -> np.ndarray:  # type: ignore[type-arg]
    """Synthesise a single-source (rank-1 signal) IQ block."""
    rng = np.random.default_rng(seed)
    steering = np.exp(1j * rng.uniform(-np.pi, np.pi, size=num_channels))
    source = rng.standard_normal(num_snapshots) + 1j * rng.standard_normal(num_snapshots)
    return np.outer(steering, source)


def test_sample_covariance_shape_and_hermitian() -> None:
    """R has shape (M, M) and is Hermitian."""
    cov = sample_covariance(_rank_one_iq())
    assert cov.shape == (5, 5)
    np.testing.assert_allclose(cov, cov.conj().T, atol=1e-9)


def test_sample_covariance_matches_manual_formula() -> None:
    """sample_covariance equals the explicit (1/N) X Xᴴ definition."""
    iq = _rank_one_iq(num_channels=3, num_snapshots=128)
    np.testing.assert_allclose(sample_covariance(iq), (iq @ iq.conj().T) / iq.shape[1], atol=1e-12)


def test_sample_covariance_of_single_source_is_rank_one() -> None:
    """A single emitter yields one dominant eigenvalue."""
    eigvals = np.sort(np.linalg.eigvalsh(sample_covariance(_rank_one_iq())))[::-1]
    assert eigvals[0] > 1e6 * max(eigvals[1], 1e-12)


def test_sample_covariance_promotes_complex64_to_complex128() -> None:
    """complex64 IQ is promoted; the covariance is always complex128."""
    assert sample_covariance(_rank_one_iq().astype(np.complex64)).dtype == np.complex128


def test_sample_covariance_rejects_1d_input() -> None:
    """A 1-D array is not a valid (M, N) IQ block."""
    with pytest.raises(InvalidCovarianceError, match="M, N"):
        sample_covariance(np.zeros(8, dtype=np.complex128))


def test_sample_covariance_rejects_nonfinite() -> None:
    """A NaN sample raises rather than producing a NaN covariance."""
    iq = _rank_one_iq()
    iq[0, 0] = np.nan
    with pytest.raises(InvalidCovarianceError, match="non-finite"):
        sample_covariance(iq)


def test_sample_covariance_rejects_zero_snapshots() -> None:
    """An empty snapshot axis raises a clear error."""
    with pytest.raises(InvalidCovarianceError, match="snapshot"):
        sample_covariance(np.zeros((5, 0), dtype=np.complex128))


def test_diagonal_load_makes_singular_matrix_invertible() -> None:
    """Loading a rank-deficient covariance restores a finite condition number."""
    loaded = diagonal_load(sample_covariance(_rank_one_iq()), loading=1e-2)
    assert np.isfinite(np.linalg.cond(loaded))
    np.testing.assert_allclose(loaded, loaded.conj().T, atol=1e-9)


def test_diagonal_load_zero_is_a_no_op() -> None:
    """Zero loading returns the covariance unchanged."""
    cov = sample_covariance(_rank_one_iq())
    np.testing.assert_allclose(diagonal_load(cov, loading=0.0), cov, atol=1e-12)


def test_diagonal_load_rejects_negative_loading() -> None:
    """A negative loading factor is meaningless."""
    with pytest.raises(ValueError, match="non-negative"):
        diagonal_load(sample_covariance(_rank_one_iq()), loading=-0.1)


def test_diagonal_load_rejects_nonsquare() -> None:
    """Diagonal loading requires a square covariance."""
    with pytest.raises(InvalidCovarianceError, match="square"):
        diagonal_load(np.zeros((3, 5), dtype=np.complex128))


def test_sample_covariance_rejects_zero_channels() -> None:
    """An empty channel axis raises a clear error."""
    with pytest.raises(InvalidCovarianceError, match="channel"):
        sample_covariance(np.zeros((0, 16), dtype=np.complex128))


@given(st.integers(min_value=2, max_value=8), st.integers(min_value=16, max_value=256))
def test_sample_covariance_is_always_hermitian(num_channels: int, num_snapshots: int) -> None:
    """X Xᴴ is Hermitian for any IQ block."""
    rng = np.random.default_rng(num_channels * 1000 + num_snapshots)
    iq = rng.standard_normal((num_channels, num_snapshots)) + 1j * rng.standard_normal(
        (num_channels, num_snapshots)
    )
    cov = sample_covariance(iq)
    np.testing.assert_allclose(cov, cov.conj().T, atol=1e-9)

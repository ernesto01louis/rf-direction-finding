"""Coherent-source decorrelation: spatial smoothing and Toeplitz reconstruction.

Two fully-correlated (coherent) sources — multipath, or a relay of the same signal —
collapse the signal covariance to rank 1, and MUSIC fails. These helpers restore the
rank MUSIC needs. The trade-off for spatial smoothing: it shrinks the effective
aperture (the subarray is smaller than the array) in exchange for decorrelation.

All three operate on a ULA covariance and return a covariance for a downstream
estimator to consume.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import toeplitz

from rfdf.dsp.doa._common import as_covariance
from rfdf.dsp.errors import InvalidCovarianceError


def forward_spatial_smoothing(covariance: ArrayLike, subarray_size: int) -> NDArray[np.complex128]:
    """Decorrelate coherent sources by forward spatial smoothing.

    Averages the covariances of the ``M - L + 1`` overlapping length-``L`` subarrays of
    a ULA (Shan, Wax & Kailath 1985). Resolves up to ``L - 1`` coherent sources.

    Args:
        covariance: An ``(M, M)`` ULA covariance.
        subarray_size: The subarray length ``L``; must satisfy ``2 <= L <= M``.

    Returns:
        The ``(L, L)`` forward-smoothed covariance.

    Raises:
        InvalidCovarianceError: If the covariance is malformed or ``L`` is out of range.
    """
    cov = as_covariance(covariance)
    num_channels = cov.shape[0]
    if not 2 <= subarray_size <= num_channels:
        raise InvalidCovarianceError(
            f"subarray_size must be in [2, {num_channels}], got {subarray_size}"
        )
    num_subarrays = num_channels - subarray_size + 1
    smoothed = np.zeros((subarray_size, subarray_size), dtype=np.complex128)
    for start in range(num_subarrays):
        smoothed += cov[start : start + subarray_size, start : start + subarray_size]
    result: NDArray[np.complex128] = smoothed / num_subarrays
    return result


def forward_backward_smoothing(covariance: ArrayLike, subarray_size: int) -> NDArray[np.complex128]:
    """Decorrelate coherent sources by forward-backward spatial smoothing.

    Averages the forward-smoothed covariance with its conjugate-reversed (backward)
    counterpart (Pillai & Kwon 1989). FB smoothing roughly doubles the decorrelation
    capacity of forward-only smoothing for a given subarray size.

    Args:
        covariance: An ``(M, M)`` ULA covariance.
        subarray_size: The subarray length ``L``; must satisfy ``2 <= L <= M``.

    Returns:
        The ``(L, L)`` forward-backward-smoothed covariance.

    Raises:
        InvalidCovarianceError: If the covariance is malformed or ``L`` is out of range.
    """
    forward = forward_spatial_smoothing(covariance, subarray_size)
    exchange = np.fliplr(np.eye(subarray_size))
    result: NDArray[np.complex128] = 0.5 * (forward + exchange @ forward.conj() @ exchange)
    return result


def toeplitz_rectify(covariance: ArrayLike) -> NDArray[np.complex128]:
    """Replace a covariance with its nearest Hermitian-Toeplitz matrix.

    A ULA covariance is theoretically Toeplitz; coherent sources break that structure.
    Averaging each diagonal and rebuilding a Hermitian-Toeplitz matrix restores the
    rank a downstream estimator needs (Williams et al. 1988). Unlike spatial smoothing
    this keeps the full aperture, but it assumes an ideal ULA manifold.

    Args:
        covariance: An ``(M, M)`` ULA covariance.

    Returns:
        The ``(M, M)`` nearest Hermitian-Toeplitz covariance.

    Raises:
        InvalidCovarianceError: If the covariance is malformed.
    """
    cov = as_covariance(covariance)
    num_channels = cov.shape[0]
    diagonal_means = np.array(
        [np.mean(np.diagonal(cov, offset=lag)) for lag in range(num_channels)],
        dtype=np.complex128,
    )
    rectified: NDArray[np.complex128] = toeplitz(np.conj(diagonal_means), diagonal_means).astype(
        np.complex128
    )
    return rectified

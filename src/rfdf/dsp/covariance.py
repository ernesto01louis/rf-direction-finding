"""Spatial covariance estimation.

The sample covariance ``R = (1/N) X Xᴴ`` is the input to every subspace and
beamforming DOA estimator. :func:`diagonal_load` regularises a rank-deficient or
snapshot-starved covariance so MVDR and MUSIC stay numerically stable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rfdf.dsp.errors import InvalidCovarianceError


def sample_covariance(iq: ArrayLike) -> NDArray[np.complex128]:
    """Estimate the spatial covariance matrix from an IQ block.

    Computes ``R = (1/N) X Xᴴ`` where ``X`` is the ``(M, N)`` IQ block (M channels,
    N snapshots). The input is promoted to complex128 for downstream eigenanalysis.

    Args:
        iq: IQ samples, shape ``(M, N)``, any complex dtype.

    Returns:
        The ``(M, M)`` complex128 sample covariance.

    Raises:
        InvalidCovarianceError: If ``iq`` is not 2-D, has no channels or snapshots,
            or contains non-finite samples.
    """
    samples = np.asarray(iq)
    if samples.ndim != 2:
        raise InvalidCovarianceError(f"expected an (M, N) IQ array, got ndim {samples.ndim}")
    num_channels, num_snapshots = samples.shape
    if num_channels < 1:
        raise InvalidCovarianceError("IQ must have at least one channel")
    if num_snapshots < 1:
        raise InvalidCovarianceError("IQ must have at least one snapshot")
    data = samples.astype(np.complex128)
    if not bool(np.all(np.isfinite(data))):
        raise InvalidCovarianceError("IQ contains non-finite samples")
    cov: NDArray[np.complex128] = (data @ data.conj().T) / num_snapshots
    return cov


def diagonal_load(covariance: ArrayLike, loading: float = 1e-3) -> NDArray[np.complex128]:
    """Apply diagonal loading to a covariance matrix.

    Returns ``R + loading * (tr R / M) * I``. The load is scaled by the mean
    diagonal power so a single ``loading`` value behaves consistently across
    covariances of different magnitude.

    Args:
        covariance: A square ``(M, M)`` covariance matrix.
        loading: Non-negative loading factor; ``0`` is a no-op.

    Returns:
        The loaded ``(M, M)`` complex128 covariance.

    Raises:
        InvalidCovarianceError: If ``covariance`` is not square.
        ValueError: If ``loading`` is negative.
    """
    cov = np.asarray(covariance, dtype=np.complex128)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise InvalidCovarianceError(f"expected a square (M, M) covariance, got shape {cov.shape}")
    if loading < 0.0:
        raise ValueError(f"loading factor must be non-negative, got {loading}")
    num_channels = cov.shape[0]
    scale = float(np.real(np.trace(cov))) / num_channels
    loaded: NDArray[np.complex128] = cov + loading * scale * np.eye(
        num_channels, dtype=np.complex128
    )
    return loaded

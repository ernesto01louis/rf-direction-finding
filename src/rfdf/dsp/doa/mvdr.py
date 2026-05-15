"""Minimum Variance Distortionless Response (Capon) DOA estimator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.covariance import diagonal_load
from rfdf.dsp.doa._common import as_covariance, peak_pick_1d, require_azimuth_manifold
from rfdf.dsp.doa.result import DoaEstimate
from rfdf.dsp.errors import InvalidCovarianceError
from rfdf.dsp.steering import SteeringManifold

#: Floor applied before reciprocating the Capon response.
_RESPONSE_FLOOR: float = 1e-300


def mvdr(
    covariance: ArrayLike,
    steering: SteeringManifold,
    num_signals: int = 1,
    *,
    loading: float = 1e-3,
) -> DoaEstimate:
    """Estimate direction of arrival by the MVDR / Capon beamformer.

    The pseudospectrum is ``P(theta) = 1 / (a^H R^-1 a)`` (Capon 1969). MVDR resolves
    better than Bartlett and needs no source-count estimate, but is sensitive to
    covariance-estimation error — hence the mandatory diagonal loading.

    Args:
        covariance: The ``(M, M)`` spatial covariance.
        steering: A 1-D azimuth steering manifold.
        num_signals: Number of peaks to report.
        loading: Diagonal-loading factor applied before inversion for robustness.

    Returns:
        A :class:`DoaEstimate` carrying the pseudospectrum and peak bearings.

    Raises:
        InvalidCovarianceError: If the covariance is malformed or its channel count
            does not match the manifold.
        ValueError: If the manifold is not a 1-D azimuth scan.
    """
    cov = as_covariance(covariance)
    elevation = require_azimuth_manifold(steering)
    manifold = steering.matrix
    if manifold.shape[1] != cov.shape[0]:
        raise InvalidCovarianceError(
            f"covariance is {cov.shape[0]}-channel but the manifold is {manifold.shape[1]}-channel"
        )
    inverse = np.linalg.inv(diagonal_load(cov, loading))
    response = np.real(np.einsum("gm,mn,gn->g", manifold.conj(), inverse, manifold))
    spectrum = 1.0 / np.maximum(response, _RESPONSE_FLOOR)
    indices, azimuths, strengths, spectrum_db = peak_pick_1d(
        spectrum, steering.az_grid_deg, num_signals
    )
    return DoaEstimate(
        algorithm="mvdr",
        num_signals=num_signals,
        azimuth_deg=azimuths,
        elevation_deg=[elevation] * len(azimuths),
        pseudospectrum_db=spectrum_db,
        peak_indices=indices,
        peak_strengths_db=strengths,
    )

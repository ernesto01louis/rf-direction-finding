"""Bartlett (conventional) beamforming DOA estimator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.doa._common import as_covariance, peak_pick_1d, require_azimuth_manifold
from rfdf.dsp.doa.result import DoaEstimate
from rfdf.dsp.errors import InvalidCovarianceError
from rfdf.dsp.steering import SteeringManifold


def bartlett(
    covariance: ArrayLike, steering: SteeringManifold, num_signals: int = 1
) -> DoaEstimate:
    """Estimate direction of arrival by Bartlett beamforming.

    The pseudospectrum is ``P(theta) = a^H R a / (a^H a)`` (Bartlett 1948; Van Trees,
    *Optimum Array Processing*, 2002). Bartlett is the robustness floor: unconditionally
    stable, aperture-limited in resolution, and needs no source-count estimate.

    Args:
        covariance: The ``(M, M)`` spatial covariance.
        steering: A 1-D azimuth steering manifold.
        num_signals: Number of peaks to report.

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
    numerator = np.real(np.einsum("gm,mn,gn->g", manifold.conj(), cov, manifold))
    denominator = np.real(np.einsum("gm,gm->g", manifold.conj(), manifold))
    spectrum = numerator / denominator
    indices, azimuths, strengths, spectrum_db = peak_pick_1d(
        spectrum, steering.az_grid_deg, num_signals
    )
    return DoaEstimate(
        algorithm="bartlett",
        num_signals=num_signals,
        azimuth_deg=azimuths,
        elevation_deg=[elevation] * len(azimuths),
        pseudospectrum_db=spectrum_db,
        peak_indices=indices,
        peak_strengths_db=strengths,
    )

"""MUSIC (MUltiple SIgnal Classification) DOA estimator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.doa._common import (
    as_covariance,
    peak_pick_1d,
    require_azimuth_manifold,
    signal_noise_subspaces,
)
from rfdf.dsp.doa.result import DoaEstimate
from rfdf.dsp.errors import InvalidCovarianceError
from rfdf.dsp.steering import SteeringManifold

#: Floor applied before reciprocating the noise-subspace projection.
_NULL_FLOOR: float = 1e-300


def music(covariance: ArrayLike, steering: SteeringManifold, num_signals: int) -> DoaEstimate:
    """Estimate direction of arrival by MUSIC.

    Eigendecomposes the covariance, takes the ``M - K`` smallest-eigenvalue eigenvectors
    as the noise subspace ``U_n``, and forms ``P(theta) = 1 / ||U_n^H a(theta)||^2``
    (Schmidt 1986). MUSIC is the asymptotically efficient workhorse for full-rank
    (non-coherent) sources.

    Args:
        covariance: The ``(M, M)`` spatial covariance.
        steering: A 1-D azimuth steering manifold.
        num_signals: Number of sources ``K``; must satisfy ``1 <= K < M``.

    Returns:
        A :class:`DoaEstimate` carrying the pseudospectrum and peak bearings.

    Raises:
        InvalidCovarianceError: If the covariance is malformed or its channel count
            does not match the manifold.
        SourceCountError: If ``num_signals`` is outside ``[1, M - 1]``.
        ValueError: If the manifold is not a 1-D azimuth scan.
    """
    cov = as_covariance(covariance)
    elevation = require_azimuth_manifold(steering)
    manifold = steering.matrix
    if manifold.shape[1] != cov.shape[0]:
        raise InvalidCovarianceError(
            f"covariance is {cov.shape[0]}-channel but the manifold is {manifold.shape[1]}-channel"
        )
    _, noise = signal_noise_subspaces(cov, num_signals)
    projection = manifold @ noise.conj()
    null = np.real(np.sum(np.abs(projection) ** 2, axis=1))
    spectrum = 1.0 / np.maximum(null, _NULL_FLOOR)
    indices, azimuths, strengths, spectrum_db = peak_pick_1d(
        spectrum, steering.az_grid_deg, num_signals
    )
    return DoaEstimate(
        algorithm="music",
        num_signals=num_signals,
        azimuth_deg=azimuths,
        elevation_deg=[elevation] * len(azimuths),
        pseudospectrum_db=spectrum_db,
        peak_indices=indices,
        peak_strengths_db=strengths,
    )

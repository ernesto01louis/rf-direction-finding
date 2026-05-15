"""Root-MUSIC DOA estimator (uniform linear array only)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.doa._common import (
    as_covariance,
    azimuth_from_phase,
    signal_noise_subspaces,
    ula_geometry,
)
from rfdf.dsp.doa.result import DoaEstimate


def root_music(
    covariance: ArrayLike, *, positions: ArrayLike, freq_hz: float, num_signals: int
) -> DoaEstimate:
    """Estimate direction of arrival by Root-MUSIC.

    Builds the polynomial whose coefficients are the diagonal sums of the noise-subspace
    projector and roots it (Barabell 1983). Root-MUSIC has higher precision than
    grid-based MUSIC at low SNR, but exists only for a ULA's Vandermonde structure.

    Args:
        covariance: The ``(M, M)`` spatial covariance.
        positions: Antenna positions, shape ``(M, 3)`` in metres — must be a ULA.
        freq_hz: RF frequency in hertz.
        num_signals: Number of sources ``K``; must satisfy ``1 <= K < M``.

    Returns:
        A :class:`DoaEstimate` with the recovered azimuths (``pseudospectrum_db`` is
        ``None`` — Root-MUSIC is parametric, not grid-based).

    Raises:
        InvalidCovarianceError: If the covariance is malformed.
        NotULAError: If ``positions`` is not a uniform linear array.
        SourceCountError: If ``num_signals`` is outside ``[1, M - 1]``.
    """
    cov = as_covariance(covariance)
    spacing, order = ula_geometry(positions)
    ordered = cov[np.ix_(order, order)]
    num_channels = ordered.shape[0]

    _, noise = signal_noise_subspaces(ordered, num_signals)
    projector = noise @ noise.conj().T
    coefficients = np.array(
        [
            np.sum(np.diagonal(projector, offset=lag))
            for lag in range(num_channels - 1, -num_channels, -1)
        ]
    )
    roots = np.roots(coefficients)
    inside = roots[np.abs(roots) < 1.0]
    if inside.size < num_signals:
        inside = roots
    closeness = np.abs(1.0 - np.abs(inside))
    signal_roots = inside[np.argsort(closeness)[:num_signals]]
    azimuths = azimuth_from_phase(signal_roots, spacing, freq_hz)
    return DoaEstimate(
        algorithm="root_music",
        num_signals=num_signals,
        azimuth_deg=sorted(float(angle) for angle in azimuths),
        elevation_deg=[0.0] * num_signals,
        pseudospectrum_db=None,
    )

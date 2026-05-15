"""ESPRIT and Unitary ESPRIT DOA estimators (uniform linear array only)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.doa._common import (
    as_covariance,
    azimuth_from_phase,
    signal_noise_subspaces,
    tls_shift_solve,
    ula_geometry,
)
from rfdf.dsp.doa.result import DoaEstimate


def esprit(
    covariance: ArrayLike, *, positions: ArrayLike, freq_hz: float, num_signals: int
) -> DoaEstimate:
    """Estimate direction of arrival by ESPRIT.

    Exploits the rotational invariance of a ULA's two maximally-overlapping subarrays:
    the eigenvalues of the total-least-squares shift operator are the per-element
    phases (Roy & Kailath 1989). Closed-form — no grid search.

    Args:
        covariance: The ``(M, M)`` spatial covariance.
        positions: Antenna positions, shape ``(M, 3)`` in metres — must be a ULA.
        freq_hz: RF frequency in hertz.
        num_signals: Number of sources ``K``; must satisfy ``1 <= K < M``.

    Returns:
        A :class:`DoaEstimate` with the recovered azimuths.

    Raises:
        InvalidCovarianceError: If the covariance is malformed.
        NotULAError: If ``positions`` is not a uniform linear array.
        SourceCountError: If ``num_signals`` is outside ``[1, M - 1]``.
    """
    cov = as_covariance(covariance)
    spacing, order = ula_geometry(positions)
    ordered = cov[np.ix_(order, order)]
    signal, _ = signal_noise_subspaces(ordered, num_signals)
    psi = tls_shift_solve(signal, num_signals)
    azimuths = azimuth_from_phase(np.linalg.eigvals(psi), spacing, freq_hz)
    return DoaEstimate(
        algorithm="esprit",
        num_signals=num_signals,
        azimuth_deg=sorted(float(angle) for angle in azimuths),
        elevation_deg=[0.0] * num_signals,
        pseudospectrum_db=None,
    )


def unitary_esprit(
    covariance: ArrayLike, *, positions: ArrayLike, freq_hz: float, num_signals: int
) -> DoaEstimate:
    """Estimate direction of arrival by Unitary ESPRIT.

    Haardt & Nossek 1995's Unitary ESPRIT obtains its accuracy gain from forward-
    backward averaging — it doubles the effective snapshot count and partially
    decorrelates coherent sources. This implementation applies FB averaging to the
    covariance and then the ESPRIT solve; it produces the same estimates as the
    real-valued Haardt-Nossek formulation, whose unitary transform is purely an
    arithmetic optimisation. FB averaging requires the array be centro-symmetric,
    which every ULA is.

    Args:
        covariance: The ``(M, M)`` spatial covariance.
        positions: Antenna positions, shape ``(M, 3)`` in metres — must be a ULA.
        freq_hz: RF frequency in hertz.
        num_signals: Number of sources ``K``; must satisfy ``1 <= K < M``.

    Returns:
        A :class:`DoaEstimate` with the recovered azimuths.

    Raises:
        InvalidCovarianceError: If the covariance is malformed.
        NotULAError: If ``positions`` is not a uniform linear array.
        SourceCountError: If ``num_signals`` is outside ``[1, M - 1]``.
    """
    cov = as_covariance(covariance)
    spacing, order = ula_geometry(positions)
    ordered = cov[np.ix_(order, order)]
    num_channels = ordered.shape[0]
    exchange = np.fliplr(np.eye(num_channels))
    fb_averaged = 0.5 * (ordered + exchange @ ordered.conj() @ exchange)
    signal, _ = signal_noise_subspaces(fb_averaged, num_signals)
    psi = tls_shift_solve(signal, num_signals)
    azimuths = azimuth_from_phase(np.linalg.eigvals(psi), spacing, freq_hz)
    return DoaEstimate(
        algorithm="unitary_esprit",
        num_signals=num_signals,
        azimuth_deg=sorted(float(angle) for angle in azimuths),
        elevation_deg=[0.0] * num_signals,
        pseudospectrum_db=None,
    )

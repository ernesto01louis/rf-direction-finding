"""Number-of-sources estimation: AIC, MDL, and SORTE.

When the operator does not know how many emitters are present, these criteria estimate
it from the sample-covariance eigenvalues. AIC and MDL (Wax & Kailath 1985) weigh the
geometric-vs-arithmetic mean of the candidate noise eigenvalues against a model-
complexity penalty; SORTE (He, Wang & Kong 2010) inspects the second-order statistics of
the eigenvalue gaps and needs no snapshot count.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.doa._common import as_covariance
from rfdf.dsp.errors import SourceCountError

_EIGENVALUE_FLOOR: float = 1e-300


def _descending_eigenvalues(covariance: ArrayLike) -> np.ndarray:
    """Return the real covariance eigenvalues sorted largest-first."""
    matrix = as_covariance(covariance)
    eigenvalues: np.ndarray = np.sort(np.linalg.eigvalsh(matrix).real)[::-1]
    return eigenvalues


def _log_likelihood_ratio(eigenvalues: np.ndarray, candidate: int) -> float:
    """Return ``(M - k) * log(arithmetic / geometric mean)`` of the noise eigenvalues."""
    noise = np.maximum(eigenvalues[candidate:], _EIGENVALUE_FLOOR)
    arithmetic = float(np.mean(noise))
    geometric = float(np.exp(np.mean(np.log(noise))))
    return float(eigenvalues.size - candidate) * float(np.log(arithmetic / geometric))


def aic(covariance: ArrayLike, snapshots: int) -> int:
    """Estimate the source count by the Akaike Information Criterion.

    Args:
        covariance: A square spatial covariance matrix.
        snapshots: Number of snapshots used to estimate the covariance.

    Returns:
        The estimated number of sources, in ``[0, M - 1]``.

    Raises:
        ValueError: If ``snapshots`` is not positive.
    """
    if snapshots < 1:
        raise ValueError(f"snapshots must be positive, got {snapshots}")
    eigenvalues = _descending_eigenvalues(covariance)
    num_channels = eigenvalues.size
    scores = [
        2.0 * snapshots * _log_likelihood_ratio(eigenvalues, k) + 2.0 * k * (2 * num_channels - k)
        for k in range(num_channels)
    ]
    return int(np.argmin(scores))


def mdl(covariance: ArrayLike, snapshots: int) -> int:
    """Estimate the source count by the Minimum Description Length criterion.

    MDL is statistically consistent — unlike AIC it does not asymptotically
    over-estimate.

    Args:
        covariance: A square spatial covariance matrix.
        snapshots: Number of snapshots used to estimate the covariance.

    Returns:
        The estimated number of sources, in ``[0, M - 1]``.

    Raises:
        ValueError: If ``snapshots`` is not positive.
    """
    if snapshots < 1:
        raise ValueError(f"snapshots must be positive, got {snapshots}")
    eigenvalues = _descending_eigenvalues(covariance)
    num_channels = eigenvalues.size
    scores = [
        snapshots * _log_likelihood_ratio(eigenvalues, k)
        + 0.5 * k * (2 * num_channels - k) * float(np.log(snapshots))
        for k in range(num_channels)
    ]
    return int(np.argmin(scores))


def sorte(covariance: ArrayLike) -> int:
    """Estimate the source count by the SORTE eigenvalue-gap criterion.

    Args:
        covariance: A square spatial covariance matrix with at least 4 channels.

    Returns:
        The estimated number of sources, in ``[1, M - 2]``.

    Raises:
        SourceCountError: If the covariance has fewer than 4 channels.
    """
    eigenvalues = _descending_eigenvalues(covariance)
    num_channels = eigenvalues.size
    if num_channels < 4:
        raise SourceCountError(f"SORTE needs at least 4 channels, got {num_channels}")
    gaps = -np.diff(eigenvalues)
    scores: list[float] = []
    for candidate in range(1, num_channels - 2):
        variance = float(np.var(gaps[candidate - 1 :]))
        next_variance = float(np.var(gaps[candidate:]))
        scores.append(next_variance / variance if variance > _EIGENVALUE_FLOOR else np.inf)
    return int(np.argmin(scores)) + 1


def estimate_num_signals(covariance: ArrayLike, snapshots: int, *, method: str = "mdl") -> int:
    """Estimate the source count by the named criterion.

    Args:
        covariance: A square spatial covariance matrix.
        snapshots: Number of snapshots (used by ``aic`` and ``mdl``; ignored by ``sorte``).
        method: One of ``"aic"``, ``"mdl"``, or ``"sorte"``.

    Returns:
        The estimated number of sources.

    Raises:
        ValueError: If ``method`` is not a recognised criterion.
    """
    if method == "mdl":
        return mdl(covariance, snapshots)
    if method == "aic":
        return aic(covariance, snapshots)
    if method == "sorte":
        return sorte(covariance)
    raise ValueError(f"unknown method {method!r}; expected 'aic', 'mdl', or 'sorte'")

"""2-D (azimuth + elevation) MUSIC DOA estimator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import maximum_filter

from rfdf.dsp.doa._common import as_covariance, signal_noise_subspaces
from rfdf.dsp.doa.result import Doa2DResult
from rfdf.dsp.errors import InvalidCovarianceError
from rfdf.dsp.steering import SteeringManifold

#: Floor applied before reciprocating the noise-subspace projection.
_NULL_FLOOR: float = 1e-300


def _peak_pick_2d(surface: np.ndarray, num_signals: int) -> list[tuple[int, int]]:
    """Return the ``(az_index, el_index)`` of the strongest local maxima."""
    is_local_max = surface == maximum_filter(surface, size=3, mode="nearest")
    coords = np.argwhere(is_local_max)
    if coords.shape[0] < num_signals:
        flat = np.argsort(surface, axis=None)[::-1][:num_signals]
        coords = np.column_stack(np.unravel_index(flat, surface.shape))
    strengths = surface[coords[:, 0], coords[:, 1]]
    chosen = coords[np.argsort(strengths)[::-1][:num_signals]]
    return [(int(az_idx), int(el_idx)) for az_idx, el_idx in chosen]


def music_2d(covariance: ArrayLike, steering: SteeringManifold, num_signals: int) -> Doa2DResult:
    """Estimate azimuth and elevation jointly by 2-D MUSIC.

    Runs the MUSIC null-spectrum over a full ``(azimuth, elevation)`` grid and peak-picks
    the surface (Schmidt 1986, generalised to a planar/3-D array). 2-D MUSIC is the
    workhorse for arrays with extent in more than one axis.

    Args:
        covariance: The ``(M, M)`` spatial covariance.
        steering: A 2-D steering manifold spanning azimuth and elevation.
        num_signals: Number of sources ``K``; must satisfy ``1 <= K < M``.

    Returns:
        A :class:`Doa2DResult` carrying the pseudospectrum surface and peak bearings.

    Raises:
        InvalidCovarianceError: If the covariance is malformed or its channel count
            does not match the manifold.
        SourceCountError: If ``num_signals`` is outside ``[1, M - 1]``.
    """
    cov = as_covariance(covariance)
    manifold = steering.matrix
    if manifold.shape[1] != cov.shape[0]:
        raise InvalidCovarianceError(
            f"covariance is {cov.shape[0]}-channel but the manifold is {manifold.shape[1]}-channel"
        )
    _, noise = signal_noise_subspaces(cov, num_signals)
    projection = manifold @ noise.conj()
    null = np.real(np.sum(np.abs(projection) ** 2, axis=1))
    spectrum = 1.0 / np.maximum(null, _NULL_FLOOR)
    num_az, num_el = steering.grid_shape
    surface = spectrum.reshape(num_az, num_el)
    surface_db = 10.0 * np.log10(np.maximum(surface, _NULL_FLOOR))

    peaks = _peak_pick_2d(surface, num_signals)
    azimuths = [float(steering.az_grid_deg[az_idx]) for az_idx, _ in peaks]
    elevations = [float(steering.el_grid_deg[el_idx]) for _, el_idx in peaks]
    return Doa2DResult(
        algorithm="music_2d",
        num_signals=num_signals,
        azimuth_deg=azimuths,
        elevation_deg=elevations,
        pseudospectrum_db=surface_db,
        az_grid_deg=steering.az_grid_deg,
        el_grid_deg=steering.el_grid_deg,
    )

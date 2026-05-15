"""Shared helpers for the narrowband DOA estimators.

Covariance validation, the signal/noise eigen-split, 1-D peak picking with parabolic
refinement, ULA geometry analysis, the rotational-phase-to-azimuth map, and the TLS
shift solve used by ESPRIT all live here so each estimator module stays small.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from rfdf.dsp.errors import InvalidCovarianceError, NotULAError, SourceCountError
from rfdf.dsp.geometry_presets import is_ula, validate_positions
from rfdf.dsp.steering import SteeringManifold, wavelength

#: Floor applied before a log so an empty bin never produces ``-inf`` dB.
_SPECTRUM_FLOOR: float = 1e-300


def as_covariance(covariance: ArrayLike) -> NDArray[np.complex128]:
    """Validate and coerce a covariance matrix to ``(M, M)`` complex128.

    Args:
        covariance: A square spatial covariance matrix.

    Returns:
        The covariance as an ``(M, M)`` complex128 array.

    Raises:
        InvalidCovarianceError: If the matrix is not square, too small, or non-finite.
    """
    matrix = np.asarray(covariance, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InvalidCovarianceError(
            f"expected a square (M, M) covariance, got shape {matrix.shape}"
        )
    if matrix.shape[0] < 2:
        raise InvalidCovarianceError("covariance must have at least 2 channels")
    if not bool(np.all(np.isfinite(matrix))):
        raise InvalidCovarianceError("covariance contains non-finite entries")
    return matrix


def signal_noise_subspaces(
    covariance: NDArray[np.complex128], num_signals: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a covariance into its signal and noise eigen-subspaces.

    Args:
        covariance: A validated ``(M, M)`` covariance matrix.
        num_signals: Number of sources ``K``; must satisfy ``1 <= K < M``.

    Returns:
        ``(signal_vectors, noise_vectors)`` — the ``K`` largest-eigenvalue eigenvectors
        and the ``M - K`` smallest, each a column-major ``(M, .)`` array.

    Raises:
        SourceCountError: If ``num_signals`` is outside ``[1, M - 1]``.
    """
    num_channels = covariance.shape[0]
    if not 1 <= num_signals < num_channels:
        raise SourceCountError(f"num_signals must be in [1, {num_channels - 1}], got {num_signals}")
    _, eigenvectors = np.linalg.eigh(covariance)  # ascending eigenvalues
    noise = eigenvectors[:, : num_channels - num_signals]
    signal = eigenvectors[:, num_channels - num_signals :]
    return signal, noise


def require_azimuth_manifold(steering: SteeringManifold) -> float:
    """Confirm a manifold is a 1-D azimuth scan and return its single elevation.

    Args:
        steering: The steering manifold a grid estimator was handed.

    Returns:
        The manifold's single elevation in degrees.

    Raises:
        ValueError: If the manifold spans more than one elevation.
    """
    if steering.el_grid_deg.size != 1:
        raise ValueError(
            "narrowband estimators require a 1-D azimuth manifold (a single elevation); "
            "use the 2D estimators for an elevation grid"
        )
    return float(steering.el_grid_deg[0])


def peak_pick_1d(
    spectrum: ArrayLike, az_grid_deg: ArrayLike, num_signals: int
) -> tuple[list[int], list[float], list[float], np.ndarray]:
    """Find ``num_signals`` peaks in a 1-D pseudospectrum with parabolic refinement.

    Args:
        spectrum: The linear-scale pseudospectrum, one value per grid point.
        az_grid_deg: The matching uniform azimuth grid in degrees.
        num_signals: Number of peaks to return.

    Returns:
        ``(peak_indices, azimuths_deg, strengths_db, spectrum_db)`` — the grid indices
        of the chosen peaks, their parabola-refined azimuths, their heights in dB, and
        the full dB spectrum.
    """
    linear = np.asarray(spectrum, dtype=np.float64)
    grid = np.asarray(az_grid_deg, dtype=np.float64)
    spectrum_db = 10.0 * np.log10(np.maximum(linear, _SPECTRUM_FLOOR))

    candidates, _ = find_peaks(spectrum_db)
    if candidates.size < num_signals:
        ranked = np.argsort(spectrum_db)[::-1]
        candidates = np.unique(np.concatenate([candidates, ranked[:num_signals]]))
    strongest = candidates[np.argsort(spectrum_db[candidates])[::-1][:num_signals]]
    chosen = np.sort(strongest)

    step = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    azimuths: list[float] = []
    strengths: list[float] = []
    for index in chosen:
        offset = 0.0
        if 0 < index < spectrum_db.size - 1:
            left, peak, right = spectrum_db[index - 1 : index + 2]
            curvature = left - 2.0 * peak + right
            if abs(curvature) > 1e-12:
                offset = float(np.clip(0.5 * (left - right) / curvature, -1.0, 1.0))
        azimuths.append(float(grid[index]) + offset * step)
        strengths.append(float(spectrum_db[index]))
    return [int(i) for i in chosen], azimuths, strengths, spectrum_db


def ula_geometry(positions: ArrayLike) -> tuple[float, np.ndarray]:
    """Validate a ULA and return its spacing and element ordering.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.

    Returns:
        ``(spacing_m, order)`` — the inter-element spacing and the index order that
        sorts the elements along the array axis.

    Raises:
        NotULAError: If the geometry is not a uniform linear array.
    """
    pos = validate_positions(positions)
    if not is_ula(pos):
        raise NotULAError(
            "this estimator requires a uniform linear array; use MUSIC for arbitrary geometries"
        )
    axis = pos[-1] - pos[0]
    axis = axis / np.linalg.norm(axis)
    coordinates = (pos - pos[0]) @ axis
    order = np.argsort(coordinates)
    spacing = float(np.mean(np.diff(coordinates[order])))
    return spacing, order


def azimuth_from_phase(phases: np.ndarray, spacing_m: float, freq_hz: float) -> np.ndarray:
    """Recover azimuths from ULA rotational phases ``exp(-j k d cos az)``.

    Args:
        phases: Complex rotational phases (Root-MUSIC roots or ESPRIT eigenvalues).
        spacing_m: ULA inter-element spacing in metres.
        freq_hz: RF frequency in hertz.

    Returns:
        Azimuths in degrees, each in ``[0, 180]`` (the ULA's unambiguous range).
    """
    wavenumber = 2.0 * np.pi / wavelength(freq_hz)
    cos_az = -np.angle(phases) / (wavenumber * spacing_m)
    azimuths: np.ndarray = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    return azimuths


def tls_shift_solve(signal_subspace: np.ndarray, num_signals: int) -> np.ndarray:
    """Solve the ESPRIT shift equation by total least squares.

    Solves ``Us1 @ Psi = Us2`` for ``Psi`` where ``Us1`` / ``Us2`` are the first and
    last ``M - 1`` rows of the signal subspace, using the SVD-based TLS solution so
    noise in both subarrays is treated symmetrically.

    Args:
        signal_subspace: The ``(M, K)`` signal-subspace eigenvectors.
        num_signals: Number of sources ``K``.

    Returns:
        The ``(K, K)`` rotation operator ``Psi``; its eigenvalues are the phases.
    """
    upper = signal_subspace[:-1, :]
    lower = signal_subspace[1:, :]
    stacked = np.hstack([upper, lower])
    _, _, vh = np.linalg.svd(stacked)
    right = vh.conj().T
    v12 = right[:num_signals, num_signals:]
    v22 = right[num_signals:, num_signals:]
    psi: np.ndarray = -v12 @ np.linalg.inv(v22)
    return psi

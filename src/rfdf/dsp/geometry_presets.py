"""Antenna geometry presets and geometry-analysis helpers.

Stage 3 algorithms operate on plain ``(M, 3)`` position arrays in metres. This module
builds the two reference geometries — a uniform linear array (ULA) and the 5-element
planar cross — and provides :func:`is_ula`, the structural test that ULA-only
algorithms (Root-MUSIC, ESPRIT, spatial smoothing) use to reject arbitrary arrays.

Presets return raw ``numpy`` arrays, never backend objects, so ``rfdf.dsp`` stays
decoupled from ``rfdf.backends``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rfdf.dsp.errors import InvalidGeometryError

#: Speed of light in vacuum (m/s).
SPEED_OF_LIGHT: float = 299_792_458.0


def validate_positions(positions: ArrayLike) -> NDArray[np.float64]:
    """Coerce and validate an antenna-position array.

    Args:
        positions: Antenna coordinates, coercible to shape ``(M, 3)`` in metres.

    Returns:
        The positions as an ``(M, 3)`` float64 array.

    Raises:
        InvalidGeometryError: If the array is not 2-D with three columns, or empty.
    """
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise InvalidGeometryError(
            f"expected (M, 3) antenna positions in metres, got shape {arr.shape}"
        )
    if arr.shape[0] < 1:
        raise InvalidGeometryError("at least one antenna position is required")
    return arr


def ula(num_elements: int, spacing_m: float) -> NDArray[np.float64]:
    """Build a uniform linear array along the +x axis.

    Args:
        num_elements: Number of antennas; must be at least 2.
        spacing_m: Inter-element spacing in metres; must be positive.

    Returns:
        A ``(num_elements, 3)`` array of element positions in metres.

    Raises:
        InvalidGeometryError: If ``num_elements < 2`` or ``spacing_m <= 0``.
    """
    if num_elements < 2:
        raise InvalidGeometryError(f"a ULA needs at least 2 elements, got {num_elements}")
    if spacing_m <= 0.0:
        raise InvalidGeometryError(f"element spacing must be positive, got {spacing_m}")
    positions = np.zeros((num_elements, 3), dtype=np.float64)
    positions[:, 0] = np.arange(num_elements, dtype=np.float64) * spacing_m
    return positions


def planar_cross(spacing_m: float) -> NDArray[np.float64]:
    """Build the reference 5-element planar cross array.

    The cross is the project's reference geometry: an antenna at the origin plus four
    at +/-x and +/-y, all in the z = 0 plane.

    Args:
        spacing_m: Distance from the origin to each outer element, in metres.

    Returns:
        A ``(5, 3)`` array of element positions in metres.

    Raises:
        InvalidGeometryError: If ``spacing_m <= 0``.
    """
    if spacing_m <= 0.0:
        raise InvalidGeometryError(f"element spacing must be positive, got {spacing_m}")
    s = spacing_m
    return np.array(
        [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0], [-s, 0.0, 0.0], [0.0, -s, 0.0]],
        dtype=np.float64,
    )


def half_wavelength_spacing(freq_hz: float) -> float:
    """Return the lambda/2 element spacing (m) for an RF frequency.

    Args:
        freq_hz: RF frequency in hertz; must be positive.

    Returns:
        Half the free-space wavelength, in metres.

    Raises:
        ValueError: If ``freq_hz <= 0``.
    """
    if freq_hz <= 0.0:
        raise ValueError(f"frequency must be positive, got {freq_hz}")
    return SPEED_OF_LIGHT / freq_hz / 2.0


def is_ula(positions: ArrayLike, *, atol_m: float = 1e-6) -> bool:
    """Test whether antenna positions form a uniform linear array.

    A ULA is collinear with equal inter-element spacing, along any axis. Root-MUSIC,
    ESPRIT, and forward spatial smoothing require this shift-invariant structure.

    Args:
        positions: Antenna coordinates, shape ``(M, 3)`` in metres.
        atol_m: Absolute tolerance, in metres, for the collinearity and
            equal-spacing checks.

    Returns:
        ``True`` if the array is a ULA, ``False`` otherwise.
    """
    pos = validate_positions(positions)
    num = pos.shape[0]
    if num < 2:
        return False
    axis = pos[-1] - pos[0]
    span = float(np.linalg.norm(axis))
    if span <= atol_m:
        return False
    axis = axis / span
    offsets = pos - pos[0]
    coords = offsets @ axis
    residual = offsets - np.outer(coords, axis)
    if bool(np.any(np.linalg.norm(residual, axis=1) > atol_m)):
        return False
    steps = np.diff(np.sort(coords))
    return bool(np.all(np.abs(steps - steps[0]) <= atol_m))

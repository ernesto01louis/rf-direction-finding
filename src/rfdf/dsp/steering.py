r"""Array steering manifold — the single definition of the phase convention.

Every DOA algorithm in :mod:`rfdf.dsp` builds its steering vectors here, so the
``exp(-j)`` far-field plane-wave convention is defined exactly once. It matches the
mock SDR's signal model (:mod:`rfdf.backends.sdr.mock`) verbatim:

.. math::

    a(\theta) = \exp\!\left(-j \frac{2\pi}{\lambda}\, P \cdot \hat{\theta}\right),
    \qquad
    \hat{\theta} = (\cos\phi_{el}\cos\phi_{az},\;
                    \cos\phi_{el}\sin\phi_{az},\;
                    \sin\phi_{el})

``P`` is the ``(M, 3)`` antenna-position matrix in metres. An estimator that builds
its manifold with the opposite sign silently mirrors every bearing — so it must not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rfdf.dsp.geometry_presets import SPEED_OF_LIGHT, validate_positions


def wavelength(freq_hz: float) -> float:
    """Return the free-space wavelength (m) for an RF frequency.

    Args:
        freq_hz: RF frequency in hertz; must be positive.

    Returns:
        The wavelength ``c / f`` in metres.

    Raises:
        ValueError: If ``freq_hz <= 0``.
    """
    if freq_hz <= 0.0:
        raise ValueError(f"frequency must be positive, got {freq_hz}")
    return SPEED_OF_LIGHT / freq_hz


def direction_unit_vector(az_deg: ArrayLike, el_deg: ArrayLike) -> NDArray[np.float64]:
    """Convert azimuth/elevation in degrees to a unit propagation vector.

    Azimuth is measured from +x toward +y; elevation from the xy-plane toward +z.
    Inputs broadcast against each other, so passing 1-D grids returns a stack.

    Args:
        az_deg: Azimuth angle(s) in degrees.
        el_deg: Elevation angle(s) in degrees.

    Returns:
        Unit vector(s) of shape ``broadcast(az, el) + (3,)``.
    """
    az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(el_deg, dtype=np.float64))
    az, el = np.broadcast_arrays(az, el)
    return np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        axis=-1,
    )


def steering_vector(
    positions: ArrayLike, *, az_deg: float, el_deg: float, freq_hz: float
) -> NDArray[np.complex128]:
    """Compute the array steering vector for a single direction.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        az_deg: Azimuth of the incoming wavefront, in degrees.
        el_deg: Elevation of the incoming wavefront, in degrees.
        freq_hz: RF frequency in hertz.

    Returns:
        The ``(M,)`` complex128 steering vector ``exp(-j (2pi/lambda) P . theta_hat)``.

    Raises:
        InvalidGeometryError: If ``positions`` is not a valid ``(M, 3)`` array.
        ValueError: If ``freq_hz <= 0``.
    """
    pos = validate_positions(positions)
    lam = wavelength(freq_hz)
    theta_hat = direction_unit_vector(az_deg, el_deg)
    phase = (2.0 * np.pi / lam) * (pos @ theta_hat)
    vector: NDArray[np.complex128] = np.exp(-1j * phase).astype(np.complex128)
    return vector


@dataclass(frozen=True)
class SteeringManifold:
    """A precomputed bank of steering vectors over an (azimuth, elevation) grid.

    The grid is flattened azimuth-major: flat index ``i`` corresponds to
    ``az_grid_deg[i // E]`` and ``el_grid_deg[i % E]`` where ``E`` is the elevation
    grid length.

    Attributes:
        matrix: Steering vectors, shape ``(G, M)`` complex128, ``G = A * E``.
        az_grid_deg: The azimuth grid, shape ``(A,)`` in degrees.
        el_grid_deg: The elevation grid, shape ``(E,)`` in degrees.
    """

    matrix: NDArray[np.complex128]
    az_grid_deg: NDArray[np.float64]
    el_grid_deg: NDArray[np.float64]

    @property
    def num_points(self) -> int:
        """Number of grid points ``G``."""
        return int(self.matrix.shape[0])

    @property
    def num_elements(self) -> int:
        """Number of antennas ``M``."""
        return int(self.matrix.shape[1])

    @property
    def grid_shape(self) -> tuple[int, int]:
        """The ``(num_azimuth, num_elevation)`` shape of the underlying grid."""
        return (int(self.az_grid_deg.size), int(self.el_grid_deg.size))

    def direction(self, index: int) -> tuple[float, float]:
        """Return the ``(azimuth_deg, elevation_deg)`` for a flat grid index.

        Args:
            index: Flat row index into :attr:`matrix`.

        Returns:
            The azimuth and elevation, in degrees, of that grid point.
        """
        az_idx, el_idx = divmod(index, int(self.el_grid_deg.size))
        return (float(self.az_grid_deg[az_idx]), float(self.el_grid_deg[el_idx]))


def build_manifold(
    positions: ArrayLike,
    az_grid_deg: ArrayLike,
    el_grid_deg: ArrayLike,
    freq_hz: float,
) -> SteeringManifold:
    """Precompute the steering manifold over an azimuth/elevation grid.

    Evaluation is fully vectorized — the ``(G, M)`` manifold for a 360x90 grid and
    five elements is only a few megabytes, so broadcasting beats a Python loop.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        az_grid_deg: 1-D azimuth grid in degrees.
        el_grid_deg: 1-D elevation grid in degrees (use ``[0.0]`` for azimuth-only).
        freq_hz: RF frequency in hertz.

    Returns:
        A :class:`SteeringManifold` whose ``matrix`` is ``(A * E, M)`` complex128.

    Raises:
        InvalidGeometryError: If ``positions`` is not a valid ``(M, 3)`` array.
        ValueError: If a grid is not 1-D, or ``freq_hz <= 0``.
    """
    pos = validate_positions(positions)
    az = np.atleast_1d(np.asarray(az_grid_deg, dtype=np.float64))
    el = np.atleast_1d(np.asarray(el_grid_deg, dtype=np.float64))
    if az.ndim != 1 or el.ndim != 1:
        raise ValueError("az_grid_deg and el_grid_deg must be 1-D")
    lam = wavelength(freq_hz)
    az_mesh, el_mesh = np.meshgrid(az, el, indexing="ij")
    dirs = direction_unit_vector(az_mesh.ravel(), el_mesh.ravel())
    phase = (2.0 * np.pi / lam) * (dirs @ pos.T)
    matrix = np.exp(-1j * phase).astype(np.complex128)
    return SteeringManifold(matrix=matrix, az_grid_deg=az, el_grid_deg=el)

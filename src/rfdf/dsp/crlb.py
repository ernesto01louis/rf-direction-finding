r"""Cramer-Rao lower bound for direction-of-arrival estimation.

Implements the deterministic (conditional) CRB of Stoica & Nehorai 1989 — the
variance floor any unbiased DOA estimator obeys, and the bar the CRLB-bounded
tests hold every algorithm to:

.. math::

    \mathrm{CRB} = \frac{\sigma^2}{2N}
        \left\{ \mathrm{Re}\!\left[ (D^H \Pi_A^\perp D) \odot R_s^T \right] \right\}^{-1}

with ``D`` the steering-vector derivatives, ``Pi_A^perp`` the projector onto the
noise subspace, ``R_s`` the source covariance, ``N`` the snapshot count and
``sigma^2`` the noise power. The formula is geometry-agnostic — it uses analytic
steering derivatives, so it covers the ULA and the planar cross alike.
:func:`crlb_ula_closed_form` gives the textbook closed form as an independent check.

For a planar array an in-plane source (``el = 0``) has zero elevation Fisher
information: its elevation is un-estimable and the bound is ``+inf``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rfdf.dsp.errors import SourceCountError
from rfdf.dsp.geometry_presets import validate_positions
from rfdf.dsp.steering import steering_vector, wavelength

#: Conversion factor from a variance in radians^2 to one in degrees^2.
_RAD2_TO_DEG2: float = (180.0 / np.pi) ** 2

#: Relative tolerance below which Fisher information counts as zero.
_RANK_TOL: float = 1e-9


def _angle_derivative_unit_vector(
    az_deg: float, el_deg: float, parameter: str
) -> NDArray[np.float64]:
    """Return d(theta_hat)/d(parameter), the derivative taken with respect to radians."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    if parameter == "azimuth":
        return np.array([-np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), 0.0], dtype=np.float64)
    if parameter == "elevation":
        return np.array(
            [-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)], dtype=np.float64
        )
    raise ValueError(f"parameter must be 'azimuth' or 'elevation', got {parameter!r}")


def steering_derivative(
    positions: ArrayLike,
    *,
    az_deg: float,
    el_deg: float,
    freq_hz: float,
    parameter: str,
) -> NDArray[np.complex128]:
    """Analytic derivative of the steering vector with respect to azimuth or elevation.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        az_deg: Source azimuth in degrees.
        el_deg: Source elevation in degrees.
        freq_hz: RF frequency in hertz.
        parameter: ``"azimuth"`` or ``"elevation"``.

    Returns:
        The ``(M,)`` complex128 derivative ``da/dparameter`` (per radian).

    Raises:
        ValueError: If ``parameter`` is not ``"azimuth"`` or ``"elevation"``.
    """
    pos = validate_positions(positions)
    wavenumber = 2.0 * np.pi / wavelength(freq_hz)
    response = steering_vector(pos, az_deg=az_deg, el_deg=el_deg, freq_hz=freq_hz)
    dtheta = _angle_derivative_unit_vector(az_deg, el_deg, parameter)
    derivative: NDArray[np.complex128] = (-1j * wavenumber * (pos @ dtheta)) * response
    return derivative


def _noise_projector(manifold: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Projector onto the orthogonal complement of the array-manifold column space."""
    num_channels = manifold.shape[0]
    identity = np.eye(num_channels, dtype=np.complex128)
    projector: NDArray[np.complex128] = identity - manifold @ np.linalg.pinv(manifold)
    return projector


def _validate_sources(
    positions: NDArray[np.float64],
    azimuths_deg: ArrayLike,
    elevations_deg: ArrayLike,
    snapshots: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce + validate source-angle inputs shared by the CRLB entry points."""
    azimuths = np.atleast_1d(np.asarray(azimuths_deg, dtype=np.float64))
    elevations = np.atleast_1d(np.asarray(elevations_deg, dtype=np.float64))
    if azimuths.ndim != 1 or azimuths.shape != elevations.shape:
        raise SourceCountError("azimuths_deg and elevations_deg must be 1-D and equal length")
    num_sources = azimuths.size
    if num_sources < 1:
        raise SourceCountError("at least one source is required")
    if num_sources >= positions.shape[0]:
        raise SourceCountError(
            f"{num_sources} sources is too many for a {positions.shape[0]}-element array"
        )
    if snapshots < 1:
        raise ValueError(f"snapshots must be positive, got {snapshots}")
    return azimuths, elevations


def crlb_azimuth(
    positions: ArrayLike,
    *,
    freq_hz: float,
    azimuths_deg: ArrayLike,
    elevations_deg: ArrayLike,
    snr_db: float,
    snapshots: int,
) -> NDArray[np.float64]:
    """Azimuth CRLB for one or more equal-power, uncorrelated sources.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        freq_hz: RF frequency in hertz.
        azimuths_deg: Source azimuths in degrees, one per source.
        elevations_deg: Source elevations in degrees, one per source.
        snr_db: Per-source signal-to-noise ratio in decibels.
        snapshots: Number of snapshots used to estimate the covariance.

    Returns:
        Per-source azimuth-estimate variance in degrees^2; ``+inf`` for a source
        whose azimuth is un-estimable (e.g. a ULA endfire source).

    Raises:
        SourceCountError: If the source lists are malformed or there are too many.
        ValueError: If ``snapshots`` is not positive.
    """
    pos = validate_positions(positions)
    azimuths, elevations = _validate_sources(pos, azimuths_deg, elevations_deg, snapshots)
    num_sources = azimuths.size

    manifold = np.stack(
        [
            steering_vector(pos, az_deg=azimuths[k], el_deg=elevations[k], freq_hz=freq_hz)
            for k in range(num_sources)
        ],
        axis=1,
    )
    derivatives = np.stack(
        [
            steering_derivative(
                pos,
                az_deg=azimuths[k],
                el_deg=elevations[k],
                freq_hz=freq_hz,
                parameter="azimuth",
            )
            for k in range(num_sources)
        ],
        axis=1,
    )
    projector = _noise_projector(manifold)
    information = np.real(np.diag(derivatives.conj().T @ projector @ derivatives))

    snr_linear = 10.0 ** (snr_db / 10.0)
    factor = 1.0 / (2.0 * snapshots * snr_linear)
    threshold = _RANK_TOL * float(np.max(information))
    crlb_rad2 = np.full(num_sources, np.inf, dtype=np.float64)
    estimable = information > threshold
    crlb_rad2[estimable] = factor / information[estimable]
    result: NDArray[np.float64] = crlb_rad2 * _RAD2_TO_DEG2
    return result


def compute_crlb(
    positions: ArrayLike,
    *,
    freq_hz: float,
    snr_db: float,
    snapshots: int,
    direction_deg: float,
    elevation_deg: float = 0.0,
) -> float:
    """Azimuth CRLB for a single source — the standard CRLB-bounded test bar.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        freq_hz: RF frequency in hertz.
        snr_db: Signal-to-noise ratio in decibels.
        snapshots: Number of snapshots used to estimate the covariance.
        direction_deg: Source azimuth in degrees.
        elevation_deg: Source elevation in degrees.

    Returns:
        The azimuth-estimate variance in degrees^2 (``+inf`` if un-estimable).
    """
    return float(
        crlb_azimuth(
            positions,
            freq_hz=freq_hz,
            azimuths_deg=[direction_deg],
            elevations_deg=[elevation_deg],
            snr_db=snr_db,
            snapshots=snapshots,
        )[0]
    )


def crlb_ula_closed_form(
    num_elements: int,
    spacing_m: float,
    *,
    freq_hz: float,
    snr_db: float,
    snapshots: int,
    azimuth_deg: float,
) -> float:
    """Closed-form single-source azimuth CRLB for a uniform linear array.

    Uses the Stoica & Nehorai 1989 result
    ``CRB = 6 / (N * SNR * (k d sin az)^2 * M (M^2 - 1))`` (radians^2), an
    independent check on the geometry-agnostic :func:`compute_crlb`.

    Args:
        num_elements: Number of ULA elements ``M``.
        spacing_m: Inter-element spacing in metres.
        freq_hz: RF frequency in hertz.
        snr_db: Signal-to-noise ratio in decibels.
        snapshots: Number of snapshots.
        azimuth_deg: Source azimuth in degrees (measured from the array axis).

    Returns:
        The azimuth-estimate variance in degrees^2 (``+inf`` at endfire).

    Raises:
        SourceCountError: If ``num_elements < 2``.
        ValueError: If ``snapshots`` is not positive.
    """
    if num_elements < 2:
        raise SourceCountError(f"a ULA needs at least 2 elements, got {num_elements}")
    if snapshots < 1:
        raise ValueError(f"snapshots must be positive, got {snapshots}")
    wavenumber = 2.0 * np.pi / wavelength(freq_hz)
    snr_linear = 10.0 ** (snr_db / 10.0)
    sin_az = float(np.sin(np.deg2rad(azimuth_deg)))
    denominator = (
        snapshots
        * snr_linear
        * (wavenumber * spacing_m * sin_az) ** 2
        * num_elements
        * (num_elements**2 - 1)
    )
    if denominator <= 0.0:
        return float("inf")
    return float(6.0 / denominator * _RAD2_TO_DEG2)


def crlb_joint_azimuth_elevation(
    positions: ArrayLike,
    *,
    freq_hz: float,
    azimuths_deg: ArrayLike,
    elevations_deg: ArrayLike,
    snr_db: float,
    snapshots: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Joint azimuth/elevation CRLB, per source.

    Builds the 2x2 Fisher information for each source. A rank-deficient block
    means the parameter is un-estimable and its bound is ``+inf`` — an in-plane
    source on a planar array (zero elevation information), or any source on a ULA
    (azimuth and elevation collapse onto the same cone).

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        freq_hz: RF frequency in hertz.
        azimuths_deg: Source azimuths in degrees, one per source.
        elevations_deg: Source elevations in degrees, one per source.
        snr_db: Per-source signal-to-noise ratio in decibels.
        snapshots: Number of snapshots.

    Returns:
        A ``(azimuth_variance, elevation_variance)`` pair of degrees^2 arrays.

    Raises:
        SourceCountError: If the source lists are malformed or there are too many.
        ValueError: If ``snapshots`` is not positive.
    """
    pos = validate_positions(positions)
    azimuths, elevations = _validate_sources(pos, azimuths_deg, elevations_deg, snapshots)
    num_sources = azimuths.size

    manifold = np.stack(
        [
            steering_vector(pos, az_deg=azimuths[k], el_deg=elevations[k], freq_hz=freq_hz)
            for k in range(num_sources)
        ],
        axis=1,
    )
    projector = _noise_projector(manifold)
    snr_linear = 10.0 ** (snr_db / 10.0)
    factor = 1.0 / (2.0 * snapshots * snr_linear)

    var_az = np.empty(num_sources, dtype=np.float64)
    var_el = np.empty(num_sources, dtype=np.float64)
    for k in range(num_sources):
        d_az = steering_derivative(
            pos, az_deg=azimuths[k], el_deg=elevations[k], freq_hz=freq_hz, parameter="azimuth"
        )
        d_el = steering_derivative(
            pos, az_deg=azimuths[k], el_deg=elevations[k], freq_hz=freq_hz, parameter="elevation"
        )
        h_aa = float(np.real(d_az.conj() @ projector @ d_az))
        h_ee = float(np.real(d_el.conj() @ projector @ d_el))
        h_ae = float(np.real(d_az.conj() @ projector @ d_el))
        scale = max(h_aa, h_ee, _RANK_TOL)
        if h_ee <= _RANK_TOL * scale:
            var_el[k] = np.inf
            var_az[k] = factor / h_aa if h_aa > _RANK_TOL * scale else np.inf
        elif (h_aa * h_ee - h_ae * h_ae) <= _RANK_TOL * h_aa * h_ee:
            var_az[k] = np.inf
            var_el[k] = np.inf
        else:
            inverse = np.linalg.inv(np.array([[h_aa, h_ae], [h_ae, h_ee]], dtype=np.float64))
            var_az[k] = factor * float(inverse[0, 0])
            var_el[k] = factor * float(inverse[1, 1])

    return var_az * _RAD2_TO_DEG2, var_el * _RAD2_TO_DEG2

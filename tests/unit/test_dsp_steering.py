"""Unit tests for rfdf.dsp.steering — the array-manifold convention anchor."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rfdf.backends.sdr.mock import CWEmitter, MockSdrScenario
from rfdf.backends.sdr.mock import create as create_mock_sdr
from rfdf.dsp.errors import InvalidGeometryError
from rfdf.dsp.geometry_presets import planar_cross, ula
from rfdf.dsp.steering import (
    SteeringManifold,
    build_manifold,
    direction_unit_vector,
    steering_vector,
    wavelength,
)
from rfdf.hal import SdrConfig


def test_wavelength_at_868_mhz() -> None:
    """wavelength() returns c / f."""
    assert wavelength(868e6) == pytest.approx(0.34538, abs=1e-4)


def test_wavelength_rejects_nonpositive() -> None:
    """A non-positive frequency has no physical wavelength."""
    with pytest.raises(ValueError, match="positive"):
        wavelength(0.0)


def test_direction_unit_vector_cardinal_directions() -> None:
    """az/el of (0,0), (90,0), (0,90) map to +x, +y, +z."""
    np.testing.assert_allclose(direction_unit_vector(0.0, 0.0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(direction_unit_vector(90.0, 0.0), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(direction_unit_vector(0.0, 90.0), [0, 0, 1], atol=1e-12)


def test_direction_unit_vector_is_unit_norm() -> None:
    """The propagation vector is always a unit vector."""
    assert np.linalg.norm(direction_unit_vector(37.0, 21.0)) == pytest.approx(1.0)


def test_direction_unit_vector_vectorized() -> None:
    """Array inputs produce a (G, 3) stack of unit vectors."""
    az = np.array([0.0, 90.0, 45.0])
    el = np.array([0.0, 0.0, 0.0])
    assert direction_unit_vector(az, el).shape == (3, 3)


def test_steering_vector_origin_element_has_zero_phase() -> None:
    """The element at the array origin always has unit (zero-phase) response."""
    a = steering_vector(planar_cross(0.17), az_deg=33.0, el_deg=7.0, freq_hz=868e6)
    assert a[0] == pytest.approx(1.0 + 0.0j)


def test_steering_vector_matches_mock_convention() -> None:
    """steering_vector reproduces the mock SDR's exp(-j) array factor exactly."""
    positions = planar_cross(0.17)
    az_deg, el_deg, freq = 41.0, 12.0, 868e6
    a = steering_vector(positions, az_deg=az_deg, el_deg=el_deg, freq_hz=freq)
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    theta_hat = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    lam = 299_792_458.0 / freq
    expected = np.exp(-1j * (2.0 * np.pi / lam) * (positions @ theta_hat))
    np.testing.assert_allclose(a, expected, atol=1e-12)


def test_steering_vector_rejects_bad_positions() -> None:
    """Malformed positions raise InvalidGeometryError, not a NumPy error."""
    with pytest.raises(InvalidGeometryError):
        steering_vector(np.zeros((4, 2)), az_deg=0.0, el_deg=0.0, freq_hz=868e6)


def test_build_manifold_shape_and_rows() -> None:
    """build_manifold rows equal steering_vector at each grid point."""
    positions = ula(6, 0.0875)
    az_grid = np.arange(-60.0, 60.0, 10.0)
    manifold = build_manifold(positions, az_grid, np.array([0.0]), freq_hz=5.8e9)
    assert isinstance(manifold, SteeringManifold)
    assert manifold.matrix.shape == (az_grid.size, 6)
    assert manifold.num_points == az_grid.size
    assert manifold.num_elements == 6
    for i, az in enumerate(az_grid):
        expected = steering_vector(positions, az_deg=az, el_deg=0.0, freq_hz=5.8e9)
        np.testing.assert_allclose(manifold.matrix[i], expected, atol=1e-12)


def test_steering_manifold_direction_lookup() -> None:
    """SteeringManifold.direction maps a flat index back to (az, el), az-major."""
    manifold = build_manifold(
        planar_cross(0.05), np.array([10.0, 20.0, 30.0]), np.array([0.0, 5.0]), freq_hz=5.8e9
    )
    assert manifold.num_points == 6
    assert manifold.grid_shape == (3, 2)
    assert manifold.direction(3) == (20.0, 5.0)


def test_build_manifold_music_peak_on_mock_iq() -> None:
    """A MUSIC pseudospectrum built on build_manifold peaks at a mock emitter."""
    true_az = 35.0
    scenario = MockSdrScenario(
        emitters=[CWEmitter(azimuth_deg=true_az, elevation_deg=0.0, power_dbm=0.0)],
        snr_db=30.0,
    )
    sdr = create_mock_sdr(scenario=scenario, block_samples=4096, seed=1)
    asyncio.run(
        sdr.configure(SdrConfig(center_freq_hz=868e6, sample_rate_hz=2e6, channels=list(range(5))))
    )
    asyncio.run(sdr.start())

    async def grab() -> np.ndarray:  # type: ignore[type-arg]
        async for block in sdr.stream():
            await sdr.stop()
            return block.iq.astype(np.complex128)
        raise AssertionError("no block")

    iq = asyncio.run(grab())
    cov = (iq @ iq.conj().T) / iq.shape[1]
    _, eigvecs = np.linalg.eigh(cov)
    noise = eigvecs[:, :-1]  # M-K = 4 noise eigenvectors

    az_grid = np.arange(-180.0, 180.0, 0.5)
    manifold = build_manifold(planar_cross(0.17), az_grid, np.array([0.0]), freq_hz=868e6)
    proj = manifold.matrix @ noise.conj()  # (G, M-K) = U_n^H a
    spectrum = 1.0 / np.maximum(np.sum(np.abs(proj) ** 2, axis=1), 1e-12)
    peak_az = az_grid[int(np.argmax(spectrum))]
    assert abs(peak_az - true_az) <= 2.0


def test_build_manifold_rejects_non_1d_grids() -> None:
    """Azimuth and elevation grids must be 1-D arrays."""
    with pytest.raises(ValueError, match="1-D"):
        build_manifold(planar_cross(0.1), np.zeros((2, 2)), np.array([0.0]), freq_hz=2.4e9)


@given(st.floats(-180.0, 180.0), st.floats(-90.0, 90.0))
def test_steering_vector_elements_are_unit_modulus(az_deg: float, el_deg: float) -> None:
    """Every steering-vector element has magnitude 1 for any direction."""
    a = steering_vector(planar_cross(0.1), az_deg=az_deg, el_deg=el_deg, freq_hz=2.4e9)
    np.testing.assert_allclose(np.abs(a), 1.0, atol=1e-9)

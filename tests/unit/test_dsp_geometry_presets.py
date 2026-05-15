"""Unit tests for rfdf.dsp.geometry_presets."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rfdf.dsp.errors import InvalidGeometryError
from rfdf.dsp.geometry_presets import (
    half_wavelength_spacing,
    is_ula,
    planar_cross,
    ula,
    validate_positions,
)


def test_ula_lays_elements_along_x_axis() -> None:
    """ula() returns equispaced elements along +x with y = z = 0."""
    positions = ula(num_elements=4, spacing_m=0.1)
    assert positions.shape == (4, 3)
    np.testing.assert_allclose(positions[:, 0], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(positions[:, 1:], 0.0)


def test_ula_rejects_too_few_elements() -> None:
    """A ULA needs at least two elements."""
    with pytest.raises(InvalidGeometryError, match="elements"):
        ula(num_elements=1, spacing_m=0.1)


def test_ula_rejects_nonpositive_spacing() -> None:
    """Spacing must be a positive distance."""
    with pytest.raises(InvalidGeometryError, match="spacing"):
        ula(num_elements=4, spacing_m=0.0)


def test_planar_cross_is_the_reference_five_element_array() -> None:
    """planar_cross() returns origin + four elements at +/-x, +/-y."""
    positions = planar_cross(spacing_m=0.17)
    assert positions.shape == (5, 3)
    expected = np.array(
        [[0, 0, 0], [0.17, 0, 0], [0, 0.17, 0], [-0.17, 0, 0], [0, -0.17, 0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(positions, expected)


def test_is_ula_true_for_a_ula() -> None:
    """A uniform linear array is recognised as one."""
    assert is_ula(ula(num_elements=6, spacing_m=0.0875)) is True


def test_is_ula_axis_agnostic() -> None:
    """A ULA aligned with +y (not +x) is still recognised."""
    positions = np.array([[0, 0, 0], [0, 0.1, 0], [0, 0.2, 0], [0, 0.3, 0]], dtype=np.float64)
    assert is_ula(positions) is True


def test_is_ula_false_for_planar_cross() -> None:
    """The 5-element planar cross is not a ULA."""
    assert is_ula(planar_cross(spacing_m=0.17)) is False


def test_is_ula_false_for_collinear_but_unequal_spacing() -> None:
    """Collinear elements with non-uniform spacing are not a ULA."""
    positions = np.array([[0, 0, 0], [0.1, 0, 0], [0.3, 0, 0]], dtype=np.float64)
    assert is_ula(positions) is False


def test_validate_positions_rejects_wrong_shape() -> None:
    """validate_positions() rejects anything that is not (M, 3)."""
    with pytest.raises(InvalidGeometryError, match=r"\(M, 3\)"):
        validate_positions(np.zeros((4, 2)))


def test_half_wavelength_spacing_at_868_mhz() -> None:
    """lambda/2 at 868 MHz is ~0.1727 m."""
    assert half_wavelength_spacing(868e6) == pytest.approx(0.17269, abs=1e-4)


def test_half_wavelength_spacing_rejects_nonpositive_frequency() -> None:
    """A non-positive frequency has no physical spacing."""
    with pytest.raises(ValueError, match="positive"):
        half_wavelength_spacing(0.0)


def test_planar_cross_rejects_nonpositive_spacing() -> None:
    """planar_cross spacing must be a positive distance."""
    with pytest.raises(InvalidGeometryError, match="spacing"):
        planar_cross(0.0)


def test_validate_positions_rejects_empty_array() -> None:
    """An array with no antennas is rejected."""
    with pytest.raises(InvalidGeometryError, match="at least one"):
        validate_positions(np.zeros((0, 3)))


def test_is_ula_false_for_single_element() -> None:
    """A single antenna does not form a ULA."""
    assert is_ula(np.zeros((1, 3))) is False


def test_is_ula_false_for_coincident_elements() -> None:
    """Coincident elements (zero span) do not form a ULA."""
    assert is_ula(np.zeros((3, 3))) is False


@given(st.integers(min_value=2, max_value=32), st.floats(min_value=0.01, max_value=1.0))
def test_ula_is_always_recognised_as_a_ula(num_elements: int, spacing_m: float) -> None:
    """Any array produced by ula() round-trips through is_ula() as True."""
    positions = ula(num_elements=num_elements, spacing_m=spacing_m)
    assert positions.shape == (num_elements, 3)
    assert is_ula(positions) is True

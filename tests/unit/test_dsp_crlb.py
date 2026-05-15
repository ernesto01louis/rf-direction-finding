"""Unit tests for rfdf.dsp.crlb — the mathematical-integrity bar."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rfdf.dsp.crlb import (
    compute_crlb,
    crlb_azimuth,
    crlb_joint_azimuth_elevation,
    crlb_ula_closed_form,
    steering_derivative,
)
from rfdf.dsp.errors import SourceCountError
from rfdf.dsp.geometry_presets import half_wavelength_spacing, planar_cross, ula

FREQ = 2.4e9
HALF_LAMBDA = half_wavelength_spacing(FREQ)


def test_compute_crlb_is_positive_and_finite() -> None:
    """A well-posed ULA scenario has a positive, finite CRLB."""
    crlb = compute_crlb(
        ula(5, HALF_LAMBDA), freq_hz=FREQ, snr_db=10.0, snapshots=1000, direction_deg=90.0
    )
    assert np.isfinite(crlb)
    assert crlb > 0.0


def test_crlb_ula_closed_form_matches_hand_computed_value() -> None:
    """The closed-form ULA CRLB equals the hand-derived Stoica-Nehorai value."""
    crlb = crlb_ula_closed_form(
        4, HALF_LAMBDA, freq_hz=FREQ, snr_db=10.0, snapshots=1000, azimuth_deg=90.0
    )
    # M=4, d=lambda/2 (k*d = pi), broadside (sin az = 1), N=1000, SNR = 10x.
    expected_rad2 = 6.0 / (1000 * 10.0 * np.pi**2 * 4 * (4**2 - 1))
    expected_deg2 = expected_rad2 * (180.0 / np.pi) ** 2
    assert crlb == pytest.approx(expected_deg2, rel=1e-9)


def test_compute_crlb_matches_closed_form_on_ula() -> None:
    """The geometry-agnostic CRLB agrees with the closed form across azimuths."""
    for az in (30.0, 60.0, 90.0, 120.0):
        general = compute_crlb(
            ula(6, HALF_LAMBDA), freq_hz=FREQ, snr_db=15.0, snapshots=512, direction_deg=az
        )
        closed = crlb_ula_closed_form(
            6, HALF_LAMBDA, freq_hz=FREQ, snr_db=15.0, snapshots=512, azimuth_deg=az
        )
        assert general == pytest.approx(closed, rel=1e-6)


def test_crlb_scales_inversely_with_snapshots() -> None:
    """Doubling the snapshots halves the CRLB."""
    base = compute_crlb(
        ula(5, HALF_LAMBDA), freq_hz=FREQ, snr_db=10.0, snapshots=1000, direction_deg=80.0
    )
    doubled = compute_crlb(
        ula(5, HALF_LAMBDA), freq_hz=FREQ, snr_db=10.0, snapshots=2000, direction_deg=80.0
    )
    assert doubled == pytest.approx(base / 2.0, rel=1e-9)


def test_crlb_scales_inversely_with_snr() -> None:
    """A 10 dB SNR increase (10x linear) divides the CRLB by 10."""
    low = compute_crlb(
        ula(5, HALF_LAMBDA), freq_hz=FREQ, snr_db=10.0, snapshots=1000, direction_deg=80.0
    )
    high = compute_crlb(
        ula(5, HALF_LAMBDA), freq_hz=FREQ, snr_db=20.0, snapshots=1000, direction_deg=80.0
    )
    assert high == pytest.approx(low / 10.0, rel=1e-9)


def test_crlb_decreases_with_more_elements() -> None:
    """A larger array tightens the bound."""
    small = compute_crlb(
        ula(4, HALF_LAMBDA), freq_hz=FREQ, snr_db=10.0, snapshots=1000, direction_deg=90.0
    )
    large = compute_crlb(
        ula(10, HALF_LAMBDA), freq_hz=FREQ, snr_db=10.0, snapshots=1000, direction_deg=90.0
    )
    assert large < small


def test_crlb_infinite_at_ula_endfire() -> None:
    """A ULA has no azimuth resolution at endfire (az = 0)."""
    crlb = compute_crlb(
        ula(5, HALF_LAMBDA), freq_hz=FREQ, snr_db=10.0, snapshots=1000, direction_deg=0.0
    )
    assert not np.isfinite(crlb)


def test_compute_crlb_finite_on_planar_cross() -> None:
    """The planar cross resolves azimuth — a finite CRLB."""
    crlb = compute_crlb(
        planar_cross(HALF_LAMBDA),
        freq_hz=FREQ,
        snr_db=10.0,
        snapshots=1000,
        direction_deg=35.0,
        elevation_deg=0.0,
    )
    assert np.isfinite(crlb)
    assert crlb > 0.0


def test_crlb_azimuth_returns_one_bound_per_source() -> None:
    """crlb_azimuth gives a positive, finite bound for each well-separated source."""
    bounds = crlb_azimuth(
        planar_cross(HALF_LAMBDA),
        freq_hz=FREQ,
        azimuths_deg=[20.0, 70.0, -40.0],
        elevations_deg=[0.0, 0.0, 0.0],
        snr_db=15.0,
        snapshots=2000,
    )
    assert bounds.shape == (3,)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds > 0.0)


def test_crlb_azimuth_rejects_too_many_sources() -> None:
    """More sources than (array size - 1) cannot be resolved."""
    with pytest.raises(SourceCountError, match="too many"):
        crlb_azimuth(
            ula(3, HALF_LAMBDA),
            freq_hz=FREQ,
            azimuths_deg=[10.0, 20.0, 30.0],
            elevations_deg=[0.0, 0.0, 0.0],
            snr_db=15.0,
            snapshots=1000,
        )


def test_joint_elevation_is_infinite_for_in_plane_source_on_cross() -> None:
    """A source in the plane of the planar cross has un-estimable elevation."""
    var_az, var_el = crlb_joint_azimuth_elevation(
        planar_cross(HALF_LAMBDA),
        freq_hz=FREQ,
        azimuths_deg=[35.0],
        elevations_deg=[0.0],
        snr_db=20.0,
        snapshots=4000,
    )
    assert np.isfinite(var_az[0])
    assert not np.isfinite(var_el[0])


def test_joint_crlb_both_infinite_on_ula() -> None:
    """A ULA cannot jointly resolve azimuth and elevation (cone ambiguity)."""
    var_az, var_el = crlb_joint_azimuth_elevation(
        ula(6, HALF_LAMBDA),
        freq_hz=FREQ,
        azimuths_deg=[50.0],
        elevations_deg=[15.0],
        snr_db=20.0,
        snapshots=4000,
    )
    assert not np.isfinite(var_az[0])
    assert not np.isfinite(var_el[0])


@given(st.floats(min_value=20.0, max_value=160.0), st.integers(min_value=128, max_value=4096))
def test_crlb_is_always_positive(azimuth_deg: float, snapshots: int) -> None:
    """The CRLB is strictly positive for any non-endfire azimuth."""
    crlb = compute_crlb(
        ula(5, HALF_LAMBDA),
        freq_hz=FREQ,
        snr_db=10.0,
        snapshots=snapshots,
        direction_deg=azimuth_deg,
    )
    assert crlb > 0.0


def test_steering_derivative_rejects_unknown_parameter() -> None:
    """Only azimuth and elevation derivatives are defined."""
    with pytest.raises(ValueError, match="must be 'azimuth' or 'elevation'"):
        steering_derivative(
            ula(4, HALF_LAMBDA), az_deg=30.0, el_deg=0.0, freq_hz=FREQ, parameter="range"
        )


def test_crlb_azimuth_rejects_mismatched_source_lists() -> None:
    """Azimuth and elevation lists must be the same length."""
    with pytest.raises(SourceCountError, match="equal length"):
        crlb_azimuth(
            ula(5, HALF_LAMBDA),
            freq_hz=FREQ,
            azimuths_deg=[10.0, 20.0],
            elevations_deg=[0.0],
            snr_db=10.0,
            snapshots=1000,
        )


def test_crlb_azimuth_rejects_empty_source_list() -> None:
    """At least one source is required."""
    with pytest.raises(SourceCountError, match="at least one"):
        crlb_azimuth(
            ula(5, HALF_LAMBDA),
            freq_hz=FREQ,
            azimuths_deg=[],
            elevations_deg=[],
            snr_db=10.0,
            snapshots=1000,
        )


def test_crlb_azimuth_rejects_nonpositive_snapshots() -> None:
    """The snapshot count must be positive."""
    with pytest.raises(ValueError, match="snapshots"):
        crlb_azimuth(
            ula(5, HALF_LAMBDA),
            freq_hz=FREQ,
            azimuths_deg=[30.0],
            elevations_deg=[0.0],
            snr_db=10.0,
            snapshots=0,
        )


def test_crlb_ula_closed_form_rejects_too_few_elements() -> None:
    """The closed form needs at least two elements."""
    with pytest.raises(SourceCountError, match="2 elements"):
        crlb_ula_closed_form(
            1, HALF_LAMBDA, freq_hz=FREQ, snr_db=10.0, snapshots=1000, azimuth_deg=90.0
        )


def test_crlb_ula_closed_form_rejects_nonpositive_snapshots() -> None:
    """The closed form needs a positive snapshot count."""
    with pytest.raises(ValueError, match="snapshots"):
        crlb_ula_closed_form(
            5, HALF_LAMBDA, freq_hz=FREQ, snr_db=10.0, snapshots=0, azimuth_deg=90.0
        )


def test_crlb_ula_closed_form_infinite_at_endfire() -> None:
    """The closed form is infinite at ULA endfire (az = 0)."""
    crlb = crlb_ula_closed_form(
        5, HALF_LAMBDA, freq_hz=FREQ, snr_db=10.0, snapshots=1000, azimuth_deg=0.0
    )
    assert not np.isfinite(crlb)


def test_joint_crlb_finite_for_elevated_source_on_cross() -> None:
    """The planar cross resolves both azimuth and elevation for an out-of-plane source."""
    var_az, var_el = crlb_joint_azimuth_elevation(
        planar_cross(HALF_LAMBDA),
        freq_hz=FREQ,
        azimuths_deg=[35.0],
        elevations_deg=[25.0],
        snr_db=20.0,
        snapshots=4000,
    )
    assert np.isfinite(var_az[0])
    assert np.isfinite(var_el[0])
    assert var_az[0] > 0.0
    assert var_el[0] > 0.0

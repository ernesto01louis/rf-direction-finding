"""Unit tests for the mock rotator backend."""

from __future__ import annotations

import asyncio

import pytest

from rfdf.backends.rotator.mock import create as create_mock_rotator
from rfdf.hal import RotatorController


def test_mock_satisfies_protocol() -> None:
    """The mock is a structural RotatorController."""
    rot = create_mock_rotator()
    assert isinstance(rot, RotatorController)


def test_capabilities_match_construction_args() -> None:
    """Reported ranges + speed + accuracy reflect constructor inputs."""
    rot = create_mock_rotator(
        azimuth_range_deg=(0.0, 180.0),
        elevation_range_deg=(-5.0, 90.0),
        max_speed_deg_per_s=12.0,
        positioning_accuracy_deg=0.1,
    )
    assert rot.azimuth_range_deg == (0.0, 180.0)
    assert rot.elevation_range_deg == (-5.0, 90.0)
    assert rot.max_speed_deg_per_s == pytest.approx(12.0)
    assert rot.positioning_accuracy_deg == pytest.approx(0.1)
    assert rot.supports_azimuth is True
    assert rot.supports_elevation is True


def test_goto_settles_within_accuracy() -> None:
    """After goto + position, the report is within positioning_accuracy_deg."""
    rot = create_mock_rotator(
        max_speed_deg_per_s=360.0,  # fast so the test doesn't sleep long
        positioning_accuracy_deg=0.5,
        seed=42,
    )
    asyncio.run(rot.goto(45.0, 30.0))
    az, el = asyncio.run(rot.position())
    # Should be within 5-sigma of the target.
    assert abs(az - 45.0) < 5 * 0.5
    assert abs(el - 30.0) < 5 * 0.5


def test_goto_out_of_range_raises() -> None:
    """Targets outside the configured range raise ValueError."""
    rot = create_mock_rotator(
        azimuth_range_deg=(0.0, 360.0),
        elevation_range_deg=(0.0, 90.0),
    )
    with pytest.raises(ValueError, match="azimuth"):
        asyncio.run(rot.goto(400.0, 30.0))
    with pytest.raises(ValueError, match="elevation"):
        asyncio.run(rot.goto(0.0, 95.0))


def test_park_returns_to_lower_left_corner() -> None:
    """park() moves to (az_min, el_min)."""
    rot = create_mock_rotator(
        azimuth_range_deg=(10.0, 350.0),
        elevation_range_deg=(5.0, 85.0),
        max_speed_deg_per_s=720.0,
        positioning_accuracy_deg=0.0,
        seed=0,
    )
    asyncio.run(rot.park())
    az, el = asyncio.run(rot.position())
    assert az == pytest.approx(10.0)
    assert el == pytest.approx(5.0)


def test_stream_position_yields_until_settle() -> None:
    """stream_position() emits at least one sample and converges on the target."""
    rot = create_mock_rotator(
        max_speed_deg_per_s=360.0,
        positioning_accuracy_deg=0.0,
        slew_tick_s=0.001,
        seed=0,
    )
    # Set up a target without awaiting the move to completion.
    rot._target_az = 30.0  # test-only direct write to seed the stream
    rot._target_el = 0.0

    async def drain() -> list[tuple[float, float]]:
        return [pos async for pos in rot.stream_position()]

    samples = asyncio.run(drain())
    assert len(samples) >= 2
    last_az, last_el = samples[-1]
    assert last_az == pytest.approx(30.0)
    assert last_el == pytest.approx(0.0)


def test_calibrate_homes_position() -> None:
    """calibrate() resets to the (az_min, el_min) corner with ok=True."""
    rot = create_mock_rotator(
        azimuth_range_deg=(0.0, 360.0),
        elevation_range_deg=(0.0, 90.0),
        max_speed_deg_per_s=720.0,
        positioning_accuracy_deg=0.0,
    )
    asyncio.run(rot.goto(180.0, 45.0))
    report = asyncio.run(rot.calibrate())
    assert report.ok is True
    assert report.backend == "mock"
    az, el = asyncio.run(rot.position())
    assert az == pytest.approx(0.0)
    assert el == pytest.approx(0.0)


def test_invalid_construction_args() -> None:
    """Invalid constructor inputs raise ValueError."""
    with pytest.raises(ValueError, match="azimuth_range"):
        create_mock_rotator(azimuth_range_deg=(180.0, 0.0))
    with pytest.raises(ValueError, match="elevation_range"):
        create_mock_rotator(elevation_range_deg=(45.0, 0.0))
    with pytest.raises(ValueError, match="max_speed"):
        create_mock_rotator(max_speed_deg_per_s=0.0)

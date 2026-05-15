"""Property-based contract for :class:`rfdf.hal.RotatorController`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rfdf.hal import RotatorController


def test_isinstance_protocol(rotator_factories: list[tuple[str, Callable[..., Any]]]) -> None:
    """Every rotator backend is a structural RotatorController."""
    assert rotator_factories, "no rotator backends discovered"
    for name, factory in rotator_factories:
        rot = factory()
        assert isinstance(rot, RotatorController), name


def test_capabilities_well_formed(
    rotator_factories: list[tuple[str, Callable[..., Any]]],
) -> None:
    """az/el ranges and max_speed have sane values for every backend."""
    for name, factory in rotator_factories:
        rot = factory()
        az_lo, az_hi = rot.azimuth_range_deg
        el_lo, el_hi = rot.elevation_range_deg
        assert az_lo < az_hi, name
        assert el_lo < el_hi, name
        assert rot.max_speed_deg_per_s > 0, name
        assert rot.positioning_accuracy_deg >= 0, name


@settings(
    derandomize=True,
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    az=st.floats(min_value=0.0, max_value=180.0),
    el=st.floats(min_value=0.0, max_value=80.0),
)
def test_goto_then_position_within_accuracy(
    az: float,
    el: float,
    rotator_factories: list[tuple[str, Callable[..., Any]]],
) -> None:
    """After goto(az, el), position() is within positioning_accuracy_deg (5-sigma)."""
    for name, factory in rotator_factories:
        rot = factory(max_speed_deg_per_s=720.0)  # fast slew so tests don't sleep

        async def run(r: object = rot, n: str = name) -> None:
            await r.goto(az, el)  # type: ignore[attr-defined]
            cur_az, cur_el = await r.position()  # type: ignore[attr-defined]
            tol = 5 * (r.positioning_accuracy_deg + 1e-3)  # type: ignore[attr-defined]
            assert abs(cur_az - az) <= tol, (n, cur_az, az)
            assert abs(cur_el - el) <= tol, (n, cur_el, el)

        asyncio.run(run())

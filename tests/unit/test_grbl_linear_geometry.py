"""Unit tests for the GRBL linear-rail geometry backend.

The rail geometry maths (projection, on-axis validation, offset->position) are
pure and fully tested here. The GRBL move/home network paths need a real
controller and are covered by ``tests/hardware/test_grbl_linear_hardware.py``.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from rfdf.backends.geometry.grbl_linear import (
    GrblLinearGeometry,
    PositionRepeatabilityReport,
    RailConfig,
    _offset_to_position,
    _solve_rail_offset,
    create,
)
from rfdf.hal import GeometryController

_AXES = ("X", "Y", "Z", "A", "B")


def _rails() -> list[RailConfig]:
    """Five rails radiating from the origin along five distinct GRBL axes."""
    directions = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
    ]
    return [
        RailConfig(
            antenna_id=i,
            origin=(0.0, 0.0, 0.0),
            direction=directions[i],
            travel_m=1.0,
            grbl_axis=_AXES[i],
        )
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# RailConfig validation
# ---------------------------------------------------------------------------


def test_railconfig_normalises_direction() -> None:
    """A non-unit direction vector is normalised on construction."""
    rail = RailConfig(
        antenna_id=0, origin=(0, 0, 0), direction=(3.0, 4.0, 0.0), travel_m=1.0, grbl_axis="X"
    )
    assert pytest.approx(rail.direction) == (0.6, 0.8, 0.0)


def test_railconfig_rejects_zero_direction() -> None:
    """A zero direction vector is rejected."""
    with pytest.raises(ValueError, match="non-zero vector"):
        RailConfig(
            antenna_id=0, origin=(0, 0, 0), direction=(0.0, 0.0, 0.0), travel_m=1.0, grbl_axis="X"
        )


def test_railconfig_rejects_unknown_axis() -> None:
    """An unknown GRBL axis letter is rejected."""
    with pytest.raises(ValueError, match="grbl_axis"):
        RailConfig(
            antenna_id=0, origin=(0, 0, 0), direction=(1.0, 0.0, 0.0), travel_m=1.0, grbl_axis="Q"
        )


# ---------------------------------------------------------------------------
# Rail-projection maths
# ---------------------------------------------------------------------------


def test_solve_rail_offset_on_axis() -> None:
    """A target on the rail axis yields the correct offset."""
    rail = _rails()[0]  # origin, +x, travel 1 m
    offset = _solve_rail_offset(np.array([0.42, 0.0, 0.0]), rail)
    assert offset == pytest.approx(0.42)


def test_solve_rail_offset_rejects_off_axis() -> None:
    """A target off the rail axis is rejected."""
    rail = _rails()[0]
    with pytest.raises(ValueError, match="off rail axis"):
        _solve_rail_offset(np.array([0.3, 0.05, 0.0]), rail)


def test_solve_rail_offset_rejects_beyond_travel() -> None:
    """A target beyond the rail travel is rejected."""
    rail = _rails()[0]
    with pytest.raises(ValueError, match="reachable travel"):
        _solve_rail_offset(np.array([1.5, 0.0, 0.0]), rail)


def test_offset_to_position_round_trips() -> None:
    """offset -> position -> offset is an identity on the rail axis."""
    rail = _rails()[3]  # diagonal (1,1,0)/sqrt2
    pos = _offset_to_position(rail, 0.5)
    assert _solve_rail_offset(pos, rail) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Construction + capabilities (no network)
# ---------------------------------------------------------------------------


def test_create_requires_rails_and_host() -> None:
    """The factory rejects a call missing rails or the controller host."""
    with pytest.raises(ValueError, match="rails is required"):
        create(controller_host="geom.local")
    with pytest.raises(ValueError, match="controller_host is required"):
        create(rails=_rails())


def test_construction_rejects_non_contiguous_ids() -> None:
    """antenna_id values must form 0..N-1."""
    rails = _rails()[:3]
    rails[2] = RailConfig(
        antenna_id=9, origin=(0, 0, 0), direction=(0, 0, 1), travel_m=1.0, grbl_axis="Z"
    )
    with pytest.raises(ValueError, match="antenna_id values"):
        GrblLinearGeometry(rails, controller_host="geom.local")


def test_construction_rejects_duplicate_axes() -> None:
    """Each rail must drive a distinct GRBL axis."""
    rails = _rails()[:2]
    rails[1] = RailConfig(
        antenna_id=1, origin=(0, 0, 0), direction=(0, 1, 0), travel_m=1.0, grbl_axis="X"
    )
    with pytest.raises(ValueError, match="distinct GRBL axis"):
        GrblLinearGeometry(rails, controller_host="geom.local")


def test_capabilities() -> None:
    """A constructed backend reports a morphable 5-antenna geometry."""
    geo = create(rails=_rails(), controller_host="geom.local")
    assert geo.num_antennas == 5
    assert geo.is_morphable is True
    assert geo.positioning_repeatability_mm == 0.05


def test_grbl_linear_is_structural_geometry_controller() -> None:
    """GrblLinearGeometry structurally satisfies the GeometryController Protocol."""
    geo = create(rails=_rails(), controller_host="geom.local")
    assert isinstance(geo, GeometryController)


def test_goto_positions_rejects_bad_shape() -> None:
    """A position array of the wrong shape is rejected before any network I/O."""
    geo = create(rails=_rails(), controller_host="geom.local")

    async def run() -> None:
        with pytest.raises(ValueError, match="positions"):
            await geo.goto_positions(np.zeros((3, 3)))

    asyncio.run(run())


def test_goto_positions_rejects_off_axis_target() -> None:
    """A reachable-shape but off-axis target is rejected before any network I/O."""
    geo = create(rails=_rails(), controller_host="geom.local")
    target = np.zeros((5, 3))
    target[0] = [0.3, 0.2, 0.0]  # off the +x rail

    async def run() -> None:
        with pytest.raises(ValueError, match="off rail axis"):
            await geo.goto_positions(target)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Preset persistence (TOML round-trip, no network)
# ---------------------------------------------------------------------------


def test_preset_save_load_round_trip(tmp_path) -> None:
    """A saved preset round-trips through the TOML store."""
    preset_file = tmp_path / "geometry-presets.toml"
    geo = create(rails=_rails(), controller_host="geom.local", presets_path=preset_file)
    positions = np.zeros((5, 3))
    positions[0] = [0.17, 0.0, 0.0]
    positions[1] = [0.0, 0.17, 0.0]

    async def run() -> None:
        await geo.save_preset("uhf_compact", positions)
        assert await geo.list_presets() == ["uhf_compact"]
        # A fresh instance reads the same preset back from disk.
        reloaded = create(rails=_rails(), controller_host="geom.local", presets_path=preset_file)
        assert await reloaded.list_presets() == ["uhf_compact"]
        np.testing.assert_allclose(reloaded.preset_positions("uhf_compact"), positions)

    asyncio.run(run())


def test_save_preset_rejects_unreachable_position(tmp_path) -> None:
    """save_preset refuses a preset with an off-axis position."""
    geo = create(
        rails=_rails(),
        controller_host="geom.local",
        presets_path=tmp_path / "p.toml",
    )
    bad = np.zeros((5, 3))
    bad[0] = [0.3, 0.3, 0.0]

    async def run() -> None:
        with pytest.raises(ValueError, match="off rail axis"):
            await geo.save_preset("bad", bad)

    asyncio.run(run())


def test_position_repeatability_report_budget_flag() -> None:
    """PositionRepeatabilityReport.within_budget reflects max vs budget."""
    report = PositionRepeatabilityReport(
        iterations=50,
        max_deviation_mm=0.4,
        rms_deviation_mm=0.2,
        per_rail_max_mm=[0.4, 0.3],
        within_budget=True,
    )
    assert report.within_budget is True
    assert report.budget_mm == 1.0

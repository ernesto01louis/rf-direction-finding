"""Property-based contract for :class:`rfdf.hal.GeometryController`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np

from rfdf.hal import GeometryController


def test_isinstance_protocol(geometry_factories: list[tuple[str, Callable[..., Any]]]) -> None:
    """Every geometry backend is a structural GeometryController."""
    assert geometry_factories, "no geometry backends discovered"
    for name, factory in geometry_factories:
        geo = factory()
        assert isinstance(geo, GeometryController), name


def test_positions_shape_and_finite(
    geometry_factories: list[tuple[str, Callable[..., Any]]],
) -> None:
    """positions() returns a (N, 3) finite float array; N matches num_antennas."""
    for name, factory in geometry_factories:
        geo = factory()
        positions = asyncio.run(geo.positions())
        assert positions.shape == (geo.num_antennas, 3), name
        assert np.all(np.isfinite(positions)), name


def test_num_antennas_positive(
    geometry_factories: list[tuple[str, Callable[..., Any]]],
) -> None:
    """num_antennas > 0 for every backend."""
    for name, factory in geometry_factories:
        geo = factory()
        assert geo.num_antennas > 0, name


def test_calibrate_returns_report(
    geometry_factories: list[tuple[str, Callable[..., Any]]],
) -> None:
    """calibrate() yields a CalibrationReport with a backend identifier."""
    for name, factory in geometry_factories:
        geo = factory()
        report = asyncio.run(geo.calibrate())
        assert report.backend, name
        assert isinstance(report.ok, bool), name

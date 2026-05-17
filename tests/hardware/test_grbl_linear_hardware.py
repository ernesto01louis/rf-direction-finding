"""Hardware-in-the-loop tests for the GRBL linear-rail geometry backend.

Marked ``@pytest.mark.hardware`` — skipped unless ``RFDF_HARDWARE=1`` is set
(see ``tests/conftest.py``). The controller host comes from
``RFDF_GRBL_HOST``; the rail configuration is site-specific and never committed.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

from rfdf.backends.geometry.grbl_linear import (
    RailConfig,
    create,
    measure_position_repeatability,
)
from rfdf.hal import GeometryController

pytestmark = pytest.mark.hardware


def _host() -> str:
    host = os.environ.get("RFDF_GRBL_HOST", "").strip()
    if not host:
        pytest.skip("set RFDF_GRBL_HOST to the GRBL controller host")
    return host


def _reference_rails() -> list[RailConfig]:
    """A reference 5-rail set radiating from the origin along five GRBL axes.

    Replace the origins / directions / travels with the operator's measured
    mechanical build before a real commissioning run.
    """
    directions = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
    ]
    axes = ("X", "Y", "Z", "A", "B")
    return [
        RailConfig(
            antenna_id=i,
            origin=(0.0, 0.0, 0.0),
            direction=directions[i],
            travel_m=1.0,
            grbl_axis=axes[i],
        )
        for i in range(5)
    ]


def test_grbl_linear_is_structural_geometry_controller() -> None:
    """The constructed backend satisfies the GeometryController Protocol."""
    geo = create(rails=_reference_rails(), controller_host=_host())
    assert isinstance(geo, GeometryController)


def test_grbl_linear_homes() -> None:
    """calibrate() homes every rail to its origin-end limit switch."""
    geo = create(rails=_reference_rails(), controller_host=_host())

    async def run() -> None:
        report = await geo.calibrate()
        assert report.ok is True, report.message
        await geo.close()

    asyncio.run(run())


def test_grbl_linear_position_repeatability_budget() -> None:
    """Position repeatability over 10 runs per preset stays inside the budget.

    The full 50-iteration commissioning sweep is run by the operator via
    ``rfdf hw selftest``; this is a short hardware smoke check.
    """
    geo = create(rails=_reference_rails(), controller_host=_host())

    async def run() -> None:
        await geo.calibrate()
        positions = np.zeros((5, 3))
        positions[0] = [0.17, 0.0, 0.0]
        positions[1] = [0.0, 0.17, 0.0]
        await geo.save_preset("smoke", positions)
        report = await measure_position_repeatability(geo, iterations=10)
        assert report.within_budget, (
            f"max deviation {report.max_deviation_mm:.3f} mm exceeds {report.budget_mm} mm budget"
        )
        await geo.close()

    asyncio.run(run())

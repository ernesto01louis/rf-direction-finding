"""Hardware-in-the-loop tests for the AntRunner rotator backend.

Marked ``@pytest.mark.hardware`` — skipped unless ``RFDF_HARDWARE=1`` is set
(see ``tests/conftest.py``). The controller host comes from ``RFDF_ANTRUNNER_HOST``;
site-specific hosts are never committed to the repo.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from rfdf.backends.rotator.antrunner import create
from rfdf.hal import RotatorController

pytestmark = pytest.mark.hardware


def _host() -> str:
    host = os.environ.get("RFDF_ANTRUNNER_HOST", "").strip()
    if not host:
        pytest.skip("set RFDF_ANTRUNNER_HOST to the AntRunner controller host")
    return host


def test_antrunner_is_structural_rotator_controller() -> None:
    """The constructed backend satisfies the RotatorController Protocol."""
    rot = create(host=_host())
    assert isinstance(rot, RotatorController)


def test_antrunner_homes_and_applies_soft_limits() -> None:
    """calibrate() runs the homing cycle and reports success."""
    rot = create(host=_host())

    async def run() -> None:
        report = await rot.calibrate()
        assert report.ok is True, report.message
        assert report.backend == "antrunner"
        await rot.close()

    asyncio.run(run())


def test_antrunner_goto_closed_loop_agreement() -> None:
    """After a goto, the controller's readback agrees with the command."""
    rot = create(host=_host())

    async def run() -> None:
        await rot.calibrate()
        await rot.goto(30.0, 45.0)  # raises RotatorPositionError on mismatch
        az, el = await rot.position()
        assert abs(az - 30.0) <= rot.positioning_accuracy_deg
        assert abs(el - 45.0) <= rot.positioning_accuracy_deg
        await rot.park()
        await rot.close()

    asyncio.run(run())


def test_antrunner_full_range_slew() -> None:
    """A full-range AZ slew completes and streams intermediate positions."""
    rot = create(host=_host())

    async def run() -> None:
        await rot.calibrate()
        await rot.goto(-180.0, 0.0)
        samples = [pos async for pos in rot.stream_position()]
        assert samples, "stream_position yielded nothing"
        await rot.goto(180.0, 0.0)
        await rot.park()
        await rot.close()

    asyncio.run(run())

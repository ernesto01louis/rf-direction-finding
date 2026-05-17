"""Unit tests for the AntRunner rotator backend + shared GRBL client.

The GRBL parsers are pure and fully tested here. The HTTP transport and the
rotator's network paths need a real AntRunner controller and are covered by
``tests/hardware/test_antrunner_hardware.py`` under the ``hardware`` marker.
"""

from __future__ import annotations

import asyncio

import pytest

from rfdf.backends._grbl import (
    GrblConnectionError,
    GrblHttpClient,
    GrblStatus,
    parse_settings_dump,
    parse_status_report,
)
from rfdf.backends.rotator.antrunner import (
    AntRunnerError,
    AntRunnerRotator,
    RotatorNotHomedError,
    create,
)
from rfdf.hal import RotatorController

# ---------------------------------------------------------------------------
# GRBL status / settings parsers
# ---------------------------------------------------------------------------


def test_parse_status_report_idle() -> None:
    """A GRBL 1.1 Idle report yields state + machine position."""
    status = parse_status_report("<Idle|MPos:12.500,-3.000,0.000|FS:0,0>")
    assert status.state == "Idle"
    assert status.is_idle is True
    assert status.machine_pos == (12.5, -3.0, 0.0)


def test_parse_status_report_alarm() -> None:
    """An Alarm report is flagged via is_alarm."""
    status = parse_status_report("<Alarm|MPos:0.000,0.000,0.000>")
    assert status.is_alarm is True
    assert status.is_idle is False


def test_parse_status_report_with_work_pos() -> None:
    """A WPos field is parsed when present."""
    status = parse_status_report("<Run|MPos:1.0,2.0,3.0|WPos:0.5,1.5,2.5>")
    assert status.work_pos == (0.5, 1.5, 2.5)


def test_parse_status_report_rejects_garbage() -> None:
    """A response with no <...> report raises a connection error."""
    with pytest.raises(GrblConnectionError, match="no GRBL status report"):
        parse_status_report("ESP32 boot log, no status here")


def test_parse_settings_dump() -> None:
    """A $$ dump parses into {setting_number: value}."""
    dump = "$20=1\n$130=180.000\n$131=90.000\nok\n"
    settings = parse_settings_dump(dump)
    assert settings == {20: 1.0, 130: 180.0, 131: 90.0}


def test_grbl_status_dataclass_defaults() -> None:
    """GrblStatus carries empty work_pos by default."""
    status = GrblStatus(state="Idle", machine_pos=(0.0, 0.0))
    assert status.work_pos == ()


def test_grbl_client_base_url() -> None:
    """The client composes a sane base URL without touching the network."""
    client = GrblHttpClient(host="rotator.local", port=8080)
    assert client.base_url == "http://rotator.local:8080"


# ---------------------------------------------------------------------------
# AntRunner — construction, validation, factory (no network)
# ---------------------------------------------------------------------------


def test_create_requires_host() -> None:
    """The factory rejects a call with no controller host."""
    with pytest.raises(AntRunnerError, match="host is required"):
        create()


def test_capabilities_and_soft_limits() -> None:
    """A constructed AntRunner reports both axes and the cable soft limits."""
    rot = create(host="rotator.local")
    assert rot.supports_azimuth is True
    assert rot.supports_elevation is True
    assert rot.azimuth_range_deg == (-180.0, 180.0)
    assert rot.elevation_range_deg == (0.0, 180.0)
    assert rot.max_speed_deg_per_s == 6.0
    assert rot.positioning_accuracy_deg == 0.1


def test_antrunner_is_structural_rotator_controller() -> None:
    """AntRunnerRotator structurally satisfies the RotatorController Protocol."""
    rot = create(host="rotator.local")
    assert isinstance(rot, RotatorController)


def test_goto_rejects_target_outside_soft_limits() -> None:
    """An out-of-range target is rejected before any network I/O."""
    rot = create(host="rotator.local")

    async def run() -> None:
        with pytest.raises(ValueError, match="azimuth"):
            await rot.goto(500.0, 0.0)
        with pytest.raises(ValueError, match="elevation"):
            await rot.goto(0.0, 270.0)

    asyncio.run(run())


def test_goto_requires_homing_first() -> None:
    """With homing required, an in-range goto raises until calibrate() runs."""
    rot = create(host="rotator.local", homing_required_on_startup=True)

    async def run() -> None:
        with pytest.raises(RotatorNotHomedError, match="homing is required"):
            await rot.goto(45.0, 30.0)

    asyncio.run(run())


def test_constructor_rejects_empty_host() -> None:
    """AntRunnerRotator rejects an empty host directly."""
    with pytest.raises(AntRunnerError, match="requires a controller host"):
        AntRunnerRotator("")

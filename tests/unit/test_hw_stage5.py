"""Unit tests for the Stage 5 hw layer — selftest formatting + rotctld server."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

from rfdf.backends.rotator.mock import MockRotator
from rfdf.hw.rotctld import RotctldServer
from rfdf.hw.selftest import format_human, format_json


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m rfdf.cli.main <args>``."""
    return subprocess.run(
        [sys.executable, "-m", "rfdf.cli.main", *args],
        capture_output=True,
        text=True,
        check=False,
    )


_REPORT = {
    "sdr": {"name": "mock", "ok": True, "latency_ms": 1.2, "error_msg": None},
    "rotator": {"name": "mock", "ok": True, "latency_ms": 0.5, "error_msg": None},
    "geometry": {"name": "static", "ok": True, "latency_ms": 0.3, "error_msg": None},
    "compute": {"name": "local", "ok": False, "latency_ms": 9.0, "error_msg": "boom"},
}


# ---------------------------------------------------------------------------
# selftest formatting
# ---------------------------------------------------------------------------


def test_format_json_round_trips() -> None:
    """format_json emits the report as parseable JSON."""
    assert json.loads(format_json(_REPORT)) == _REPORT


def test_format_human_summarises_health() -> None:
    """format_human renders a per-backend tree + a healthy-count summary."""
    text = format_human(_REPORT, color=False)
    assert "3/4 backends healthy" in text
    assert "FAIL" in text  # the failing compute backend
    assert "error: boom" in text


def test_format_human_all_healthy() -> None:
    """An all-green report ends with an OK summary."""
    report = {k: {**v, "ok": True, "error_msg": None} for k, v in _REPORT.items()}
    text = format_human(report, color=False)
    assert "4/4 backends healthy. OK" in text


# ---------------------------------------------------------------------------
# rotctld protocol
# ---------------------------------------------------------------------------


def _server() -> RotctldServer:
    return RotctldServer(MockRotator(max_speed_deg_per_s=720.0))


def test_rotctld_get_position() -> None:
    """A `p` command returns az + el on two lines."""
    server = _server()
    reply = asyncio.run(server._dispatch("p"))
    assert reply is not None
    az_line, el_line, _ = reply.split("\n")
    assert float(az_line) >= 0.0
    assert float(el_line) >= 0.0


def test_rotctld_set_position() -> None:
    """A `P <az> <el>` command slews and replies RPRT 0."""
    server = _server()
    reply = asyncio.run(server._dispatch("P 30.0 45.0"))
    assert reply == "RPRT 0\n"


def test_rotctld_set_position_out_of_range() -> None:
    """A `P` command outside the rotator range replies an error code."""
    server = _server()
    reply = asyncio.run(server._dispatch("P 999.0 0.0"))
    assert reply == "RPRT -1\n"


def test_rotctld_stop_and_quit() -> None:
    """`S` replies RPRT 0; `q` closes the connection (None)."""
    server = _server()
    assert asyncio.run(server._dispatch("S")) == "RPRT 0\n"
    assert asyncio.run(server._dispatch("q")) is None


def test_rotctld_unknown_command() -> None:
    """An unknown command replies the Hamlib not-implemented errno."""
    server = _server()
    assert asyncio.run(server._dispatch("wat")) == "RPRT -11\n"


# ---------------------------------------------------------------------------
# geometry / rotator CLI sub-groups
# ---------------------------------------------------------------------------


def test_geometry_list_presets_runs() -> None:
    """`rfdf hw geometry list-presets` emits a JSON list against the static backend."""
    result = _run_cli("hw", "geometry", "list-presets")
    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result.stdout), list)


def test_geometry_goto_unknown_preset_errors() -> None:
    """`rfdf hw geometry goto <unknown>` exits non-zero with a clean message."""
    result = _run_cli("hw", "geometry", "goto", "no-such-preset")
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_rotator_status_runs() -> None:
    """`rfdf hw rotator status` reports the mock rotator's position as JSON."""
    result = _run_cli("hw", "rotator", "status")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "azimuth_deg" in report
    assert "elevation_deg" in report

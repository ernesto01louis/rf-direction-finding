"""Unit tests for the rfdf CLI subcommands (Stage 2)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run `python -m rfdf.cli.main <args>` and return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "rfdf.cli.main", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_hw_and_config_subcommands() -> None:
    """`rfdf --help` mentions both new subcommand groups."""
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "hw" in result.stdout
    assert "config" in result.stdout


def test_hw_list_backends_emits_json_catalog() -> None:
    """`rfdf hw list-backends` prints valid JSON enumerating Stage 2 backends."""
    result = _run_cli("hw", "list-backends")
    assert result.returncode == 0, result.stderr
    catalog = json.loads(result.stdout)
    assert set(catalog) == {
        "rfdf.backends.sdr",
        "rfdf.backends.rotator",
        "rfdf.backends.geometry",
        "rfdf.backends.compute",
    }
    assert "mock" in catalog["rfdf.backends.sdr"]
    assert "file-replay" in catalog["rfdf.backends.sdr"]
    assert "mock" in catalog["rfdf.backends.rotator"]
    assert "static" in catalog["rfdf.backends.geometry"]
    assert "mock-morph" in catalog["rfdf.backends.geometry"]
    assert "local" in catalog["rfdf.backends.compute"]


def test_hw_selftest_emits_json_report_and_exits_zero() -> None:
    """`rfdf hw selftest --format json` exercises every backend and exits 0."""
    result = _run_cli("hw", "selftest", "--format", "json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    for section in ("geometry", "rotator", "sdr", "compute"):
        assert section in report
        assert report[section]["ok"] is True, report[section]


def test_hw_selftest_human_format_is_default() -> None:
    """`rfdf hw selftest` defaults to the colour-coded human tree."""
    result = _run_cli("hw", "selftest")
    assert result.returncode == 0, result.stderr
    assert "backends healthy" in result.stdout


def test_config_show_table_format_default() -> None:
    """`rfdf config show` prints a Rich table without crashing."""
    result = _run_cli("config", "show")
    assert result.returncode == 0, result.stderr
    # Rich table writes section headers; check for at least one section name.
    assert "sdr" in result.stdout


def test_config_show_json_format() -> None:
    """`rfdf config show --format=json` emits valid JSON with origin tags."""
    result = _run_cli("config", "show", "--format=json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    for section in ("default", "sdr", "rotator", "geometry", "compute", "eirp"):
        assert section in payload
        assert payload[section]["source"] in {"cli", "env", "toml", "default"}
        assert isinstance(payload[section]["values"], dict)


def test_config_show_invalid_format_errors() -> None:
    """An unknown --format value is rejected."""
    result = _run_cli("config", "show", "--format=yaml")
    assert result.returncode != 0
    assert "unknown format" in result.stderr.lower() or "yaml" in result.stderr.lower()


def test_config_validate_returns_zero_on_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-in defaults validate cleanly."""
    monkeypatch.delenv("RFDF_CONFIG", raising=False)
    result = _run_cli("config", "validate")
    assert result.returncode == 0
    assert "config OK" in result.stdout

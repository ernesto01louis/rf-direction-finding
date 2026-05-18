"""Tests for the ``rfdf orchestrator`` / ``rfdf api`` CLI (Stage 7 R8)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rfdf.cli.main import app

_runner = CliRunner()


def test_orchestrator_degrades_without_extra(monkeypatch) -> None:
    """Without the [orchestrator] extra, subcommands exit 1 with a hint."""
    monkeypatch.setattr("rfdf.orchestrator.is_available", lambda: False)
    result = _runner.invoke(app, ["orchestrator", "status"])
    assert result.exit_code == 1
    assert "pip install rfdf[orchestrator]" in result.output


def test_orchestrator_app_is_wired() -> None:
    result = _runner.invoke(app, ["orchestrator", "--help"])
    assert result.exit_code == 0
    for sub in ("status", "register", "hindsight", "vault", "planner"):
        assert sub in result.output


def test_api_app_is_wired() -> None:
    result = _runner.invoke(app, ["api", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output


def test_orchestrator_status_reports_capabilities() -> None:
    """With the extra installed, `status` prints the declared capabilities
    and exits 0 whether or not an orchestrator is reachable."""
    pytest.importorskip("ai_orchestrator_client")
    result = _runner.invoke(app, ["orchestrator", "status"])
    assert result.exit_code == 0
    assert "rf.doa.run" in result.output
    assert "connection:" in result.output

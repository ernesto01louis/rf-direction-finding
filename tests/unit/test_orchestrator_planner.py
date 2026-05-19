"""Tests for the FlowgraphBridge — planner-dispatched flowgraphs (Stage 7 R5).

``grcc`` is genuinely absent in the build environment, so the
``validate`` no-grcc path is exercised for real; the success path and
``deploy`` use a monkeypatched subprocess.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("ai_orchestrator_client")

from rfdf.orchestrator.planner import (
    Flowgraph,
    FlowgraphBridge,
    _extract_grc,
)


class _FakeClient:
    """Fake orchestrator client for the generate() pipeline."""

    def __init__(self, result: dict) -> None:
        self._result = result
        self.last_request = None

    def run(self, req):
        self.last_request = req
        return SimpleNamespace(run_id="run-123")

    def wait_for_completion(self, run_id, timeout=600.0):
        return SimpleNamespace(run_id=run_id, phase="completed")

    def get_result(self, run_id):
        return self._result


# ── _extract_grc ─────────────────────────────────────────────────────


def test_extract_grc_prefers_grc_file() -> None:
    result = {"files": {"readme.md": "x", "fg.grc": "<flow_graph/>"}}
    assert _extract_grc(result) == "<flow_graph/>"


def test_extract_grc_falls_back_to_first_file() -> None:
    assert _extract_grc({"files": {"main.py": "code"}}) == "code"


def test_extract_grc_falls_back_to_raw_result() -> None:
    assert _extract_grc({"unexpected": 1}) == "{'unexpected': 1}"


# ── generate ─────────────────────────────────────────────────────────


def test_generate_wraps_orchestrate_request() -> None:
    client = _FakeClient({"files": {"ais.grc": "<flow_graph version='1'/>"}})
    bridge = FlowgraphBridge(client)
    fg = bridge.generate("detect AIS at 162 MHz", name="ais")
    assert isinstance(fg, Flowgraph)
    assert fg.name == "ais"
    assert fg.run_id == "run-123"
    assert fg.grc_xml == "<flow_graph version='1'/>"
    # The flowgraph request flows through a real OrchestrateRequest.
    assert "AIS" in client.last_request.prompt
    assert client.last_request.project_name == "rfdf-flowgraph"


# ── validate ─────────────────────────────────────────────────────────


def test_validate_reports_missing_grcc() -> None:
    """grcc is not installed in the build env — the real no-grcc path."""
    bridge = FlowgraphBridge(_FakeClient({}))
    result = bridge.validate(Flowgraph(name="fg", grc_xml="<flow_graph/>"))
    assert result.ok is False
    assert "grcc not found" in result.output


def test_validate_success_with_mocked_grcc(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="compiled ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    bridge = FlowgraphBridge(_FakeClient({}))
    result = bridge.validate(Flowgraph(name="fg", grc_xml="<flow_graph/>"))
    assert result.ok is True
    assert "compiled ok" in result.output


def test_validate_failure_with_mocked_grcc(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="block error")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    bridge = FlowgraphBridge(_FakeClient({}))
    result = bridge.validate(Flowgraph(name="fg", grc_xml="<bad/>"))
    assert result.ok is False
    assert "block error" in result.output


# ── deploy ───────────────────────────────────────────────────────────


def test_deploy_returns_handle_on_success(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    bridge = FlowgraphBridge(_FakeClient({}))
    handle = bridge.deploy(Flowgraph(name="ais", grc_xml="<flow_graph/>"))
    assert handle.host == "rfdf-tools"
    assert handle.path.endswith("ais.grc")


def test_deploy_raises_on_scp_failure(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=255, stdout="", stderr="no route")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    bridge = FlowgraphBridge(_FakeClient({}))
    with pytest.raises(RuntimeError, match="deploy to rfdf-tools failed"):
        bridge.deploy(Flowgraph(name="ais", grc_xml="<flow_graph/>"))

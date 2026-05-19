"""Tests for the rfdf REST API + capability server (Stage 7 R7)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ai_orchestrator_client")

from starlette.testclient import TestClient

from rfdf.api import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "rfdf"


def test_root_banner(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["service"] == "rfdf"
    assert body["healthz"] == "/healthz"


def test_capabilities_discovery(client: TestClient) -> None:
    body = client.get("/capabilities").json()
    assert "rf.doa.run" in body["capabilities"]
    assert body["dispatch"] == "ready"


def test_capability_dispatch_geometry(client: TestClient) -> None:
    resp = client.post(
        "/capabilities/rf.geometry.morph",
        json={"preset": "ula", "num_elements": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["capability"] == "rf.geometry.morph"
    assert body["result"]["num_elements"] == 5


def test_capability_dispatch_doa(client: TestClient) -> None:
    resp = client.post("/capabilities/rf.doa.run", json={"azimuths": [55.0]})
    assert resp.status_code == 200
    assert resp.json()["result"]["num_sources"] >= 1


def test_unknown_capability_is_404(client: TestClient) -> None:
    assert client.post("/capabilities/rf.ghost", json={}).status_code == 404


def test_bad_geometry_preset_is_400(client: TestClient) -> None:
    resp = client.post("/capabilities/rf.geometry.morph", json={"preset": "spiral"})
    assert resp.status_code == 400


def test_token_enforced_on_capability_routes(monkeypatch) -> None:
    monkeypatch.setenv("RFDF_API_TOKEN", "s3kret")
    client = TestClient(create_app())
    # healthz stays open regardless of the token.
    assert client.get("/healthz").status_code == 200
    # capability dispatch without a token is rejected.
    no_token = client.post("/capabilities/rf.geometry.morph", json={"preset": "ula"})
    assert no_token.status_code == 401
    # with the correct bearer token it succeeds.
    ok = client.post(
        "/capabilities/rf.geometry.morph",
        json={"preset": "ula"},
        headers={"Authorization": "Bearer s3kret"},
    )
    assert ok.status_code == 200

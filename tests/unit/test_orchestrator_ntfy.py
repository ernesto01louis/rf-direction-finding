"""Tests for RfdfAlerts — ntfy alerting on three channels (Stage 7 R6)."""

from __future__ import annotations

import pytest

pytest.importorskip("ai_orchestrator_client")

from ai_orchestrator_client import ServiceUnavailable

from rfdf.orchestrator.ntfy import RfdfAlerts


class _OkClient:
    """Fake client capturing every notification."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, int | None, list[str]]] = []

    def send_notification(self, consumer_id: str, notification) -> dict:
        self.sent.append((notification.title, notification.priority, notification.tags))
        return {"status": "sent"}


class _DownClient:
    def send_notification(self, consumer_id: str, notification) -> dict:
        raise ServiceUnavailable("orchestrator unreachable")


def test_channel_default_priorities() -> None:
    client = _OkClient()
    alerts = RfdfAlerts(client)
    alerts.alerts("anomaly", "high-SNR drone")
    alerts.ops("capture started", "868 MHz sweep")
    alerts.research("novel pattern", "unusual OFDM variant")
    priorities = [p for _, p, _ in client.sent]
    assert priorities == [4, 3, 2]  # alerts / ops / research defaults


def test_channel_prefixes_title_and_tags() -> None:
    client = _OkClient()
    RfdfAlerts(client).alerts("drift", "calibration drift", tags=["5.8GHz"])
    title, _priority, tags = client.sent[0]
    assert title == "[rfdf-alerts] drift"
    assert "rfdf-alerts" in tags
    assert "5.8GHz" in tags


def test_explicit_priority_overrides_channel_default() -> None:
    client = _OkClient()
    RfdfAlerts(client).ops("urgent op", "now", priority=5)
    assert client.sent[0][1] == 5


def test_unknown_channel_raises() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        RfdfAlerts(_OkClient()).alert("satellite", "t", "m")


def test_alert_is_fail_tolerant_when_orchestrator_down() -> None:
    out = RfdfAlerts(_DownClient()).alerts("drift", "cal drift")
    assert out["status"] == "failed"
    assert out["channel"] == "alerts"

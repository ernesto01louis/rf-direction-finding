"""Tests for RfdfRecorder — Hindsight memory + L5 vault writes (Stage 7 R4)."""

from __future__ import annotations

from ai_orchestrator_client import ServiceUnavailable

from rfdf.orchestrator.vault import RfdfRecorder


class _OkClient:
    """Fake orchestrator client that records every write."""

    def __init__(self) -> None:
        self.memory: list[str] = []
        self.notes: list[tuple[str, list[str]]] = []

    def write_memory(self, consumer_id: str, content: str) -> dict:
        self.memory.append(content)
        return {"status": "success"}

    def write_vault_note(self, consumer_id: str, note) -> dict:
        self.notes.append((note.title, note.tags))
        return {"status": "success"}


class _DownClient:
    """Fake client where every orchestrator-bound write fails."""

    def write_memory(self, consumer_id: str, content: str) -> dict:
        raise ServiceUnavailable("orchestrator unreachable")

    def write_vault_note(self, consumer_id: str, note) -> dict:
        raise ServiceUnavailable("orchestrator unreachable")


def test_record_detection_writes_memory_and_note() -> None:
    client = _OkClient()
    recorder = RfdfRecorder(client)
    result = recorder.record_detection(
        bearing_deg=47.0,
        frequency_hz=5.8e9,
        modulation="OFDM",
        confidence=0.91,
        bundle_id="bundle-1",
    )
    assert result == {"memory": "ok", "vault": "ok"}
    assert "47.0 deg" in client.memory[0]
    _title, tags = client.notes[0]
    assert "rf-detection" in tags
    assert "OFDM" in tags


def test_record_calibration_note() -> None:
    client = _OkClient()
    out = RfdfRecorder(client).record_calibration(
        name="lab-A", procedure="pilot_tone", geometry_hash="abc123"
    )
    assert out == {"vault": "ok"}
    title, tags = client.notes[0]
    assert title == "calibration-lab-A"
    assert "rf-calibration" in tags


def test_record_geometry_preset_and_model_card_and_campaign() -> None:
    client = _OkClient()
    recorder = RfdfRecorder(client)
    recorder.record_geometry_preset(name="ula8", preset="ula", num_elements=8)
    recorder.record_model_card(
        model_id="resnet1d-v3",
        architecture="resnet1d",
        metrics={"accuracy": 0.94},
    )
    recorder.record_campaign(campaign_id="sweep-2", summary="868 MHz sweep")
    titles = [t for t, _ in client.notes]
    assert titles == ["geometry-ula8", "model-resnet1d-v3", "campaign-sweep-2"]
    tag_sets = [set(tags) for _, tags in client.notes]
    assert "rf-geometry-preset" in tag_sets[0]
    assert "rf-model-card" in tag_sets[1]
    assert "rf-campaign" in tag_sets[2]


def test_writes_are_fail_tolerant_when_orchestrator_down() -> None:
    """A down orchestrator must never raise out of a record_* call."""
    recorder = RfdfRecorder(_DownClient())
    det = recorder.record_detection(bearing_deg=10.0, frequency_hz=1e9)
    assert det["memory"] == "failed"
    assert det["vault"] == "failed"
    cal = recorder.record_calibration(name="x", procedure="pilot_tone", geometry_hash="h")
    assert cal["vault"] == "failed"


def test_record_campaign_does_not_raise_when_orchestrator_down() -> None:
    """No exception escapes even when the orchestrator is unreachable."""
    recorder = RfdfRecorder(_DownClient())
    # Reaching the assertion proves no exception propagated.
    result = recorder.record_campaign(campaign_id="c", summary="s")
    assert result["vault"] == "failed"

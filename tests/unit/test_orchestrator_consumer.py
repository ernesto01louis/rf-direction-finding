"""Tests for RfdfConsumer capability adapters (Stage 7 R2).

The adapters are exercised through ``Consumer.dispatch`` against the
mock SDR — no hardware, the platform's load-bearing principle.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("ai_orchestrator_client")

from rfdf.orchestrator import RfdfConsumer
from rfdf.orchestrator.consumer import DATA_PLANE_CAPABILITIES


def _dispatch(consumer: RfdfConsumer, capability: str, payload: dict) -> dict:
    return asyncio.run(consumer.dispatch(capability, payload))


def test_consumer_declares_expected_capabilities() -> None:
    assert RfdfConsumer().capabilities == [
        "rf.classify",
        "rf.doa.run",
        "rf.geometry.morph",
    ]


def test_run_doa_recovers_emitter_bearings() -> None:
    out = _dispatch(
        RfdfConsumer(),
        "rf.doa.run",
        {"algorithm": "music", "azimuths": [30.0, 75.0], "channels": 8, "snr_db": 25.0},
    )
    assert out["algorithm"] == "music"
    assert out["num_sources"] == 2
    bearings = sorted(out["azimuth_deg"])
    assert bearings[0] == pytest.approx(30.0, abs=2.0)
    assert bearings[1] == pytest.approx(75.0, abs=2.0)


def test_run_doa_default_payload() -> None:
    out = _dispatch(RfdfConsumer(), "rf.doa.run", {})
    assert out["num_sources"] >= 1
    assert out["azimuth_deg"][0] == pytest.approx(30.0, abs=3.0)


def test_geometry_morph_ula() -> None:
    out = _dispatch(
        RfdfConsumer(),
        "rf.geometry.morph",
        {"preset": "ula", "num_elements": 6, "freq_hz": 2.4e9},
    )
    assert out["preset"] == "ula"
    assert out["num_elements"] == 6
    assert len(out["positions_m"]) == 6


def test_geometry_morph_planar_cross() -> None:
    out = _dispatch(RfdfConsumer(), "rf.geometry.morph", {"preset": "planar_cross"})
    assert out["preset"] == "planar_cross"
    assert out["num_elements"] == 5


def test_geometry_morph_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown geometry preset"):
        _dispatch(RfdfConsumer(), "rf.geometry.morph", {"preset": "spiral"})


def test_to_registration_merges_data_plane_capabilities() -> None:
    reg = RfdfConsumer().to_registration(
        "http://rfdf.lan:8000",
        callback_token="tok",
        extra_capabilities=DATA_PLANE_CAPABILITIES,
    )
    assert reg.consumer_id == "rfdf"
    assert "rf.doa.run" in reg.capabilities
    assert "memory.write" in reg.capabilities
    assert "evidence.push" in reg.capabilities

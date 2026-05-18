"""Tests for rfdf evidence bundles (Stage 7 R3)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("ai_orchestrator_client")

from rfdf.orchestrator import evidence


@pytest.fixture(autouse=True)
def _isolate_bundle_store(monkeypatch, tmp_path):
    """Redirect the local bundle store to a temp dir."""
    monkeypatch.setattr(evidence, "bundles_dir", lambda: tmp_path)


def _doa_bundle() -> evidence.RfdfEvidenceBundle:
    return evidence.build_bundle(
        "doa",
        inputs={"algorithm": "music", "freq_hz": 2.4e9, "channels": 8},
        process={"algorithm": "music", "num_signals": 2},
        outputs={"azimuth_deg": [30.0, 75.0]},
    )


def test_build_bundle_is_citation_grade_by_default() -> None:
    bundle = _doa_bundle()
    assert bundle.kind == "doa"
    assert bundle.quality == "citation-grade"
    assert bundle.schema_version == "rfdf-evidence/1.0"
    assert bundle.reproducibility_hash
    assert "git_sha" in bundle.provenance
    assert "platform_version" in bundle.provenance
    assert "host" in bundle.provenance


def test_reproducibility_hash_is_deterministic() -> None:
    inputs = {"algorithm": "music", "freq_hz": 2.4e9}
    a = evidence.build_bundle("doa", inputs=inputs, process={}, outputs={})
    b = evidence.build_bundle("doa", inputs=inputs, process={}, outputs={})
    # Same inputs + same code SHA → same reproducibility hash.
    assert a.reproducibility_hash == b.reproducibility_hash
    # Distinct runs still get distinct bundle ids.
    assert a.bundle_id != b.bundle_id


def test_reproducibility_hash_changes_with_inputs() -> None:
    a = evidence.build_bundle("doa", inputs={"f": 1}, process={}, outputs={})
    b = evidence.build_bundle("doa", inputs={"f": 2}, process={}, outputs={})
    assert a.reproducibility_hash != b.reproducibility_hash


def test_save_local_writes_bundle_json(tmp_path) -> None:
    bundle = _doa_bundle()
    path = evidence.save_local(bundle)
    assert path.is_file()
    on_disk = json.loads(path.read_text())
    assert on_disk["bundle_id"] == bundle.bundle_id
    assert on_disk["quality"] == "citation-grade"


def test_to_evidence_push_bridges_to_sdk_payload() -> None:
    bundle = _doa_bundle()
    push = evidence.to_evidence_push(bundle)
    assert push.bundle_id == bundle.bundle_id
    assert push.bundle["kind"] == "doa"
    assert push.bundle["reproducibility_hash"] == bundle.reproducibility_hash


def test_push_bundle_success() -> None:
    bundle = _doa_bundle()

    class _OkClient:
        def push_evidence(self, consumer_id, push):
            return {"status": "stored", "bundle_id": push.bundle_id}

    status = evidence.push_bundle(bundle, _OkClient())
    assert status["pushed"] is True
    assert status["quality"] == "citation-grade"


def test_push_bundle_marks_degraded_on_failure() -> None:
    from ai_orchestrator_client import ServiceUnavailable

    bundle = _doa_bundle()
    assert bundle.quality == "citation-grade"

    class _DownClient:
        def push_evidence(self, consumer_id, push):
            raise ServiceUnavailable("orchestrator unreachable")

    status = evidence.push_bundle(bundle, _DownClient())
    assert status["pushed"] is False
    assert status["quality"] == "degraded"
    # The bundle object and its persisted copy are both downgraded.
    assert bundle.quality == "degraded"

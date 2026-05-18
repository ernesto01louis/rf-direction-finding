"""rfdf evidence bundles (Stage 7 R3).

Every DOA run, classification, capture, or training run can produce a
citation-grade **evidence bundle**: a self-contained record of inputs,
process, outputs, and provenance with a reproducibility hash.

The bundle uses rfdf's **own** schema (:class:`RfdfEvidenceBundle`),
decoupled from the orchestrator's. :func:`to_evidence_push` is the
bridge that converts it to the SDK's ``EvidencePush`` payload — keeping
both schemas independently evolvable.

``quality`` is **always set explicitly**:

* ``citation-grade`` — built standalone with complete local provenance
  (git SHA + inputs + platform version). A standalone rfdf run has no
  LLM trace and that is correct, not degraded — the provenance is the
  git SHA and the reproducibility hash.
* ``degraded`` — an orchestrator-bound write was attempted mid-run and
  the orchestrator was unreachable, so orchestrator-side enrichment
  (LlmCall records from a dispatched run) is missing.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from platformdirs import user_data_dir
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ai_orchestrator_client import EvidencePush, OrchestratorClient

BundleKind = Literal["doa", "classification", "capture", "training"]
BundleQuality = Literal["citation-grade", "degraded"]

_SCHEMA_VERSION = "rfdf-evidence/1.0"


def _git_sha() -> str:
    """Return the current commit SHA, or 'unknown' outside a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _platform_version() -> str:
    try:
        from rfdf import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


class RfdfEvidenceBundle(BaseModel):
    """A citation-grade evidence bundle in rfdf's own schema.

    Persisted locally and optionally pushed to the orchestrator. The
    schema is intentionally independent of the orchestrator's
    ``EvidenceBundle`` — the bridge (:func:`to_evidence_push`) converts.
    """

    model_config = ConfigDict(frozen=False)

    schema_version: str = _SCHEMA_VERSION
    bundle_id: str
    kind: BundleKind
    quality: BundleQuality
    created_at: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    process: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    reproducibility_hash: str


def _reproducibility_hash(inputs: dict[str, Any], git_sha: str) -> str:
    """SHA-256 of the canonicalised inputs plus the code SHA."""
    canonical = json.dumps({"inputs": inputs, "code_sha": git_sha}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_bundle(
    kind: BundleKind,
    *,
    inputs: dict[str, Any],
    process: dict[str, Any],
    outputs: dict[str, Any],
    quality: BundleQuality = "citation-grade",
    operator: str | None = None,
) -> RfdfEvidenceBundle:
    """Assemble an :class:`RfdfEvidenceBundle` with full local provenance.

    ``quality`` defaults to ``citation-grade`` — a standalone run with a
    git SHA and a reproducibility hash IS citation-grade. Pass
    ``degraded`` only when orchestrator-side enrichment was expected and
    lost (see :func:`push_bundle`).
    """
    git_sha = _git_sha()
    bundle_id = uuid.uuid4().hex
    provenance = {
        "git_sha": git_sha,
        "platform_version": _platform_version(),
        "host": socket.gethostname(),
        "operator": operator,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return RfdfEvidenceBundle(
        bundle_id=bundle_id,
        kind=kind,
        quality=quality,
        created_at=provenance["timestamp"],
        inputs=inputs,
        process=process,
        outputs=outputs,
        provenance=provenance,
        reproducibility_hash=_reproducibility_hash(inputs, git_sha),
    )


def bundles_dir() -> Path:
    """Local bundle store: ``~/.local/share/rfdf/bundles/``."""
    path = Path(user_data_dir("rfdf")) / "bundles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_local(bundle: RfdfEvidenceBundle) -> Path:
    """Persist a bundle under the local store; return the written path."""
    dest = bundles_dir() / bundle.bundle_id
    dest.mkdir(parents=True, exist_ok=True)
    bundle_path = dest / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return bundle_path


def to_evidence_push(bundle: RfdfEvidenceBundle) -> EvidencePush:
    """Bridge: convert an rfdf bundle to the SDK's ``EvidencePush`` payload.

    The orchestrator stores the bundle verbatim in rfdf's own schema —
    the conversion is a wrapper, not a schema translation, so both
    sides stay independently evolvable.
    """
    from ai_orchestrator_client import EvidencePush

    return EvidencePush(bundle_id=bundle.bundle_id, bundle=bundle.model_dump())


def push_bundle(
    bundle: RfdfEvidenceBundle,
    client: OrchestratorClient,
    *,
    consumer_id: str = "rfdf",
) -> dict[str, Any]:
    """Push a bundle to the orchestrator, marking it degraded on failure.

    Never raises — a standalone platform must not break when the
    orchestrator is unreachable. On any connectivity / API error the
    bundle's ``quality`` is downgraded to ``degraded``, the local copy
    is rewritten, and a status dict is returned.
    """
    from ai_orchestrator_client import OrchestratorError

    try:
        result = client.push_evidence(consumer_id, to_evidence_push(bundle))
        return {"pushed": True, "quality": bundle.quality, "result": result}
    except (OrchestratorError, OSError) as exc:
        bundle.quality = "degraded"
        save_local(bundle)
        return {"pushed": False, "quality": "degraded", "error": str(exc)}

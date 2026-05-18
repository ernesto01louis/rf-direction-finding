# Evidence bundles

Every DOA run, classification, capture, or training run can produce a
citation-grade **evidence bundle** — a self-contained record of inputs,
process, outputs, and provenance with a reproducibility hash.

## rfdf's own schema

`rfdf.orchestrator.RfdfEvidenceBundle` is rfdf's **own** schema,
deliberately decoupled from the orchestrator's `EvidenceBundle`. The
two evolve independently; `to_evidence_push()` is the bridge that wraps
an rfdf bundle into the SDK's `EvidencePush` payload. The orchestrator
stores the bundle verbatim — it never parses rfdf's schema.

```python
from rfdf.orchestrator import build_bundle
from rfdf.orchestrator.evidence import save_local, push_bundle

bundle = build_bundle(
    "doa",
    inputs={"freq_hz": 2.4e9, "algorithm": "music"},
    process={"algorithm": "music", "num_signals": 2},
    outputs={"azimuth_deg": [35.0, 80.0]},
)
save_local(bundle)                       # ~/.local/share/rfdf/bundles/<id>/
push_bundle(bundle, client)              # optional — to the orchestrator
```

## The quality flag

`RfdfEvidenceBundle.quality` is **always set explicitly**, one of:

- **`citation-grade`** — built standalone with complete local
  provenance (git SHA + canonicalised inputs + platform version + a
  SHA-256 reproducibility hash). A standalone rfdf run has no LLM
  trace, and that is *correct, not degraded* — the provenance is the
  git SHA and the hash.
- **`degraded`** — an orchestrator-bound write was attempted mid-run
  and the orchestrator was unreachable, so orchestrator-side
  enrichment (LlmCall records from a dispatched run) is missing.

`push_bundle()` never raises: if the orchestrator is unreachable it
downgrades the bundle to `degraded`, rewrites the local copy, and
returns a status dict. The orchestrator's `evidence verify` warns on
degraded bundles.

## Reproducibility hash

`reproducibility_hash` is `SHA-256(canonical(inputs) + code_sha)`. Two
runs with identical inputs on the same commit produce the same hash —
the bundle ids still differ (one per run).

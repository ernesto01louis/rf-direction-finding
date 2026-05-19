# Audit-pass — orchestrator audit findings vs rf-direction-finding

The `ai-orchestrator` audit (see `00-CONTEXT-project-brief.md` §8) catalogued
patterns to avoid. This document checks each finding against the
rf-direction-finding codebase as a final Stage 7 / v0.1.0 gate.

| # | Orchestrator audit finding | rfdf rule | Status | Evidence |
|---|---|---|---|---|
| 1 | `VISION.md` claimed in README but missing | Write `VISION.md` in Stage 1, before any code | ✅ | `VISION.md` at repo root, present since Stage 1 |
| 2 | README status stale (claimed Phase 0 when Phase 3 done) | CI asserts the README status line references the highest git tag | ✅ | `.github/workflows/ci.yml` job `readme-status-truthful`; README `## Status` |
| 3 | HITL config in 2.5 places | Every config concern lives in exactly one canonical place | ✅ | `src/rfdf/config.py` + `[tool.*]` in `pyproject.toml`; no duplication |
| 4 | Inline schemas in `app.py` mirror on-disk JSON, silently drift | Schemas loaded from disk / declared once; no mirrored copies | ✅ | rfdf has no mirrored schemas; HAL Protocols are the single contract (`src/rfdf/hal/`) |
| 5 | `blender-mcp` in base requirements (domain-flavoured) | Domain-flavoured deps in optional extras only | ✅ | `pyproject.toml` — base deps carry no RF/ML/domain libs; CI `zero-domain-deps` enforces it |
| 6 | Branch protection deferred | Branch protection ON from Stage 1 | ✅ | Activated at Stage 5 per the operator (see `CLAUDE.md` caveats); on for the v0.1.0 release |
| 7 | Coverage report deferred | Coverage from day 1, badge in README | ✅ | `README.md` Codecov badge; CI `coverage` job, `fail_under = 80` in `pyproject.toml` |
| 8 | SOPS decision carried in both states | Decision made + documented in Stage 1 | ✅ | `SECURITY.md` records the secrets-handling decision |
| 9 | Sudo allowlist not threat-modeled | `SECURITY.md` from Stage 1 with an explicit threat model | ✅ | `SECURITY.md` present since Stage 1 |
| 10 | PyPI publish blocked external consumers | rfdf's PyPI publish is part of the v0.1.0 release gate | ✅ | `pyproject.toml` `version = "0.1.0"`; `.github/workflows/release.yml`; operator registers the Trusted Publisher, tags `v0.1.0` |
| 11 | `api/routes.py` grew to 2000 LoC | Split routes by capability domain from day 1 | ✅ | `src/rfdf/api/` — `app.py` is ~120 LoC; routes are health / discovery / capability dispatch, kept small |

## Stage 7 specifics

- **Optionality.** `import rfdf.orchestrator` never raises without the
  `[orchestrator]` extra; `rfdf.orchestrator.is_available()` gates every
  orchestrator code path; `ai-orchestrator-client` is imported only inside
  `rfdf/orchestrator/_real.py` and its siblings. CI job `test-orchestrator`
  verifies graceful degradation. Audit finding #5 applied to the orchestrator
  dependency itself.
- **No schema coupling (#4).** rfdf produces evidence bundles in its **own**
  schema (`RfdfEvidenceBundle`); `to_evidence_push` converts at the bridge
  layer. The two schemas evolve independently — no mirrored definition.
- **Degraded-evidence honesty.** `RfdfEvidenceBundle.quality` is always set
  explicitly. A bundle is `degraded` only when an orchestrator-bound write was
  attempted mid-run and the orchestrator was unreachable — documented in
  `docs/orchestrator/evidence-bundles.md` and `CLAUDE.md`.
- **License compliance.** The platform is Apache-2.0. `[orchestrator]` adds
  only `ai-orchestrator-client` (Apache-2.0); `[api]` adds FastAPI / uvicorn /
  websockets (MIT / BSD). No GPL enters the dependency tree — GNU Radio's
  `grcc` is invoked only as a subprocess by `FlowgraphBridge`, never imported.

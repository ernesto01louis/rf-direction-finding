# Stage 7 outputs — orchestrator integration + REST API (v0.1.0 GA)

**Date:** 2026-05-19 · **Result:** `rfdf v0.1.0` shipped to PyPI — the build is
complete. This file is the handoff: what landed, what diverged from the Stage 7
brief, and what a future session needs to know to continue or fix.

---

## 1. What shipped

Stage 7 turned out to be a **three-repo** effort, because the Stage 7 brief
assumed an SDK surface that did not exist. See §3.

| Repo | Change | Tag / release |
|---|---|---|
| `ai-orchestrator` | Phase 3.6 — generic external consumer registry | `v0.3.6-phase3.6`, PR #45, **deployed live** (192.168.2.218) |
| `ai-orchestrator-client` | Consumer SDK surface (`Consumer`, `@capability`, push clients) | `v0.1.0` + `v0.1.1` on PyPI, PRs #2 #3 |
| `rf-direction-finding` | Stage 7 — orchestrator integration + REST API | `v0.1.0` on PyPI, PRs #48 #49 #50 |

### `ai-orchestrator` — Phase 3.6 (consumer registry)
- `api/routes/consumers.py` — registry (`POST /consumers/register`, `GET
  /consumers`, `GET/DELETE /consumers/{id}`, `POST /consumers/{id}/heartbeat`),
  capability dispatch (`POST /capabilities/{capability}/invoke` — outbound
  proxy), data-plane push (`POST /consumers/{id}/{memory,vault,notify,evidence}`).
- `memory/consumers.json` registry (JSON-canonical, file-locked).
- `core/consumer_health.py` — health daemon, ships dormant
  (`consumers.health_poll_seconds=0`).
- 3 Prometheus counters; bearer-auth gated; CLAUDE.md + RUNBOOK + ROADMAP 3.6.

### `ai-orchestrator-client` — consumer SDK
- `Consumer` base + `@capability` decorator (framework-free; `dispatch()` is
  the entry point a consumer's web layer calls).
- `Hindsight` / `Vault` / `Ntfy` thin push clients.
- Client methods: `register_consumer`, `list_consumers`, `invoke_capability`,
  `write_memory`, `write_vault_note`, `send_notification`, `push_evidence`.
- Models: `ConsumerRegistration`, `ConsumerRecord`, `EvidencePush`, etc.

### `rf-direction-finding` — Stage 7
- `src/rfdf/orchestrator/` — lazy-import wrapper (`import rfdf.orchestrator`
  never raises; `ai-orchestrator-client` imported only in `_real.py` + siblings):
  `RfdfConsumer` (capability adapters), `RfdfEvidenceBundle`/`build_bundle`
  (own schema + `quality` flag + `to_evidence_push` bridge), `RfdfRecorder`
  (Hindsight + L5 vault), `FlowgraphBridge`, `RfdfAlerts`.
- `src/rfdf/api/` — FastAPI app behind the `[api]` extra: `GET /healthz`,
  `POST /capabilities/{capability}` (the orchestrator's callback target).
  `rfdf api serve`; `create_app()` factory.
- `rfdf orchestrator {status,register,hindsight,vault,planner}` CLI.
- `examples/06-standalone-vs-orchestrator/`, `docs/orchestrator/` (7 pages),
  `docs/standalone-vs-orchestrator.md`, `docs/audit-pass.md`.
- ~54 tests; Stage 7 code coverage 85.7%.

---

## 2. Decisions that held

- **Standalone-first is real.** Every orchestrator path is gated on
  `rfdf.orchestrator.is_available()`. `pip install rfdf` is a complete platform;
  `[orchestrator]` is a bonus. Verified by the clean-venv smoke test.
- **Bidirectional capability RPC** (the operator's chosen depth): the
  orchestrator can dispatch `rf.doa.run` etc. back into rfdf's REST API.
- **rfdf and `ai-orchestrator-client` are independent PyPI packages.** rfdf is
  not bundled "inside" the SDK — bundling would force every SDK user to carry
  the RF platform and break the platform-not-hub principle.

---

## 3. What diverged from the plan / brief — read this first

The Stage 7 brief (`STAGE-7-orchestrator-integration.pdf`) was **substantially
aspirational**. Its code sketches assumed SDK classes that did not exist. The
divergences below are deliberate and documented.

| Brief / plan said | What we did | Why |
|---|---|---|
| SDK has `Consumer`, `@capability`, `Hindsight`/`Vault`/`Ntfy`/`Planner`, `EvidenceBundle` | None existed. We **built** the consumer SDK surface in `ai-orchestrator-client` (the cross-repo expansion) | The brief was written before the SDK; the real SDK was campaign-only |
| Orchestrator has external write APIs for memory/vault/notify + a consumer-registration endpoint | None existed. We **built** Phase 3.6 in `ai-orchestrator` | Same — brief assumed a surface that wasn't there |
| Plan: alembic `0003` + Postgres `consumers` table + `mirror_consumers` | **JSON-only** (`consumers.json`); no Postgres mirror | The map is small, read on demand, no aggregate-query need. Revisit only if a consumers dashboard lands |
| Brief: 8 docs under `docs/orchestrator/` | 7 pages under `docs/orchestrator/` + `docs/standalone-vs-orchestrator.md` + `docs/audit-pass.md` | Same coverage, less padding |
| (not in plan) | SDK **v0.1.1** patch — `@capability` made signature-preserving (TypeVar) | v0.1.0's decorator returned `Callable[...,Any]`, tripping `mypy --strict`'s `untyped-decorator` in rfdf |
| (not in plan) | `release.yml` `publish-pypi` job had a Stage-1 `if: false` guard; missed in R9, fixed in PR #49 by re-pointing the `v0.1.0` tag | Caught only when the first `v0.1.0` release skipped the PyPI upload |
| Stage 6 handoff implied Stage 7 wires `src/rfdf/api/`; Stage 7 brief did not scope it | We **did** wire the REST API — it doubles as the capability callback target and un-inerts the Stage 6 `api.rf.lan` / `rfdf-api` infra | Bidirectional RPC needs the server anyway |

---

## 4. Known issues + follow-ups

1. **pre-commit mypy hook pins the SDK by git URL.** `.pre-commit-config.yaml`
   `additional_dependencies` carries
   `ai-orchestrator-client @ git+https://github.com/ernesto01louis/ai-orchestrator-client@main`
   — added while `0.1.1` was unpublished. **`0.1.1` is now on PyPI**, so this
   should be switched to `ai-orchestrator-client>=0.1.1`. Low-risk one-line
   follow-up; the comment in that file marks the upgrade path.
2. **`test-demo-no-hardware` is intermittently flaky.** Failure signature:
   `ValueError: Passband ripple was unable to meet ripple specs` from
   `torchsig/transforms/functional.py` — a random filter-generation flake in
   the TorchSig augmentation, not an rfdf bug. Re-run the job; it passes. A
   proper de-flake (seed the augmentation) is a separate task.
3. **`FlowgraphBridge.validate()` / `deploy()` are not exercised in CI.** `grcc`
   (GNU Radio) is not installed in the build env — the no-`grcc` path is unit-
   tested; the live compile/deploy path is operator-verified only.
4. **`rf.classify` capability** needs the `[ml]` extra **and** a model in the
   registry. The adapter raises a clear error otherwise; it is not covered by
   the mock-only test suite.
5. **`ai-orchestrator` consumer-health daemon ships dormant**
   (`consumers.health_poll_seconds=0`). Set a positive interval + restart to
   enable `/healthz` polling of registered consumers.
6. **mypy hook reach.** `src/rfdf/orchestrator/consumer.py` imports
   `rfdf.ml.inference` via `importlib` (dynamic) on purpose — so mypy does not
   follow the orchestrator subtree into the heavy `[ml]` stack. Keep it dynamic.

---

## 5. Where things live

- **rfdf orchestrator integration:** `src/rfdf/orchestrator/` — `__init__.py`
  (lazy `__getattr__`), `availability.py`, `_real.py` (the only module that
  imports `ai_orchestrator_client` at module scope, plus its siblings
  `consumer`/`evidence`/`vault`/`planner`/`ntfy`).
- **rfdf REST API:** `src/rfdf/api/` — `create_app()` factory; `app.py`.
- **rfdf CLI:** `src/rfdf/cli/orchestrator.py`, `src/rfdf/cli/api_cmd.py`.
- **Orchestrator Phase 3.6:** `ai-orchestrator` repo —
  `api/routes/consumers.py`, `core/consumer_health.py`, `memory_pkg`
  (`load_consumers`/`save_consumers`/`vault_write_consumer_note`).
- **SDK consumer surface:** `ai-orchestrator-client` repo —
  `ai_orchestrator_client/consumer.py`, `models/consumers.py`.
- **Docs:** `docs/orchestrator/` (7 pages), `docs/standalone-vs-orchestrator.md`,
  `docs/audit-pass.md` (orchestrator-audit findings checked off).

---

## 6. Verification done

- Clean-venv smoke test: `pip install rfdf` → standalone works, accessing
  `RfdfConsumer` raises `OrchestratorNotAvailableError`; `pip install
  rfdf[orchestrator]` → `is_available()` True, `RfdfConsumer` resolves, pulls
  `ai-orchestrator-client 0.1.1`.
- `ai-orchestrator` Phase 3.6 deployed: `GET /consumers` returns 200 on the
  live orchestrator.
- All rfdf CI green at merge (the one `readme-status-truthful` red was the
  expected pre-tag chicken-and-egg).

## 7. Not done (by design / out of scope)

- Real-hardware capability runs (`grcc` deploy, B210 capture) — operator-
  verified, not in this build env.
- Phased-array integration (Open.Space Mini) — explicitly out of scope per
  the project brief; the HAL leaves a clean attach point.

---

*Next: the platform is `v0.1.0` stable. Remaining work is research + ecosystem,
not platform-building — see `ROADMAP.md` "After Stage 7".*

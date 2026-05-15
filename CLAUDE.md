# CLAUDE.md

> Auto-loaded by Claude Code at session start. Keep under 200 lines. Update whenever
> reality diverges. A stale CLAUDE.md misleads every session until it's fixed.

## What this project is — and isn't

`rfdf` is a **hardware-agnostic RF research platform** for direction finding, signal
classification, and phased-array experimentation. It runs **standalone** (default) and
**also** as an `ai-orchestrator` consumer when `[orchestrator]` extra is installed.

It is **NOT** a hub for specific RF projects (drone-detection, antenna manufacturing,
domain-specific fingerprinting). Those are downstream consumers in their own repos.

**Test for "does this change belong here?":**
- Would a user with a HackRF / Pluto / file-replay also benefit? → yes: belongs here.
- Does it reference drone-detection, EME, jamming-research, or any specific signal
  application? → no: belongs in a downstream consumer.

License: **Apache-2.0** (`LICENSE` at repo root).

## Where to find things

| Doc | What's in it |
|---|---|
| [VISION.md](VISION.md) | The principles + when-in-doubt questions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layered design + HAL + config precedence |
| [ROADMAP.md](ROADMAP.md) | Stages 1–7 with status |
| [SECURITY.md](SECURITY.md) | Threat model + secret handling |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup + commit conventions |

## Coding conventions

- **Python 3.11+**, type hints on all new code, `mypy --strict` clean on `src/rfdf/`.
- **Docstrings: Google style.** `[D]` lint rules enabled via ruff with
  `pydocstyle.convention = "google"`.
- **Line length 100.** ruff format enforces.
- **FastAPI handlers sync** where they call into sync helpers (most do); async only when
  there's a real win (streaming, WebSocket fan-out).
- **No `import uhd` / `import torch` / `import runpod` outside `src/rfdf/backends/`.**
  Base install must work with **zero RF or ML dependencies**.

## Testing conventions

- pytest, markers: `hardware`, `slow`, `integration`, `gpu`. All four skipped by default.
- Property-based tests for HAL contracts via Hypothesis.
- Tests live under `tests/`. Fixtures in `tests/conftest.py`.
- `tests/demo_no_hardware/` is the CI gate: full pipeline must run on synthetic data
  with CRLB-bounded accuracy from Stage 3 onward.

## Commit + branch conventions

- **Conventional Commits enforced** via commitlint in CI + commitizen pre-commit hook.
- Allowed types: `feat, fix, docs, style, refactor, perf, test, chore, build, ci, revert`.
- Allowed scopes (from 00-CONTEXT §10): `hal, dsp, ml, backends, capture, cli, api,
  orchestrator, config, docs, ci, tests, tools, examples, release, deps`.
- Branch from `main`. Branch name: `<type>/<short-slug>` (e.g. `feat/hal`,
  `fix/b210-clock-lock`).
- One approval + green CI before merge. Squash-merge is the default.
- **Branch protection is currently DEFERRED** — see `docs/operational-decisions.md`.
  Revisit before Stage 5.

## When you (Claude) work on this repo

1. Read this file, then VISION.md, then ARCHITECTURE.md before any non-trivial change.
2. **Propose first, execute later** for any multi-file work. Use a plan file.
3. Update CHANGELOG.md under `[Unreleased]` for any user-visible change.
4. Update ROADMAP.md status when finishing a stage.
5. **Never modify the HAL Protocol classes outside Stage 2.** If a later stage seems to
   need a new method, stop and fix Stage 2 properly — don't paper over with shims.

## The hardware-agnostic rule

This is the load-bearing architectural commitment. If `pip install rfdf` (no extras)
ever fails because something tries to import `uhd` or `torch` at module load time, the
principle is broken.

CI enforces this via the `zero-domain-deps` job: installs the base package, asserts no
domain libraries appear in `sys.modules` after `import rfdf`.

## Caveats

*(Populated as discovered. Add gotchas here so future sessions don't relearn them.)*

- **Stage 1 (current):** Branch protection on `main` is deferred — solo operator chose
  faster iteration during scaffold/algo work. Revisit before Stage 5.
- **B210 fractional-N PLL retune phase randomization** (Stage 5): coherent operation
  *requires* pilot-tone calibration after every retune. The mock SDR (Stage 2) models
  this via `mock_b210_behavior=True` so calibration code can be developed against it.
- **Prefect-fallback-style degraded evidence bundles** (Stage 7): when the orchestrator
  is unreachable mid-run, bundles are marked `quality: degraded`. Verifiers should
  warn but not fail on the flag.

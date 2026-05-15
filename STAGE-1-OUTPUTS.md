# STAGE-1-OUTPUTS

> Real shipped state from Stage 1, intended for Stage 2 (and beyond) to read at
> session start. **Every subsequent stage MUST ship a `STAGE-N-OUTPUTS.md` of
> the same shape** — see [§7 "Convention for future stages"](#7-convention-for-future-stages).

## 1. Identifiers

| | |
|---|---|
| Stage | 1 — Repository scaffold and conventions |
| Tag | [`v0.0.1`](https://github.com/ernesto01louis/rf-direction-finding/releases/tag/v0.0.1) |
| Squash-merge commit | `ef8a4a1` |
| Initial commit (LICENSE only, auto-created) | `5b525b6` |
| PR | [#1 `feat(scaffold): Stage 1 — initial repository scaffold and conventions`](https://github.com/ernesto01louis/rf-direction-finding/pull/1) |
| Tagged | 2026-05-15 |
| Handoff source | `STAGE-1-scaffold-and-conventions.pdf` in the operator's handoff bundle |

## 2. CI gate inventory (10 jobs green at tag)

These are the gates Stage 2+ must keep green. **Adding a stage MUST NOT drop a gate**;
new gates may be added.

| Job | Purpose |
|---|---|
| `lint` | ruff check + ruff format + mypy strict on `src/rfdf/` |
| `test-base (py3.11)` | Base + dev extras install + `pytest tests/unit` |
| `test-base (py3.12)` | Same on Python 3.12 |
| `test-demo-no-hardware` | Load-bearing principle (Stage 2+ extends to CRLB-bounded tests) |
| `test-orchestrator (lazy-import)` | `import rfdf` cleanly without `ai-orchestrator-client` |
| `zero-domain-deps` | **Audit-lesson check:** base install must not pull in `uhd / torch / runpod / fastapi / ai_orchestrator_client / …` |
| `coverage` | Tokenless Codecov upload |
| `readme-status-truthful` | **Audit-lesson check:** README "Status" line must reference latest git tag |
| `conventional-commits` | `wagoid/commitlint-github-action@v6` on PR title + commits |
| `pre-commit` | Defense in depth (full hook set) |

## 3. Decisions taken (with cross-reference)

| Decision | Choice | Recorded in |
|---|---|---|
| Branch protection on `main` | **Deferred** until before Stage 5 | [`docs/operational-decisions.md`](docs/operational-decisions.md) |
| Secret-at-rest encryption (SOPS / age) | **Deferred** until non-solo contributor | [`docs/operational-decisions.md`](docs/operational-decisions.md), [`SECURITY.md`](SECURITY.md) §2 |
| PyPI publish workflow | **Scaffolded, inactive** (`release.yml:publish-pypi` has `if: false`) until v0.1.0 | [`docs/operational-decisions.md`](docs/operational-decisions.md), [`.github/workflows/release.yml`](.github/workflows/release.yml) |
| Python target | **3.11+** (CI runs 3.11 + 3.12) | [`pyproject.toml`](pyproject.toml) |
| Build backend | **Hatchling** | [`pyproject.toml`](pyproject.toml) |
| Docstring style | **Google** | [`pyproject.toml`](pyproject.toml) `[tool.ruff.lint.pydocstyle]` |
| Codecov account | Same account as `ai-orchestrator` (tokenless GitHub App for public OSS) | [`codecov.yml`](codecov.yml) |
| Pre-commit hook versions | **Pinned** | [`.pre-commit-config.yaml`](.pre-commit-config.yaml) |
| Dependabot scope | **Security advisories only** at Stage 1 | [`.github/dependabot.yml`](.github/dependabot.yml) |
| Tag style | Semver (`v0.0.1`, `v0.0.2`, …, `v0.1.0`) — **deviates** from aero's `v0.x-stageN-partial` style by design | [`ROADMAP.md`](ROADMAP.md) |

## 4. Deviations from the Stage 1 PDF

**None of substance.** Notes on minor implementation choices:

- The Stage 1 PDF lists CI jobs `lint / test-base / test-demo-no-hardware / test-orchestrator / coverage / readme-status-truthful / conventional-commits` (7 jobs). We ship **10 jobs** by additionally splitting `test-base` across Python `3.11` + `3.12` (matrix) and by adding the audit-lesson `zero-domain-deps` job and the defense-in-depth `pre-commit` job. All additions strengthen the gate.
- `cz.toml` does not include a `scope_enum` list (Commitizen's config doesn't enforce scope membership). Scope enforcement lives in `.commitlintrc.yml` (used by the `conventional-commits` CI job). This is functionally equivalent.
- The PDF prescribes `git tag v0.0.1 && git push origin v0.0.1` immediately after merge. Done. Verified via `git describe --tags --abbrev=0` ⇒ `v0.0.1`.
- Branch protection on `main` is **not enabled** per the operator decision tracked in `docs/operational-decisions.md`. The PDF strongly recommends enabling from day 1 and lists the `gh api -X PUT` script; we will run that script before Stage 5 (`v0.1.0-alpha`).

## 5. Verification artifacts

### Local (pre-push)

```
$ .venv/bin/python -c "import rfdf; print(rfdf.__version__)"
0.0.1

$ .venv/bin/rfdf --version
0.0.1

$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
8 files already formatted

$ .venv/bin/mypy
Success: no issues found in 3 source files

$ .venv/bin/pytest -q
.....                                                                    [100%]
5 passed in 1.26s

$ .venv/bin/pre-commit run --all-files
trailing-whitespace.....Passed
end-of-file-fixer.......Passed
check-yaml..............Passed
check-toml..............Passed
check-added-large-files.Passed
check-merge-conflict....Passed
detect-private-key......Passed
ruff....................Passed
ruff-format.............Passed
mypy....................Passed
```

### Remote (PR #1)

All 10 CI jobs ✅ on PR #1 before squash-merge. URLs visible in the
[PR Checks tab](https://github.com/ernesto01louis/rf-direction-finding/pull/1/checks)
and via `gh pr checks 1` post-hoc.

### Coverage at tag

0 % as expected — only smoke tests exist, no functional code yet. `fail_under = 0`
in `pyproject.toml`; the floor rises per stage per ROADMAP.md.

### `zero-domain-deps` audit-lesson check

```
$ .venv/bin/python -c "
import sys, rfdf
banned = {'uhd','pyadi_iio','SoapySDR','torch','torchsig','torchvision',
          'onnx','onnxruntime','runpod','modal','vastai','sky',
          'fastapi','uvicorn','skrf','PyNEC','necpp','ai_orchestrator_client'}
leaked = banned & set(sys.modules.keys())
print('zero-domain-deps OK:' if not leaked else f'LEAKED: {leaked}',
      len(sys.modules), 'modules loaded')
"
zero-domain-deps OK: 139 modules loaded
```

### `AGENTS.md` symlink (audit-lesson check)

```
$ git ls-tree HEAD AGENTS.md
120000 blob 681311eb9cf453d0faddf3aacaec7357e97ba8e9	AGENTS.md
$ readlink AGENTS.md
CLAUDE.md
```

Mode `120000` confirms a real symlink (not a regular file copy).

## 6. Handoff to Stage 2

**Stage 2 inherits:**

- A clean repo with all meta docs in place; **Stage 2 must not duplicate** content
  from CLAUDE.md / VISION.md / ARCHITECTURE.md — extend in place.
- `src/rfdf/{hal,backends/{sdr,rotator,geometry,compute}}/` directories exist as
  empty placeholders (`.gitkeep` only). Stage 2 ships the Protocol classes + mock +
  SigMF file-replay backends here.
- `tests/demo_no_hardware/test_placeholder.py` is the load-bearing CI gate. Stage 2
  **replaces** it with `test_pipeline_smoke.py` that wires mock-SDR → static geometry
  → mock rotator → local compute through a stubbed DOA call.
- Entry-point groups `rfdf.backends.{sdr,rotator,geometry,compute}` are reserved by
  convention; Stage 2 wires the first concrete registrations in `pyproject.toml`.
- `pyproject.toml` already declares the `[sdr-uhd] / [sdr-pluto] / [sdr-soapy] /
  [rotator-hamlib]` extras; Stage 2 does **not** need to bump deps for mock + replay
  backends (those use only base `numpy / sigmf`).

**Stage 2 must NOT:**

- Modify any of the 10 CI gates from §2 (additions OK; removals are red flags).
- Couple any Protocol class to a specific hardware vendor (no `uhd_serial` field on
  `SdrConfig`, no `lock_pps` flag specific to B210, etc.).
- Import `uhd / pyadi-iio / torch / runpod / …` at module load time anywhere in
  `src/rfdf/`. Backends in `src/rfdf/backends/` may import their own SDK — that's
  the *only* place that's allowed.
- Add `numpy / scipy / pydantic / …` version bumps unless Stage 2's algorithms
  genuinely require them; bumps land in their own commit with a CHANGELOG entry.

**Stage 2 acceptance criteria** (mirror in `STAGE-2-OUTPUTS.md`):

- Four Protocol classes in `src/rfdf/hal/{sdr,rotator,geometry,compute}.py`
- Mock + SigMF file-replay backends implementing each Protocol
- Backend discovery via entry-points; `rfdf hw list-backends` shows the catalog
- Property-based tests (Hypothesis) for each Protocol — these become the contract
  every future backend must satisfy
- `tests/demo_no_hardware/test_pipeline_smoke.py` green (replaces placeholder)
- `CHANGELOG.md [0.0.2]` entry, `v0.0.2` tag pushed
- Coverage ≥ 70 % on the new HAL + mock modules

## 7. Convention for future stages

**Every Stage N (for N ∈ {2, 3, 4, 5, 6, 7}) MUST ship a `STAGE-N-OUTPUTS.md` at
the repo root** with the same six sections + this seventh:

1. **Identifiers** — tag, squash-merge SHA, PR URL, date, handoff source PDF
2. **CI gate inventory** — full list of CI jobs green at tag (additions allowed,
   removals require explicit justification in §4)
3. **Decisions taken** — choices the stage PDF left to the operator, with
   cross-references to where each is recorded (do **not** duplicate the decision
   text — link)
4. **Deviations from the stage PDF** — anything skipped, modified, or added.
   "None" is an acceptable value but **must** be stated explicitly. The audit
   lesson here is: a missing deviations section reads as "shipped as spec'd",
   which is unfalsifiable.
5. **Verification artifacts** — pasted output of local + CI verification commands
   that confirm the stage's acceptance criteria. Real numbers, not paraphrases.
6. **Handoff to Stage N+1** — what the next stage inherits, what it must not do,
   acceptance criteria for the next stage (mirroring its handoff PDF).
7. **This section reproduced verbatim** in every stage's output file so the
   convention is self-perpetuating. Update only when the convention itself
   changes (and bump the CHANGELOG when it does).

**Why this convention exists:** the orchestrator audit revealed that *planned*
state and *shipped* state drift apart over time. Handoff PDFs describe the plan;
`STAGE-N-OUTPUTS.md` describes reality. Future-you (or a future contributor, or
a future AI agent) reading at the start of Stage N+1 needs to know what *is*
true, not what was *supposed to be* true.

**When to write the outputs file:** in the same PR that tags the stage — or in
an immediate follow-up PR if the tag PR is already merged (this Stage-1 outputs
file is the latter case). Always before starting Stage N+1.

**When NOT to skip it:** never. If a stage ships without a `STAGE-N-OUTPUTS.md`
the next stage's session must produce one retroactively before doing other work.

# STAGE-3-OUTPUTS

> Real shipped state from Stage 3, intended for Stage 4 (and beyond) to read at
> session start. Per the convention established in
> [STAGE-1-OUTPUTS §7](STAGE-1-OUTPUTS.md#7-convention-for-future-stages),
> every subsequent stage MUST ship a `STAGE-N-OUTPUTS.md` of the same shape.

## 1. Identifiers

| | |
|---|---|
| Stage | 3 — DOA pipeline (classical algorithms) |
| Tag | [`v0.0.3`](https://github.com/ernesto01louis/rf-direction-finding/releases/tag/v0.0.3) |
| Tagged | 2026-05-15 |
| Handoff source | `STAGE-3-doa-pipeline.pdf` + `00-CONTEXT-project-brief.pdf` |
| Pull requests | #5 DSP core · #6 CRLB · #7 calibration · #8 narrowband DOA · #9 2-D/wideband/coherent · #10 model-order · #11 synthetic aperture · #12 Doa class + CLI · plus this release PR |
| Branches | `feat/doa-*` (one short-lived branch per PR, squash-merged) |

## 2. CI gate inventory (10 jobs green at tag)

No CI jobs were added or removed. The `coverage` job's install step was extended to
also install `scikit-rf` and the `crosscheck` extra so the calibration Touchstone path
and the pyArgus cross-checks are exercised — a step modification, not a gate change.

| Job | Stage 3 status |
|---|---|
| `lint (ruff + mypy)` | green; ruff + `mypy --strict` clean on 48 source files |
| `test-base (py3.11)` | green |
| `test-base (py3.12)` | green |
| `test-demo-no-hardware` | green; real MUSIC + a CRLB-bounded 3-estimator gate |
| `test-orchestrator (lazy-import)` | green; base import clean |
| `zero-domain-deps` | green; `scikit-rf` lazy-imported, never at module load |
| `coverage` | green; **fail_under raised 70 → 80**; achieved ~87.8 % |
| `readme-status-truthful` | green; Status line references `v0.0.3` (and `v0.0.2`) |
| `conventional-commits` | green |
| `pre-commit` | green |

## 3. Decisions taken

Ratified by the operator before implementation (see the approved plan,
`/root/.claude/plans/we-are-continuing-development-wobbly-mochi.md`):

| Decision | Choice | Recorded in |
|---|---|---|
| PR strategy | Eight component PRs, CI-green + squash-merged at each | git history (`feat/doa-*`) |
| Cross-validation | Add `pyargus` as a test-only extra; cross-check Bartlett/MVDR vs pyArgus | `pyproject.toml` `[crosscheck]`; `tests/unit/test_doa_narrowband.py` |
| Coverage floor | Ratchet 70 → 80 | `pyproject.toml` `[tool.coverage.report]` |
| Grid evaluation | Vectorised steering manifold (broadcasting) | `docs/doa-algorithms.md` |

## 4. Deviations from the Stage 3 PDF

- **`unitary_esprit`** ships as ESPRIT applied to the forward-backward-averaged
  covariance. This is statistically equivalent to the Haardt & Nossek 1995 formulation
  — FB averaging is the substance of its accuracy gain; the real-valued unitary
  arithmetic transform is an optimisation, deferred.
- **Block-diagonal synthetic-aperture fusion** assembles the per-station covariances
  into a block-diagonal virtual covariance; the optional pilot-phase cross-blocks
  (which would add partial coherent gain) are left zero. Coherent and incoherent
  fusion are full implementations.
- **Khatri-Rao difference-coarray method** (PDF §4) is deferred. It targets
  more-sources-than-sensors for *uncorrelated* sources — tangential to the
  coherent-source spatial-smoothing deliverable, and not in the `[0.0.3]` CHANGELOG.
- **Standalone 2-D Unitary ESPRIT** is deferred; the PDF allows falling back to 2-D
  MUSIC, which ships.
- **MUSIC cross-check**: pyArgus's `DOA_MUSIC` is broken on modern NumPy (it relies on
  removed `np.matrix` scalar-assignment behaviour). MUSIC is instead cross-checked
  analytically (exact null on a noiseless covariance) and against Root-MUSIC + ESPRIT
  agreement. pyArgus `DOA_Bartlett` / `DOA_Capon` work and cross-check Bartlett/MVDR.
- **Example** ships as a runnable script (`examples/01-doa-on-mock-array/demo.py`)
  rather than a Jupyter notebook — the script is the canonical demo, is CI-protected
  by `tests/unit/test_examples.py`, and avoids notebook-execution tooling.
- The demo-no-hardware emitters were moved into the array plane (elevation 0) so
  azimuth DOA on the planar cross is exact; the Stage 2 in-process
  `test_no_domain_libs_loaded_on_import` smoke check was fixed to probe a fresh
  interpreter (it failed once any test imported a domain library).

## 5. Verification artifacts

```
$ rfdf --version
0.0.3

$ ruff check . && ruff format --check . && mypy
All checks passed!
81 files already formatted
Success: no issues found in 48 source files

$ pytest -q --cov
263 passed
Required test coverage of 80.0% reached. Total coverage: 87.82%

$ python examples/01-doa-on-mock-array/demo.py
simulated 3 emitters at azimuths [35.0, 80.0, 125.0] deg (8-element ULA, 20 dB SNR)
  MUSIC  recovered: [35.0, 80.0, 125.0]
  ESPRIT recovered: [35.0, 79.99, 125.0]
demo: DOA pipeline PASS

$ rfdf doa --help        # run / calibrate / benchmark / morph-capture
```

### Stage 3 acceptance-criteria checklist

| Criterion (from the Stage 3 PDF DELIVERABLES) | Status |
|---|---|
| DOA algorithms in `src/rfdf/dsp/doa/` (Bartlett, MVDR, MUSIC, Root-MUSIC, ESPRIT, Unitary ESPRIT) | ✅ |
| 2-D MUSIC; wideband (incoherent + CSSM); spatial smoothing | ✅ |
| Position-domain synthetic aperture (3 fusion modes) | ✅ |
| Calibration framework (`dsp/calibration.py`) | ✅ |
| CRLB calculator + CRLB-bounded test per algorithm | ✅ |
| Number-of-signals estimation (AIC, MDL, SORTE) | ✅ |
| `Doa` orchestration class | ✅ |
| `rfdf doa {run,calibrate,benchmark,morph-capture}` CLI | ✅ |
| `tests/demo_no_hardware/test_pipeline_smoke.py` extended with real MUSIC + CRLB | ✅ |
| `examples/01-doa-on-mock-array/` | ✅ (script) |
| Docs: `doa-algorithms` / `calibration` / `synthetic-aperture` / `crlb` + ARCHITECTURE | ✅ |
| `CHANGELOG.md [0.0.3]`; `v0.0.3` tag | ✅ |
| Coverage ≥ 80 % | ✅ 87.8 % |

## 6. Handoff to Stage 4

**Stage 4 inherits:**

- The complete classical DOA layer under `rfdf.dsp` — steering, covariance, calibration,
  the six narrowband estimators, 2-D MUSIC, wideband, coherent-source smoothing, the
  synthetic aperture, the CRLB calculator, model-order estimation, and the `Doa`
  orchestration class. All pure NumPy/SciPy, all CRLB-verified.
- The `rfdf doa` CLI and the `examples/01-doa-on-mock-array/` demo.
- A `DoaEstimate` / `Doa2DResult` result contract Stage 4's ML estimators can reuse.

**Stage 4 must NOT:**

- Modify the 10 CI gates (additions OK; removals need justification in its §4).
- Change the HAL Protocol surfaces or the `rfdf.dsp` public API without a minor
  version bump.
- Import `torch` / `torchsig` / `onnx` at module load anywhere in `src/rfdf/`. ML
  dependencies live in the `ml` extra and are lazy-imported, exactly as `scikit-rf` is.

**Stage 4 acceptance criteria** (ML-DOA / signal classification — mirror its handoff
PDF): deep-learning models under `src/rfdf/ml/` (datasets, models, training, inference,
export), kept behind the `ml` extra; the `demo-no-hardware` gate stays green; coverage
ratchets per the Stage 4 PDF; `CHANGELOG.md [0.0.4]` entry and `v0.0.4` tag.

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

# STAGE-5-OUTPUTS

> Real shipped state from Stage 5, intended for Stage 6 (and beyond) to read at
> session start. Per the convention established in
> [STAGE-1-OUTPUTS §7](STAGE-1-OUTPUTS.md#7-convention-for-future-stages),
> every subsequent stage MUST ship a `STAGE-N-OUTPUTS.md` of the same shape.

## 1. Identifiers

| | |
|---|---|
| Stage | 5 — Reference hardware backends |
| Tag | `v0.1.0-alpha` |
| Tagged | 2026-05-17 |
| Handoff source | `STAGE-5-reference-hardware-backends.pdf` + `00-CONTEXT-project-brief.md` |
| Pull requests | #28 HAL `status()` · #29 B210 SDR backend · #30 AntRunner rotator + shared GRBL client · #31 GRBL linear-rail geometry · #32 udev rules generator · #33 RTL-SDR contrib backend · #34 KrakenSDR contrib backend · plus this selftest/CLI/examples/docs/release PR |
| Branches | `feat/hardware-backends` (the stage integration branch); one short-lived `feat/hw-*` branch per sub-PR, squash-merged into it |

The stage was built on the `feat/hardware-backends` integration branch with one
sub-PR per major component, squash-merged into it; this final PR adds the
`rfdf hw selftest` extension, the `geometry`/`rotator`/`rotator-server` CLI
sub-groups, example 05, the troubleshooting doc, the `v0.1.0-alpha` release
chores, and this outputs file.

## 2. CI gate inventory

One CI workflow was **added** this stage (`hardware.yml`). The `ci.yml` branch
trigger was retargeted from the dead `feat/ml-stage-4` integration branch to
`feat/hardware-backends`. No CI job was removed.

| Job | Stage 5 status |
|---|---|
| `lint (ruff + mypy)` | green; ruff + `mypy --strict` clean on `src/rfdf` (84 source files) |
| `test-base (py3.11)` | green |
| `test-base (py3.12)` | green |
| `test-demo-no-hardware` | green; the mock pipeline is unaffected by the hardware backends |
| `test-orchestrator (lazy-import)` | green |
| `zero-domain-deps` | green; `import rfdf` loads no `uhd` / `httpx` / hardware SDK |
| `coverage` | green; base `src/rfdf` floor (80%) — the hardware backend modules whose device lifecycle cannot run in CI are omitted (see §4) |
| `coverage-ml` | green; unchanged — Stage 5 touches no `src/rfdf/ml` or compute code |
| `readme-status-truthful` | green |
| `conventional-commits` | green; `hw` added to the commitlint scope-enum |
| `pre-commit` | green |
| `hardware-tests (self-hosted)` | **added** this stage (`hardware.yml`); gated on the `hardware-required` PR label, runs `pytest -m hardware` on a self-hosted runner with the gear attached. Skipped on every GitHub-hosted run. |

## 3. Decisions taken

| Decision | Choice | Recorded in |
|---|---|---|
| `v0.1.0-alpha` tag gate | Tagged on CI-green (mocked-SDK verification); real-hardware verification is a tracked operator follow-up, not a tag blocker | §4, §5 below |
| `SdrSource.status()` HAL gap | Added `status()` to the `SdrSource` Protocol as a sanctioned Stage-2 fix (the one permitted, additive HAL change) rather than backend-specific shims | PR #28, §4 below |
| Hardware verification | No physical B210 / AntRunner / GRBL available in the build environment — hardware backends are unit-tested with mocked vendor SDKs; `@pytest.mark.hardware` suites are skipped in CI and run by the operator on the self-hosted runner | §4, §5 below |

## 4. Deviations from the Stage 5 PDF

This section is load-bearing. A missing deviations section reads as "shipped as
spec'd", which is unfalsifiable.

- **Real-hardware acceptance evidence is deferred.** The PDF's headline
  deliverables — `rfdf hw selftest` output against a real B210 + AntRunner +
  GRBL set, the position-error budget report, and the 100-retune pilot-tone
  phase-repeatability measurement — require physical hardware that was not
  available in the build environment. Every backend is implemented and
  unit-tested against **mocked vendor SDKs**; the UHD / GRBL device lifecycle
  is covered by `@pytest.mark.hardware` suites that are skipped unless
  `RFDF_HARDWARE=1`. `v0.1.0-alpha` is tagged on CI-green; the operator runs
  the real-hardware verification when the gear arrives (the deferred checklist
  is in §6).
- **`SdrSource.status()` added to the HAL.** The PDF (guardrail #7) assumes a
  generic `SdrSource.status()` exists for surfacing device health (GPS/clock
  lock); Stage 2 never created it. Rather than paper over the gap with
  backend-specific shims, `status()` was added to the `SdrSource` Protocol
  (PR #28) — the one permitted, *additive* HAL change — and implemented on the
  `mock` and `file-replay` backends plus a new contract test. The 0.0.4 → 0.1.0
  minor bump covers the API addition.
- **`measure_position_repeatability`, not `test_position_repeatability`.** The
  PDF §4 names the position-error-budget routine `test_position_repeatability`.
  It ships as `measure_position_repeatability` so pytest never collects it as a
  test (a `test_`-prefixed function in importable code is a collection
  footgun).
- **Example 05 ships as a `demo.py` script, not a Jupyter notebook.** The PDF
  §8 describes example 05 as a notebook; it ships as a runnable script,
  matching the Stage 3 / Stage 4 precedent. It is hardware-required and is NOT
  added to the `demo-no-hardware` CI gate or `tests/unit/test_examples.py`;
  with no B210 configured it prints its hardware banner and exits 0.
- **B210 `calibration_pilot` raises `NotImplementedError`.** The B210 backend
  is RX-only this stage (per the PDF guardrail). The pilot tone is radiated by
  *external* hardware (a DDS module / SigGen / coupled B210 TX path); the
  backend's automatic recalibration captures that external pilot. `SdrSource`'s
  `calibration_pilot` (a TX method) therefore raises, like `file-replay`.
- **Hardware backend modules are omitted from the base `coverage` floor.**
  `b210.py`, `_grbl.py`, `antrunner.py`, and `grbl_linear.py` drive real
  devices — their device lifecycle cannot execute in CI. Their *pure* helpers
  (USB-topology parsing, the data-rate envelope, the pilot estimator, GRBL
  status/settings parsers, rail-axis projection) are unit-tested; the full
  backends are verified by the `hardware`-marked suites. The modules are listed
  in `[tool.coverage.run] omit`, mirroring how the Stage-4 cloud compute
  backends are handled. `rfdf/hw/` (udev, selftest, rotctld) is pure Python and
  stays measured under the 80% floor.
- **Contrib backends are outside the core CI gate.** `contrib/rfdf-backend-rtlsdr/`
  and `contrib/rfdf-backend-krakensdr/` are separate pip-installable packages,
  not dependencies of core rfdf. Each has its own `tests/`, verified by
  installing the package and running its `pytest`; they are deliberately not
  part of the core `rfdf` suite so a broken contrib backend never blocks a core
  release.
- **KrakenSDR uses a `HeimdallInterface` Protocol seam.** The PDF recommends
  option (a) — talk to a running Heimdall DAQ daemon. `HeimdallShmInterface` is
  the real shared-memory adapter; its live-attach path needs a running daemon
  and is hardware-verified, while a `HeimdallInterface` Protocol lets the unit
  tests inject an in-memory fake so the whole backend is exercised CI-side.
- **`rfdf hw selftest` now defaults to `--format human`.** The PDF specifies a
  `--format json|human` flag defaulting to human; the pre-existing selftest
  emitted JSON unconditionally. The existing selftest test was updated to pass
  `--format json` — a test modification, not a gate removal.
- **The `00-CONTEXT-project-brief.md §4.5` position-error budget is not in the
  repo.** The brief is an external handoff input, not committed. The 1 mm
  acceptance budget used by `measure_position_repeatability` is taken from the
  Stage-5 PDF's own §4 ("max deviation < 1 mm ... sufficient for synthetic
  aperture at 5.8 GHz").

## 5. Verification artifacts

```
$ .venv/bin/rfdf --version
0.1.0a0

$ .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
All checks passed!
161 files already formatted
Success: no issues found in 84 source files

$ .venv/bin/pytest -q          # full suite, all extras installed
717 passed, 23 skipped, 48 warnings in 288.60s (0:04:48)
# 717 passed = 648 (Stage 4) + 69 net new Stage 5 tests.

$ .venv/bin/pytest tests/demo_no_hardware -q
4 passed, 2 warnings in 38.93s

$ .venv/bin/pytest tests/hardware -q          # skipped without RFDF_HARDWARE=1
11 skipped in 0.17s
# 11 = the B210 + AntRunner + GRBL hardware suites, skipped per tests/conftest.py.

$ python -c "import sys, rfdf; assert 'uhd' not in sys.modules and 'httpx' not in sys.modules"
# zero-domain-deps holds — import rfdf loads no hardware SDK / transport.

$ .venv/bin/rfdf hw list-backends        # b210 / antrunner / grbl-linear registered
$ .venv/bin/rfdf hw udev generate        # emits 70-rfdf.rules for 6 known devices
$ .venv/bin/rfdf hw selftest             # human report, 4/4 backends healthy

# Contrib packages (each its own suite):
$ pip install -e contrib/rfdf-backend-rtlsdr/[dev]    && pytest contrib/rfdf-backend-rtlsdr/tests -q
6 passed
$ pip install -e contrib/rfdf-backend-krakensdr/[dev] && pytest contrib/rfdf-backend-krakensdr/tests -q
6 passed
```

### Stage 5 acceptance-criteria checklist

| Criterion (from the Stage 5 PDF DELIVERABLES) | Status |
|---|---|
| B210 SDR backend at `src/rfdf/backends/sdr/b210.py`, behind `[sdr-uhd]` | ✅ (PR #29) |
| AntRunner rotator backend at `src/rfdf/backends/rotator/antrunner.py` | ✅ (PR #30) |
| GRBL linear-rail geometry backend at `src/rfdf/backends/geometry/grbl_linear.py` | ✅ (PR #31) |
| Contrib backends — RTL-SDR + KrakenSDR, each a separate package | ✅ (PR #33, #34) |
| udev rules system + CLI subcommands | ✅ (PR #32) |
| `rfdf hw selftest` extended with real-hardware checks | ✅ (this PR; `--format json|human`) |
| `rfdf hw geometry` / `rfdf hw rotator` CLI sub-commands | ✅ (this PR; + `rotator-server` rotctld) |
| Position-error budget verification routine | ✅ (`measure_position_repeatability` — §4) |
| `examples/05-real-b210-coherent-capture/` (hardware required) | ✅ (this PR; `demo.py` script — §4) |
| Documentation — 6 hardware docs from §9 | ✅ (this PR) |
| Hardware-marked tests for each backend, run on a self-hosted runner | ✅ (`tests/hardware/`, `hardware.yml`, `hardware-required` label) |
| `CHANGELOG.md [0.1.0-alpha]` entry; `v0.1.0-alpha` tag | ✅ entry this PR; tag applied after merge |
| HAL contract tests passing against real hardware (output committed) | ⚠️ deferred — no physical hardware in the build environment (§4, §6) |
| Branch protection enabled on `main` | ✅ enabled at the close of the stage (deferred from Stage 1) |

## 6. Handoff to Stage 6

**Stage 6 inherits:**

- Three reference hardware backends behind their extras — `b210` (`[sdr-uhd]`),
  `antrunner` (`[rotator-antrunner]`), `grbl-linear` (`[geometry-grbl]`) — plus
  the shared `rfdf.backends._grbl` GRBL-over-HTTP client.
- Two contrib backends as separate packages under `contrib/` — `rfdf-backend-rtlsdr`
  and `rfdf-backend-krakensdr` — the canonical "how to write a contrib backend"
  examples.
- `SdrSource.status()` on the HAL; the udev generator (`rfdf hw udev`); the
  extended `rfdf hw selftest` + `geometry` / `rotator` / `rotator-server` CLI;
  the `rfdf.hw` package (udev, selftest, rotctld).
- The `hardware`-marked test layer + the self-hosted `hardware.yml` workflow.

**Deferred — the operator runs these when physical hardware arrives** (tracked
so the work is not lost):

- `rfdf hw selftest --format human` against the real B210 + AntRunner + GRBL
  set — capture the output and commit it to `docs/`.
- The position-error budget report from the linear rails (`max < 1 mm`).
- **Pilot-tone phase repeatability over 100 retunes** — the single most
  important real-hardware metric.
- HAL contract tests passing against real hardware: label a PR `hardware-required`
  so `hardware.yml` runs `pytest -m hardware` on the self-hosted runner.

**Stage 6 must NOT:**

- Modify the CI gates (additions OK; removals need explicit justification).
- Change the HAL Protocol surfaces or the `rfdf.dsp` / `rfdf.ml` public APIs
  without a minor version bump.
- Import `torch` / `torchsig` / `onnx` / `uhd` / `httpx` / hardware SDKs at
  module load anywhere in `src/rfdf/`. Domain dependencies live behind their
  extras and are lazy-imported.

**Stage 6 acceptance criteria**: mirror the Stage 6 handoff PDF and
[ROADMAP.md](ROADMAP.md). Stage 6 is the documentation + regulatory-compliance
stage (TX EIRP enforcement per band, the full operator handbook).

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
an immediate follow-up PR if the tag PR is already merged.

**When NOT to skip it:** never. If a stage ships without a `STAGE-N-OUTPUTS.md`
the next stage's session must produce one retroactively before doing other work.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (from `v0.1.0` onward) adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`v0.1.0` releases (`v0.0.x`) are pre-alpha — breaking changes are allowed and
recorded here per the release.

## [Unreleased]

## [0.0.2] - 2026-05-15

### Added — Hardware abstraction layer

- Four HAL Protocol classes in `src/rfdf/hal/`: `SdrSource`,
  `RotatorController`, `GeometryController`, `ComputeBackend`. All
  `@runtime_checkable`; async at the I/O boundary.
- Supporting Pydantic models: `SdrConfig`, `StreamBlock`, `Recording`,
  `ComputeJob`, `JobHandle`, `CostEstimate`, plus the `JobStatus` enum,
  `CalibrationReport`, and `BackendLoadError` in `hal/types.py`.
- Backend discovery helper at `hal/discovery.py` with three public callables
  (`discover_backends`, `load_backend`, `list_backends`). Fail-tolerant on
  broken entry-points: WARN + skip on `.load()` failure / non-callable target
  / duplicate name. Factory exceptions are wrapped in `BackendLoadError`.

### Added — Stage 2 backends

- `rfdf.backends.sdr.mock`: synthetic emitter scenarios with a physically-
  faithful array-factor signal model (`x_m(t) = sum_k a_m(theta_k)·s_k(t) +
  n_m(t)`). Emitters: `CWEmitter`, `PilotTone`, `NoiseEmitter`. Optional
  fidelity knobs: `mock_b210_behavior` (per-channel phase randomisation on
  retune), `gain_errors_db`, `mutual_coupling`. `calibration_pilot` is
  decorated with `@requires_eirp_check`.
- `rfdf.backends.sdr.file_replay`: single-channel SigMF playback with
  configurable block size, looping, `seek()`, and optional adversarial
  injection (AWGN + CFO shift). Multi-channel SigMF deferred to Stage 5.
- `rfdf.backends.rotator.mock`: 2-axis rotator with constant-velocity slew,
  Gaussian post-settle noise, `stream_position()` async generator.
- `rfdf.backends.geometry.static`: fixed `(x, y, z)` antenna list.
- `rfdf.backends.geometry.mock_morph`: morphable mock with simulated rail
  repeatability (1-sigma Gaussian) + TOML-backed presets under
  `platformdirs.user_data_path("rfdf")/presets.toml`.
- `rfdf.backends.compute.local`: subprocess execution. CUDA/MPS/CPU
  auto-detected via `nvidia-smi` (no torch import). `cost_estimate` returns
  $0 — local compute is free against the budget. `container_image` deferred
  to Stage 4 with a clear `NotImplementedError`.
- Entry-points registered in `pyproject.toml` under
  `rfdf.backends.{sdr,rotator,geometry,compute}`.

### Added — Configuration + cross-cutting policy

- `src/rfdf/config.py`: Pydantic-settings `RfdfConfig` aggregate with six
  sub-sections (default / sdr / rotator / geometry / compute / eirp). Source
  precedence (cli > env > toml > defaults) implemented via
  `settings_customise_sources`; the **rule itself is documented in exactly
  one place** — [`ARCHITECTURE.md` §4](ARCHITECTURE.md#4-configuration).
  Env vars use `RFDF_` prefix + `__` nested delimiter. Default TOML location
  is `platformdirs.user_config_path("rfdf")/config.toml`; override via
  `$RFDF_CONFIG`.
- `src/rfdf/core/eirp.py`: `EirpCapExceededError` + `enforce_eirp_cap()` +
  `@requires_eirp_check` decorator. Default cap 14 dBm (EU SRD 25 mW
  general). Override gate `eirp.override_explicit` makes the operator
  decision visible in config diffs.

### Added — CLI

- `rfdf hw list-backends`: JSON catalog of every entry-point-registered backend.
- `rfdf hw selftest`: exercises each configured backend (~0.1 s) and emits a
  JSON status report; exit 0 on all-green / exit 1 on any failure.
- `rfdf config show [--format=table|json]`: resolved config with origin
  annotations (cli / env / toml / default) per section.
- `rfdf config validate`: re-loads + re-validates, exit 0 on success.

### Added — Tests + documentation

- `tests/contracts/`: Hypothesis-driven property tests parametrized over
  every entry-point in the relevant HAL group (SDR, rotator, geometry,
  compute). Hardware-only backends skipped in CI.
- `tests/demo_no_hardware/test_pipeline_smoke.py`: replaces the Stage 1
  placeholder. Composes Static geometry + mock rotator + mock SDR + 3
  emitters + local compute end-to-end. Includes a geometry-change-changes-
  IQ assertion that's load-bearing for Stage 3 DOA.
- Tests count: 5 (Stage 1) -> 112 (Stage 2). Coverage floor raised
  `0 -> 70`; achieved ~77%.
- New docs: `docs/hal.md` (Protocol surface + capabilities), `docs/adding-a-
  backend.md` (5-step contributor guide), `docs/configuration.md` (TOML +
  env-var reference). `ARCHITECTURE.md` §2 expanded.
- `tests/conftest.py` `tiny_sigmf` session-scoped fixture: generates a
  0.1 s / 256-sample / single-channel SigMF pair into `tmp_path_factory`
  so no binary ships in the repo (audit-lesson `check-added-large-files`).

### Changed

- Pre-commit `mypy` hook gains `pydantic-settings`, `structlog`,
  `platformdirs`, `numpy` in `additional_dependencies` so the isolated hook
  env can resolve the new imports.
- `src/rfdf/cli/main.py` wires `rfdf hw` and `rfdf config` sub-apps via
  `app.add_typer(...)`.

## [0.0.1] - 2026-05-15

### Added

- Repository scaffold: full directory tree per the Stage 1 handoff bundle
  (`src/rfdf/{hal,backends,dsp,ml,capture,api,cli,orchestrator}`,
  `tests/{unit,integration,hardware,demo_no_hardware}`, `examples/`, `docs/`,
  `ansible/`, `docker-compose/`, `contrib/`, `recipes/`).
- Meta documents: `README.md`, `VISION.md`, `ARCHITECTURE.md`, `ROADMAP.md`,
  `CLAUDE.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md` (this file).
- `AGENTS.md` symlink to `CLAUDE.md` (cross-tool AI-agent context).
- `pyproject.toml` with Hatchling build backend, Python `>=3.11`, **zero RF or ML
  base dependencies**, and the full extras matrix: `sdr-uhd`, `sdr-pluto`,
  `sdr-soapy`, `rotator-hamlib`, `ml`, `ml-onnx`, `ml-hailo`, `compute-runpod`,
  `compute-modal`, `compute-vastai`, `compute-skypilot`, `orchestrator`, `api`,
  `antenna`, `dev`, `docs`, plus convenience meta-extras `all-hardware`,
  `all-compute`, `all`.
- `rfdf` CLI entry point (Typer) with `--version` wired. Subcommands land per stage.
- CI workflows: `ci.yml` (lint, test-base on 3.11 + 3.12 matrix, test-demo-no-hardware,
  test-orchestrator, coverage upload to Codecov, readme-status-truthful,
  conventional-commits, zero-domain-deps), `release.yml` (PyPI Trusted Publishing —
  scaffolded but **inactive** until `v0.1.0`), `pre-commit.yml`, `docs.yml`.
- Pre-commit config: pinned `pre-commit-hooks v5.0.0`, `ruff-pre-commit v0.7.4`,
  `mirrors-mypy v1.13.0`, `commitizen v3.31.0`.
- Conventional Commits: `cz.toml` with locked scope list, `commitlint-github-action`
  on PRs.
- Codecov config (`codecov.yml`), Dependabot config for security advisories only.
- Apache-2.0 LICENSE (auto-created at repo init).
- `Makefile` with `dev / test / lint / format / typecheck / coverage / docs / clean /
  verify` targets.
- `.gitignore`, `.gitattributes`, `.editorconfig`.
- `docs/operational-decisions.md` capturing the branch-protection-deferred and
  SOPS-deferred decisions in **one canonical place** (kills the audit-lesson
  decision-carried-in-two-places pattern).

### Deferred (operator decision, documented in `docs/operational-decisions.md`)

- **Branch protection on `main`** — deferred to enable faster iteration during
  Stages 1–4 (scaffold + HAL + DOA + ML). Revisited and enabled before Stage 5
  (real-hardware integration).
- **SOPS / age-based secret-at-rest encryption** — deferred until a non-solo
  contributor joins. `.env` + `chmod 600` is the interim mechanism. See
  [SECURITY.md](SECURITY.md) §2.
- **Actual PyPI publish** — workflow scaffolded; first publish at `v0.1.0` (post-Stage
  7) per the Stage 1 handoff guidance.

[Unreleased]: https://github.com/ernesto01louis/rf-direction-finding/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/ernesto01louis/rf-direction-finding/releases/tag/v0.0.2
[0.0.1]: https://github.com/ernesto01louis/rf-direction-finding/releases/tag/v0.0.1

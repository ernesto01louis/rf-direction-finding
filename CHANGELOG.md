# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (from `v0.1.0` onward) adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`v0.1.0` releases (`v0.0.x`) are pre-alpha — breaking changes are allowed and
recorded here per the release.

## [Unreleased]

### Added

- `ml-coreml` (`coremltools`) and `ml-tflite` (`ai-edge-torch`) optional-dependency
  extras, completing the Stage 4 ML export surface.
- `coverage-ml` CI job — a dedicated 75% coverage floor for `src/rfdf/ml/` and the
  cloud compute backends, which the base `coverage` job omits because they require
  the `[ml]` / `[compute-*]` extras.
- `rfdf.ml.errors` — `MlError` exception taxonomy with subclasses `DatasetError`,
  `AugmentationError`, `ModelError`, `TrainingError`, `InferenceError`, `ExportError`,
  and `RegistryError`.  Pure stdlib, importable without any ML extras.
- `rfdf.ml.datasets.augmentation` — pure-NumPy IQ augmentation framework with
  `AugmentationConfig` (AWGN, frequency shift, gain variation, IQ imbalance, multipath,
  impulsive noise, sample-rate jitter) and `apply_augmentations`.  Torch-free.
- `rfdf.ml.datasets._torchsig_compat` — TorchSig 2.x compatibility shim (verified
  against TorchSig 2.1.1).  The sole file that imports torchsig; isolates all 2.x
  API surface so future torchsig drift only touches this one module.
- `rfdf.ml.datasets.synthetic` — `make_modulation_dataset` and
  `make_protocol_dataset` backed by TorchSig's iterable dataset, returning
  `torch.utils.data.Dataset` instances with per-item deterministic augmentation.
- `rfdf.ml.datasets.radioml` — RadioML 2018.01A HDF5 loader with SNR/class filters,
  download-on-first-use (separately mockable `_download_radioml`), and CC-BY-NC-SA
  license notice.  Lazy `torch` + `h5py` imports.
- `rfdf.ml.datasets.captured` — SigMF capture loader with directory-glob discovery,
  annotation-based label extraction, fixed-length windowing, and **by-session**
  train/val/test split to prevent data leakage.  Lazy `torch` import.
- `coverage-ml` CI gate activated (removed `if: false`); enforces ≥75% ML coverage.
- `rfdf.ml.models` — four neural-network architectures for RF signal classification,
  all subclassing `RfdfClassifier` (which provides `predict_proba`, `features`, and
  `to_onnx`):
  - `ResNet1D` — 1-D residual CNN over raw IQ as two real channels `(2, num_samples)`;
    configurable depth (`"resnet18"` / `"resnet34"`).  Modulation-classification baseline.
  - `ResNet2D` — 2-D residual CNN (ResNet-18 layout) over a spectrogram /
    time-frequency image `(channels, freq_bins, time_bins)`.
  - `SignalTransformer` — lightweight Transformer encoder over non-overlapping spatial
    patches of a 2-D time-frequency input; uses a `[CLS]` token following ViT conventions.
  - `EfficientNetClassifier` — `torchvision.models.efficientnet_b0` adapted for
    multi-channel RF spectrograms (stem replaced for arbitrary channel count, head
    replaced for `num_classes`; pretrained weights never loaded).
- `rfdf.ml.models.build_model` — lazy factory: maps an architecture name to the right
  module via `importlib.import_module` so that `import rfdf.ml.models` never loads
  `torch`.  Raises `ModelError` for unknown names.
- `rfdf.ml.recipes` — torch-free recipe system: `DatasetSpec`, `ModelSpec`,
  `TrainingSpec`, `ComputeSpec`, and `TrainingRecipe` (all frozen Pydantic models).
  `load_recipe(path)` parses a TOML file.  `TrainingRecipe.to_compute_job(working_dir)`
  compiles the recipe into a HAL `ComputeJob`: converts `timeout_h` → `timeout_s`,
  writes a 3-line `train.py` runpy shim, injects a JSON-serialised recipe via
  `$RFDF_RECIPE`, and honours the XOR `container_image` / `pip_requirements` constraint.
- `rfdf.ml._manifest` — torch-free provenance record: `TrainingManifest` (frozen
  Pydantic), `write_manifest(manifest, path)`, and `current_git_sha()` best-effort
  helper (returns `"unknown"` when git is unavailable).
- `rfdf.ml.datasets.build_datasets(spec)` — dispatches on `DatasetSpec.kind` to return
  `(train, val, test)` datasets with disjoint seed offsets for synthetic kinds.
  All torch imports are lazy inside the function body so `rfdf.ml.datasets` stays
  torch-free at module level.
- `rfdf.ml.training` — backend-agnostic training loop: `train(recipe, train_dataset,
  val_dataset, output_dir, ...)` returning `TrainingResult`.  Features: deterministic
  seeding (random/numpy/torch/CUDA/DataLoader workers), AdamW + linear-warmup cosine
  schedule, AMP (`autocast` + `GradScaler`), gradient accumulation, top-K checkpoint
  retention, DDP support (`DistributedDataParallel` + `DistributedSampler`), IQ→model-
  input adapter (2-tuple `input_shape` → `(2, N)` raw IQ; 3-tuple → `(2, F, T)` STFT
  spectrogram), optional WandB / MLflow (lazy-imported), and `manifest.json` written on
  completion via `rfdf.ml._manifest`.
- `rfdf.ml.train_entrypoint` — thin `<300`-line entry point invoked as
  `python -m rfdf.ml.train_entrypoint <recipe.toml>` or via `$RFDF_RECIPE` env var.
  Logic lives in `rfdf.ml.training`; the entrypoint only resolves the recipe, calls
  `build_datasets` + `train`, and prints a `"<name>: trained best_val_acc=… PASS"` line.
- `src/rfdf/ml/recipes/` — 6 operator-ready TOML recipe templates:
  `sig53-resnet1d-baseline`, `sig53-resnet2d-baseline`, `radioml-resnet2d`,
  `wideband-detection-detr` (uses `transformer` architecture — DETR not yet in the
  platform; noted as a deviation), `protocol-id-resnet2d`, `fingerprint-finetune`.

## [0.0.3] - 2026-05-15

### Added — classical DOA pipeline

- `rfdf.dsp` — the pure-NumPy/SciPy direction-of-arrival layer: steering manifold
  (`dsp.steering`), covariance estimation (`dsp.covariance`), geometry presets, and a
  `DspError` exception taxonomy.
- Narrowband estimators (`dsp.doa`): Bartlett, MVDR, MUSIC, Root-MUSIC, ESPRIT, and
  Unitary ESPRIT, with a shared `DoaEstimate` result type and parabola-refined peak
  picking.
- 2-D (azimuth + elevation) MUSIC with a `Doa2DResult` surface.
- Wideband DOA: incoherent wideband MUSIC and the CSSM coherent signal-subspace method.
- Coherent-source decorrelation (`dsp.coherent`): forward and forward-backward spatial
  smoothing, and Toeplitz reconstruction.
- Position-domain synthetic aperture (`dsp.doa.synthetic_aperture`) with coherent,
  incoherent, and block-diagonal fusion.
- Calibration framework (`dsp.calibration`): pilot-tone and mutual-coupling procedures,
  an identity-calibration loader, and `.npz` + TOML persistence.
- Cramer-Rao lower bound (`dsp.crlb`): the deterministic CRB, a closed-form ULA
  cross-check, and joint azimuth/elevation. Every estimator has a CRLB-bounded test.
- Number-of-sources estimation (`dsp.model_order`): AIC, MDL, and SORTE.
- The `Doa` orchestration class and the `rfdf doa` CLI (`run`, `calibrate`,
  `benchmark`, `morph-capture`).
- `examples/01-doa-on-mock-array/` demo and `docs/{doa-algorithms,calibration,
  synthetic-aperture,crlb}.md`.

### Changed

- Coverage floor raised from 70% to 80%.
- New `crosscheck` optional extra adds `pyargus` for DOA cross-validation tests; the
  `coverage` CI job additionally installs `scikit-rf`.
- `test_pipeline_smoke.py` runs real MUSIC and asserts CRLB-bounded accuracy on three
  estimators — the Stage 1/2 stub DOA is gone.

### Notes

- `unitary_esprit` ships as ESPRIT with forward-backward averaging — statistically
  equivalent to the Haardt & Nossek formulation; the real-valued arithmetic transform
  is deferred. Block-diagonal synthetic-aperture fusion leaves the optional pilot-phase
  cross-blocks zero. The Khatri-Rao difference-coarray method is deferred. See
  `STAGE-3-OUTPUTS.md` §4.

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
- Pre-commit `ruff-pre-commit` pin bumped `v0.7.4 → v0.15.13` to track the
  dev/CI ruff line (`pyproject.toml [dev]` uses `ruff>=0.7`). The stale pin
  caused a format-rule skew — assert-message wrapping changed between ruff
  0.7 and 0.15, so the hook and CI's `ruff format --check` disagreed.
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

# ROADMAP

Seven stages take `rfdf` from empty repo to citation-grade v0.1.0 GA. Each stage ends
with a git tag and a CHANGELOG entry.

| Stage | Status | Target tag | Summary |
|---|---|---|---|
| 1 | DONE | `v0.0.1` | Repository scaffold and conventions (this PR) |
| 2 | DONE | `v0.0.2` | HAL Protocol classes + mock + SigMF file-replay backends |
| 3 | DONE | `v0.0.3` | Classical DOA (MUSIC, ESPRIT, Bartlett, MVDR) + calibration + CRLB tests |
| 4 | DONE | `v0.0.4` | TorchSig ML pipeline + multi-cloud compute backends + model registry |
| 5 | PLANNED | `v0.1.0-alpha` | Reference hardware backends (B210, AntRunner, GRBL rails) |
| 6 | PLANNED | `v0.1.0-beta` | Ansible-provisioned tool ecosystem (Kasm, Guacamole, Authelia, Homepage) |
| 7 | PLANNED | `v0.1.0` | Optional orchestrator integration + PyPI publish |

## Stage 1 — Repository scaffold and conventions (`v0.0.1`)

**Acceptance criteria:**

- Directory tree from 00-CONTEXT §3 exists
- All meta docs written: README, VISION, ARCHITECTURE, ROADMAP, CLAUDE, SECURITY,
  CONTRIBUTING, CHANGELOG, AGENTS.md (symlink to CLAUDE.md)
- `pyproject.toml` with zero RF deps in base + full extras matrix
- CI workflows green: `lint`, `test-base` (3.11 + 3.12 matrix), `test-demo-no-hardware`,
  `test-orchestrator`, `coverage` (Codecov), `readme-status-truthful`,
  `conventional-commits`, `zero-domain-deps`
- Pre-commit hooks installed and pinned
- AGENTS.md is a real symlink (`git ls-tree` shows mode `120000`)
- `rfdf --version` prints `0.0.1`
- Branch protection: **deferred** per operator decision; documented in
  `docs/operational-decisions.md` and revisited before Stage 5

## Stage 2 — Hardware Abstraction Layer (`v0.0.2`)

**Acceptance criteria:**

- Four Protocol classes in `src/rfdf/hal/`: `SdrSource`, `RotatorController`,
  `GeometryController`, `ComputeBackend`
- Mock + SigMF file-replay backends implementing each Protocol
- Backend discovery via entry-points; `rfdf hw list-backends` shows the catalog
- Property-based tests (Hypothesis) for each Protocol — these become the contract
  every future backend must satisfy
- `tests/demo_no_hardware/test_pipeline_smoke.py` green: full mock pipeline runs
  end-to-end with a stubbed DOA call

## Stage 3 — Classical DOA pipeline (`v0.0.3`)

**Acceptance criteria:**

- Algorithms: Bartlett, MVDR, MUSIC, Root-MUSIC, ESPRIT, Unitary ESPRIT (1D + 2D)
- Wideband DOA: incoherent + CSSM
- Spatial smoothing for coherent sources
- Position-domain synthetic aperture with coherent / incoherent / block-diagonal fusion
- Pilot-tone + mutual-coupling calibration framework
- CRLB calculator + CRLB-bounded tests for each algorithm
- Number-of-signals estimation (AIC, MDL, SORTE)
- `examples/01-doa-on-mock-array/` notebook runs end-to-end in < 30s

## Stage 4 — ML pipeline + multi-cloud GPU (`v0.0.4`)

**Acceptance criteria:**

- TorchSig + RadioML dataset loaders, augmentation framework
- Models: ResNet1D, ResNet2D, Transformer, EfficientNet-B0
- Backend-agnostic training loop (DDP, AMP, checkpointing)
- Compute backends: local + RunPod + Vast.ai + Modal + SkyPilot
- Cost estimation + explicit confirmation flow before job submission
- Inference paths: PyTorch, ONNX Runtime, HailoRT (behind `[ml-hailo]`)
- Export: ONNX, HEF, TFLite, CoreML
- Model registry with full provenance manifests

## Stage 5 — Reference hardware backends (`v0.1.0-alpha`)

**Acceptance criteria:**

- B210 backend with multi-device coherent capture + mandatory pilot-tone calibration
- AntRunner rotator backend with closed-loop encoder validation
- GRBL linear-rail geometry backend with sub-mm repeatability
- Contrib backends: RTL-SDR, KrakenSDR (separate packages in `contrib/`)
- `udev` rules generator + installer
- `rfdf hw selftest` extended with real-hardware checks
- Branch protection enabled on `main` (deferred from Stage 1)

## Stage 6 — Tool ecosystem hosting (`v0.1.0-beta`)

**Acceptance criteria:**

- Ansible playbooks for Proxmox-based deployment
- Docker Compose stacks: Homepage, Traefik, Authelia, Kasm, Guacamole, OpenWebRX+,
  JupyterLab, Prometheus/Grafana
- Kasm custom workspace images (Linux RF tools + Wine-based MMANA-GAL Pro + Kali RF)
- Authelia OIDC SSO across all services
- Zero changes to `src/rfdf/` — pure infrastructure stage

## Stage 7 — Orchestrator integration + GA (`v0.1.0`)

**Acceptance criteria:**

- Optional dependency via `[orchestrator]` extra (lazy-imported)
- Consumer registration mirroring `aero-research-platform`
- Evidence bundle production for captures, DOA runs, training (with `quality:
  degraded` flag for fallback runs)
- Hindsight memory writes + L5 vault notes
- Planner-dispatched GNU Radio flowgraphs with validation + deployment
- ntfy alerting on 3 channels
- Coverage ≥ 80%
- Published to PyPI as `rfdf`

## Deferred (post-v0.1.0)

- **Open.Space Mini integration** (when hardware ships March 2026 +): new Phased-Array
  HAL Protocol class
- **Phaser CN0566 integration** as an educational rig
- **Signed artifacts** (Sigstore / cosign) for evidence bundles
- **PyPI publish of contrib backends** — currently they live in `contrib/` and are
  installed via `pip install -e contrib/rfdf-backend-rtlsdr/`
- **Self-hosted CI runner** for `@pytest.mark.hardware` jobs
- **Per-device fingerprinting research** as a downstream consumer repo

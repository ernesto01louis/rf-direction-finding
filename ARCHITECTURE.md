# ARCHITECTURE

## 1. Layered architecture

```
┌───────────────────────────────────────────────────────┐
│ Application:  CLI / REST API / Notebooks / Examples   │
├───────────────────────────────────────────────────────┤
│ Pipeline:     Campaigns / Capture / Train / Infer     │
├───────────────────────────────────────────────────────┤
│ Algorithm:    DOA / DSP / Calibration / ML Models     │
├───────────────────────────────────────────────────────┤
│ HAL:          SdrSource / Rotator / Geometry / Compute│
├───────────────────────────────────────────────────────┤
│ Backend:      mock / file_replay / b210 / runpod / …  │
└───────────────────────────────────────────────────────┘
                Optional: Orchestrator Adapter
                          (lazy-imported)
```

Lower layers know nothing about higher layers. The HAL is the only contract the
algorithm layer relies on; concrete backends are interchangeable.

## 2. HAL contracts

The HAL is implemented in `src/rfdf/hal/` (lands in Stage 2). Four Protocol classes:

| Protocol | Concern | Reference backends |
|---|---|---|
| `SdrSource` | IQ capture, tuning, calibration | mock, file-replay (SigMF), b210, RTL-SDR (contrib), KrakenSDR (contrib) |
| `RotatorController` | Mechanical AZ/EL pointing | mock, AntRunner (GRBL_ESP32), hamlib `rotctld`, SPID |
| `GeometryController` | Per-antenna 3D position | static, mock-morph, grbl-linear (motorized rails) |
| `ComputeBackend` | ML / batch job dispatch | local, runpod, vastai, modal, skypilot |

Stage 2 ships the Protocol definitions plus mock + file-replay implementations.
Stage 5 wires the real hardware backends against the same contracts.

## 3. Backend discovery

Backends register via Python entry-points under four groups:

```toml
[project.entry-points."rfdf.backends.sdr"]
mock = "rfdf.backends.sdr.mock:create"
file-replay = "rfdf.backends.sdr.file_replay:create"

[project.entry-points."rfdf.backends.rotator"]
mock = "rfdf.backends.rotator.mock:create"

[project.entry-points."rfdf.backends.geometry"]
static = "rfdf.backends.geometry.static:create"

[project.entry-points."rfdf.backends.compute"]
local = "rfdf.backends.compute.local:create"
```

Third-party packages (e.g. `rfdf-backend-hackrf`, `rfdf-backend-yaesu`) register the same
way and are discovered automatically. A `rfdf hw list-backends` CLI shows the catalog.

## 4. Configuration

`rfdf` uses Pydantic-settings (`pydantic-settings>=2.3`) with the following precedence
rule (this is **the one place** this rule is documented — do not repeat it elsewhere):

> **CLI flag > environment variable > config file > built-in defaults**

Config file location follows `platformdirs.user_config_path("rfdf")` — typically
`~/.config/rfdf/config.toml` on Linux. The `--config /path/to/config.toml` CLI flag
overrides that location.

## 5. Logging

`structlog` with JSON output by default. When stderr is a TTY, a Rich console handler
attaches for human-readable output. The base configuration ships in `src/rfdf/log.py`
(Stage 2).

## 6. Threading / async model

- **Async I/O** for SDR streams and the REST API (live capture, WebSocket fan-out, job
  submission to remote compute).
- **Sync** for internal DSP code (NumPy / SciPy are CPU-bound; threading them adds
  contention).
- Boundary helpers: `asyncio.to_thread()` and `loop.run_in_executor()` at the well-defined
  boundary between the async HAL and the sync algorithm layer.

## 7. Storage layout

User data lives under `platformdirs.user_data_path("rfdf")` — typically
`~/.local/share/rfdf/` on Linux:

```
~/.local/share/rfdf/
├── captures/         # SigMF .sigmf-meta + .sigmf-data pairs (or .sigmf archives)
├── calibrations/     # per-(sdr_backend, frequency, geometry_hash) Calibration files
├── models/           # trained model registry (one dir per model_id with manifest)
├── bundles/          # local evidence bundles (optionally POSTed to the orchestrator)
└── datasets/         # cached dataset downloads (RadioML, TorchSig sig53, …)
```

All IQ artifacts use the **SigMF** open format. All manifests are JSON with explicit
schema versions.

## 8. Orchestrator integration (optional)

`rfdf.orchestrator` is **lazy-imported**: the module is always importable; concrete
integration classes raise `OrchestratorNotAvailableError` if `ai-orchestrator-client`
is not installed. See [docs/orchestrator/](docs/) (lands in Stage 7) for the consumer
registration pattern, evidence-bundle schema, and the standalone-vs-orchestrator
comparison.

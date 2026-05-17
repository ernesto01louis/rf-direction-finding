# rf-direction-finding

[![CI](https://github.com/ernesto01louis/rf-direction-finding/actions/workflows/ci.yml/badge.svg)](https://github.com/ernesto01louis/rf-direction-finding/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/ernesto01louis/rf-direction-finding/branch/main/graph/badge.svg)](https://codecov.io/gh/ernesto01louis/rf-direction-finding)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-not--yet--published-lightgrey)](#)

> Hardware-agnostic RF direction finding, signal classification, and phased-array research.

## What this is

`rfdf` is a Python platform for RF direction-finding (DOA), signal classification, and
phased-array experimentation. It is **hardware-agnostic** by design: every SDR, rotator,
antenna geometry, and compute provider sits behind an abstract interface (the HAL). The
B210 + AntRunner reference set is the *first concrete validation* of the abstractions, not
the system itself. A user with a HackRF, an RTL-SDR, or just a directory of SigMF files
is a first-class user.

`rfdf` runs **standalone** without `ai-orchestrator`. Installing the `[orchestrator]`
extra makes it a consumer of [`ernesto01louis/ai-orchestrator`](https://github.com/ernesto01louis/ai-orchestrator),
adding evidence-bundle production, Hindsight memory writes, L5 vault notes,
planner-dispatched GNU Radio flowgraphs, and ntfy alerts.

> **Validate RF research workflows on canned data before buying €8500 of gear.**

## Status

**v0.1.0-alpha — reference hardware backends (prior: v0.0.4 ML pipeline).** The
first user-facing pre-release. Stages 1–4 proved the math, the ML, and the
multi-cloud compute on canned data; this stage wires real, imperfect hardware
to the Stage-2 HAL contracts: a **USRP B210** SDR backend with multi-device
coherent capture and mandatory pilot-tone calibration; an **AntRunner** rotator
backend with closed-loop encoder validation; a **GRBL linear-rail** geometry
backend with sub-mm position-error budgeting; **RTL-SDR** and **KrakenSDR**
contrib backends as separate packages under `contrib/`; a `udev` rules
generator; and an extended `rfdf hw` CLI (`selftest` / `geometry` / `rotator` /
`rotator-server`). All hardware SDKs sit behind extras (`[sdr-uhd]`,
`[rotator-antrunner]`, `[geometry-grbl]`) and are lazy-imported — the base
install stays RF/ML/hardware-free, and the whole pipeline is still drivable
end-to-end with zero physical hardware via the mock backends. The API is not
yet stable but the platform is fully functional. See
[STAGE-5-OUTPUTS.md](STAGE-5-OUTPUTS.md) and [ROADMAP.md](ROADMAP.md) for
what's next.

## Install (zero hardware path)

```bash
pip install rfdf  # not yet published; install from source until v0.1.0
```

The base install pulls only platform-essential dependencies (NumPy, SciPy, Pydantic,
Typer, Rich, sigmf, structlog, platformdirs). **Zero RF or ML dependencies are required
to use the platform.** Hardware and compute providers live behind extras and entry points.

## Install (with hardware / compute extras)

| Extras | What it pulls in |
|---|---|
| `[sdr-uhd]` | Ettus UHD (B210 / B200mini / X-series) |
| `[sdr-pluto]` | pyadi-iio (PlutoSDR) |
| `[sdr-soapy]` | SoapySDR (HackRF, LimeSDR, others) |
| `[rotator-hamlib]` | hamlib-py (Yaesu / SPID / generic rotctld) |
| `[ml]` | PyTorch + TorchSig + scikit-learn |
| `[ml-onnx]` | ONNX Runtime for portable inference |
| `[compute-runpod]` | RunPod GPU rental |
| `[compute-modal]` | Modal serverless GPU |
| `[compute-vastai]` | Vast.ai marketplace GPU |
| `[compute-skypilot]` | SkyPilot multi-cloud GPU |
| `[orchestrator]` | `ai-orchestrator-client` integration |
| `[api]` | FastAPI + uvicorn (REST + WebSocket) |
| `[antenna]` | scikit-rf + PyNEC for antenna modelling |
| `[dev]` | Test + lint + type-check toolchain |
| `[all-hardware]` | All SDR + rotator extras together |
| `[all-compute]` | All compute backends together |
| `[all]` | The kitchen sink (ML + API + antenna + hardware + compute + orchestrator) |

## Documentation

| Doc | What's in it |
|---|---|
| [VISION.md](VISION.md) | Why this exists, the principles, when-in-doubt questions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layered design, HAL contracts, backend discovery, config precedence |
| [ROADMAP.md](ROADMAP.md) | Stages 1–7 with status + acceptance criteria |
| [CLAUDE.md](CLAUDE.md) | Working conventions (used by AI coding assistants) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, branch flow, commit conventions |
| [SECURITY.md](SECURITY.md) | Threat model, secret handling, EIRP enforcement |
| [CHANGELOG.md](CHANGELOG.md) | Per-release changes |

## License + acknowledgments

[Apache-2.0](LICENSE). Matches `ai-orchestrator` and `aero-research-platform`.

This project builds on the work of many upstream RF and ML communities. Specific
attributions live in [docs/acknowledgments.md](docs/) once that document is populated
(Stage 3+).

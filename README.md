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

**v0.1.0-beta — software ecosystem hosting (prior: v0.1.0-alpha reference
hardware backends).** Stage 6 is **pure infrastructure** — the platform core
has not changed since `v0.1.0-alpha`. It adds the operator-facing "workshop":
an Ansible-provisioned ecosystem of RF / SDR / EM-simulation tools, hosted
*alongside* the platform on adjacent Proxmox infrastructure and reached through
a browser — Kasm Workspaces, Apache Guacamole, OpenWebRX+, JupyterLab,
code-server, a Homepage dashboard, Authelia SSO behind a Traefik v3 reverse
proxy, and Prometheus/Grafana monitoring. None of it is a dependency of
`pip install rfdf`: a user running `rfdf doa` needs none of these services.
The platform itself remains feature-complete from `v0.1.0-alpha` — a **USRP
B210** SDR backend, an **AntRunner** rotator, a **GRBL linear-rail** geometry
backend, **RTL-SDR**/**KrakenSDR** contrib backends, a `udev` generator, and
the `rfdf hw` CLI — all hardware SDKs behind lazy-imported extras, with the
whole pipeline drivable end-to-end on mock backends. The API is not yet stable
but the platform is fully functional. See
[STAGE-6-OUTPUTS.md](STAGE-6-OUTPUTS.md), `docs/infrastructure/`, and
[ROADMAP.md](ROADMAP.md) for what's next.

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

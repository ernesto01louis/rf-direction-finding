# rfdf User Guide

A complete, start-from-zero guide to the `rfdf` platform — what it is, how to
install it, and how to use every feature. If you have never seen this project
before, start here and read top to bottom.

For the *companion* guide to the hosted open-source RF software stack (GNU Radio,
SDR++, OpenWebRX+, etc. served in a browser), see
[SOFTWARE-STACK.md](SOFTWARE-STACK.md).

---

## Table of contents

1. [What rfdf is](#1-what-rfdf-is)
2. [Glossary](#2-glossary)
3. [Installation](#3-installation)
4. [Your first run (no hardware needed)](#4-your-first-run-no-hardware-needed)
5. [Configuration](#5-configuration)
6. [Complete CLI reference](#6-complete-cli-reference)
7. [Direction-of-arrival (DOA) algorithms](#7-direction-of-arrival-doa-algorithms)
8. [Hardware: the HAL and its backends](#8-hardware-the-hal-and-its-backends)
9. [The machine-learning pipeline](#9-the-machine-learning-pipeline)
10. [Renting GPUs](#10-renting-gpus)
11. [The REST API](#11-the-rest-api)
12. [Orchestrator integration (optional)](#12-orchestrator-integration-optional)
13. [The bundled examples](#13-the-bundled-examples)
14. [End-to-end workflows](#14-end-to-end-workflows)
15. [Getting help and troubleshooting](#15-getting-help-and-troubleshooting)

---

## 1. What rfdf is

`rfdf` ("RF direction finding") is a Python platform for three related kinds of
radio research:

- **Direction finding (DOA)** — given a radio signal arriving at an antenna
  array, estimate the *bearing* it came from.
- **Signal classification** — train a neural network to recognise *what kind* of
  signal it is (the modulation, the protocol, even the individual transmitter).
- **Phased-array experimentation** — try out different antenna geometries,
  wideband captures, and synthetic-aperture techniques.

### The one idea that makes it different: hardware-agnostic

Every piece of hardware — the radio (SDR), the antenna rotator, the array
geometry, the compute that trains models — sits behind an *abstract interface*
called the **HAL** (Hardware Abstraction Layer). The same code runs whether you
have:

- **No hardware at all** — a built-in *mock* radio synthesises the exact IQ
  samples a real array would receive.
- **A €40 RTL-SDR dongle** on a laptop.
- **A €4500 coherent Ettus B210 array** with a precision clock.
- **A directory of recorded `.sigmf` files** from someone else's capture.

You write and validate your workflow once, on synthetic data, and only later
swap in real hardware. The project's tagline says it plainly:

> **Validate RF research workflows on canned data before buying €8500 of gear.**

Every feature is required to work end-to-end on synthetic data — there is a CI
job (`test-demo-no-hardware`) that fails the build if anything couples to real
hardware.

### Standalone first, orchestrator optional

`pip install rfdf` gives you the *complete* platform: DOA, classification,
capture, the CLI, everything. It has no dependency on any server.

Installing the optional `[orchestrator]` extra connects rfdf to a running
[`ai-orchestrator`](https://github.com/ernesto01louis/ai-orchestrator) instance.
That adds citation-grade *evidence bundles*, persistent memory writes, and
planner-generated flowgraphs — **context and traceability, never a different
answer**. You can ignore it entirely and lose nothing core.

### Status

`rfdf` is **v0.1.0 — first stable release** (Stages 1–7 complete, published on
[PyPI](https://pypi.org/project/rfdf/)). The public API is stable under SemVer.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| **DOA** | Direction of Arrival — the bearing an RF signal arrives from |
| **DF** | Direction Finding — the broader application of DOA |
| **SDR** | Software-Defined Radio — a radio whose processing is done in software |
| **IQ** | In-phase/Quadrature samples — the raw complex-valued radio data |
| **HAL** | Hardware Abstraction Layer — `rfdf`'s abstract hardware interfaces |
| **ULA** | Uniform Linear Array — antennas equally spaced on a line |
| **MUSIC** | MUltiple SIgnal Classification — a subspace DOA algorithm |
| **ESPRIT** | Estimation of Signal Parameters via Rotational Invariance Techniques |
| **MVDR** | Minimum Variance Distortionless Response (Capon beamforming) |
| **CRLB** | Cramér-Rao Lower Bound — the theoretical best accuracy any estimator can reach |
| **SigMF** | Signal Metadata Format — the open standard for storing IQ recordings |
| **SEI** | Specific Emitter Identification — fingerprinting individual transmitters |
| **EIRP** | Effective Isotropic Radiated Power — how much power an antenna transmits |
| **B210** | Ettus USRP B210 — the reference SDR hardware |
| **UHD** | USRP Hardware Driver — Ettus's SDR software API |
| **TorchSig** | A PyTorch framework for synthetic RF signal datasets |
| **RadioML** | A public modulation-classification dataset |
| **pilot tone** | A known reference signal used to calibrate an antenna array |

---

## 3. Installation

`rfdf` needs **Python 3.11 or newer**.

### The zero-hardware install

```bash
pip install rfdf
```

This pulls only lightweight, platform-essential dependencies (NumPy, SciPy,
Pydantic, Typer, Rich, sigmf, structlog, platformdirs). **No RF drivers, no
PyTorch, no cloud SDKs.** You can run the full DOA pipeline on synthetic data
immediately.

### Optional extras

Hardware drivers, ML, cloud compute, and integrations live behind *extras* —
install only what you need. The syntax is `pip install 'rfdf[extra1,extra2]'`.

| Extra | Adds | Use it when you want to… |
|---|---|---|
| `sdr-uhd` | Ettus UHD driver | use a USRP B210 / B200mini / X-series |
| `sdr-pluto` | pyadi-iio | use an ADALM-Pluto SDR |
| `sdr-soapy` | SoapySDR | use a HackRF, LimeSDR, or other SoapySDR radio |
| `rotator-hamlib` | hamlib-py | drive a Yaesu / SPID rotator via `rotctld` |
| `rotator-antrunner` | httpx | drive an AntRunner rotator over HTTP |
| `geometry-grbl` | httpx | drive motorised GRBL linear rails |
| `ml` | PyTorch + TorchSig + scikit-learn | train signal classifiers |
| `ml-onnx` | ONNX Runtime | export/run portable ONNX models |
| `ml-coreml` | coremltools | export CoreML models (Apple devices) |
| `ml-tflite` | ai-edge-torch | export TensorFlow Lite models |
| `ml-hailo` | *(vendor SDK, no PyPI package)* | export Hailo HEF models for edge TPUs |
| `compute-runpod` | runpod SDK | train on rented RunPod GPUs |
| `compute-modal` | modal SDK | train on Modal serverless GPUs |
| `compute-vastai` | vastai SDK | train on the Vast.ai GPU marketplace |
| `compute-skypilot` | skypilot | train on AWS/GCP via SkyPilot |
| `api` | FastAPI + uvicorn | run the REST API server |
| `orchestrator` | ai-orchestrator-client | connect to an ai-orchestrator |
| `antenna` | scikit-rf + PyNEC | model antennas / load S-parameter files |
| `dev` | pytest, ruff, mypy, … | develop / test rfdf itself |
| `all-hardware` | every SDR + rotator + geometry extra | — |
| `all-compute` | every compute backend | — |
| `all` | the kitchen sink (ML + API + antenna + hardware + compute + orchestrator) | — |

Examples:

```bash
pip install 'rfdf[ml]'                       # train classifiers, no radio hardware
pip install 'rfdf[sdr-uhd]'                  # talk to a real B210
pip install 'rfdf[ml,compute-runpod]'        # train on a rented RunPod GPU
pip install 'rfdf[api,orchestrator]'         # REST API + orchestrator integration
pip install 'rfdf[all]'                      # everything
```

After installation the `rfdf` command is on your PATH:

```bash
rfdf --version
rfdf --help
```

---

## 4. Your first run (no hardware needed)

With just `pip install rfdf`, run a direction-finding estimate against a
synthetic signal:

```bash
rfdf doa run
```

This builds an **8-element uniform linear array** in software, places a simulated
emitter at 30°, has the mock SDR synthesise the IQ those antennas would receive,
runs the **MUSIC** algorithm, and prints the recovered bearing.

Add more emitters and pick the algorithm:

```bash
rfdf doa run --algorithm music --azimuths "30,80,120" --channels 8 --snr-db 20
```

`--azimuths` lists the true bearings to simulate; rfdf then *estimates* them back
from the synthetic IQ and you can see how close it gets. `--num-signals 0` (the
default) means "I don't know how many sources there are — figure it out", using
the MDL model-order criterion.

You can also run the bundled canonical demo, which checks every estimate lands
within 1° of truth and prints `demo: DOA pipeline PASS`:

```bash
python examples/01-doa-on-mock-array/demo.py
```

That is the whole "is it working?" loop — no radio, no antenna, no cost.

---

## 5. Configuration

### Where config lives

rfdf reads a single TOML file:

- Default location: `~/.config/rfdf/config.toml` (Linux; platform-aware
  elsewhere).
- Override the path with the `RFDF_CONFIG` environment variable.

You do **not** need a config file — every setting has a sensible default. Create
one only to change defaults.

### The precedence rule

When the same setting is given in more than one place, this order wins
(highest first):

```
CLI flag  >  environment variable  >  config.toml  >  built-in default
```

### Annotated example config

```toml
[default]
log_level = "info"                 # debug | info | warning | error | critical
data_dir  = "~/.local/share/rfdf"   # where captures, models, calibrations live

[sdr]
backend         = "mock"            # mock | file-replay | b210
center_freq_hz  = 868e6
sample_rate_hz  = 2e6
bandwidth_hz    = 0                 # 0 / unset => same as sample_rate_hz
rx_gain_db      = 30.0
channels        = [0]
coherent        = false
reference_clock = "internal"        # internal | external | gpsdo
timing_source   = "internal"        # internal | external | gpsdo | pps

[rotator]
backend = "mock"                    # mock | antrunner

[geometry]
backend  = "static"                 # static | mock-morph | grbl-linear
# antennas: list of [x, y, z] positions in metres. The default below is the
# 5-element pentagonal LPDA cluster at half-wavelength spacing for 868 MHz.
antennas = [
  [0.0,   0.0,  0.0],
  [0.17,  0.0,  0.0],
  [0.0,   0.17, 0.0],
  [-0.17, 0.0,  0.0],
  [0.0,  -0.17, 0.0],
]

[compute]
backend                  = "local"  # local | runpod | modal | vastai | skypilot
default_gpu_min_vram_gb  = 16.0
require_cost_confirmation = true     # never auto-submit a paid cloud job

[eirp]
max_eirp_dbm     = 14.0              # 14 dBm = 25 mW — the EU SRD general limit
override_explicit = false            # set true ONLY when deliberately raising the cap
```

### Environment variables

Every setting is also an environment variable: prefix `RFDF_`, and use `__`
(double underscore) to descend into a section.

```bash
export RFDF_SDR__BACKEND=file-replay
export RFDF_SDR__CENTER_FREQ_HZ=2400e6
export RFDF_GEOMETRY__BACKEND=mock-morph
export RFDF_EIRP__MAX_EIRP_DBM=20
```

### The EIRP cap (transmit safety)

`rfdf` enforces a configurable **EIRP cap** so a transmit experiment cannot
accidentally exceed legal radiated power. The default is 14 dBm (25 mW — the EU
short-range-device general limit). Raising it is deliberate: you must set
`override_explicit = true` *and* change `max_eirp_dbm`, which makes the decision
visible in any config diff. rfdf provides guardrails — it does not certify legal
compliance. See [SECURITY.md](../SECURITY.md).

### The data directory

Captures, trained models, calibrations, and local evidence bundles live under
`data_dir` (default `~/.local/share/rfdf/`):

```
~/.local/share/rfdf/
├── captures/        # SigMF .sigmf-meta + .sigmf-data recordings
├── calibrations/    # saved array calibrations
├── models/          # the trained-model registry
├── datasets/        # cached dataset downloads
└── bundles/         # local evidence bundles (orchestrator mode)
```

### Inspecting config

```bash
rfdf config show                 # resolved config, with the source of each section
rfdf config show --format json   # same, as JSON
rfdf config validate             # re-load + re-validate; non-zero exit on error
```

---

## 6. Complete CLI reference

Everything `rfdf` does is reachable from the `rfdf` command. Append `--help` to
any command or subcommand for its exact flags.

```
rfdf
├── hw            Inspect + exercise hardware-abstraction-layer backends
│   ├── list-backends                List all installed backends as JSON
│   ├── selftest [--format human|json]   Exercise every configured backend; exit 1 on failure
│   ├── rotator
│   │   ├── status                   Print the rotator's current (az, el)
│   │   ├── goto AZ EL               Slew the rotator to a bearing
│   │   └── park                     Park the rotator at its safe position
│   ├── rotator-server [--host] [--port 4533]   Serve the Hamlib rotctld protocol (for Gpredict)
│   ├── geometry
│   │   ├── list-presets             List the geometry presets the backend knows
│   │   └── goto PRESET              Move the array to a named preset
│   └── udev
│       ├── list                     List the SDR devices rfdf ships udev rules for
│       ├── generate                 Print the udev rules to stdout
│       └── install                  Install udev rules to /etc/udev/rules.d (needs root)
├── config
│   ├── show [--format table|json]   Print the resolved config with source annotations
│   └── validate                     Re-load + re-validate the config
├── doa            Classical direction-of-arrival estimation on the mock SDR
│   ├── run            Estimate DOA on a synthetic scenario
│   ├── calibrate      Run pilot-tone calibration against a mock SDR and save it
│   ├── benchmark      Sweep estimators across SNR vs. the Cramér-Rao bound
│   └── morph-capture  Capture IQ across morphing-array rail stations
├── ml             Train, export, and manage signal-classification models
│   ├── train          Load a recipe, estimate cost, confirm, submit a training job
│   ├── export MODEL_ID    Export a model to ONNX / HEF / TFLite / CoreML
│   └── registry
│       ├── list                     List registered models
│       ├── show MODEL_ID             Print a model's manifest
│       ├── export MODEL_ID           Bundle a model into a .tar.gz archive
│       ├── import ARCHIVE            Import a model archive
│       └── delete MODEL_ID           Delete a model and its artefacts
├── compute        Inspect + exercise compute backends for ML dispatch
│   ├── list           List every discovered compute backend
│   ├── test [--backend local]        Submit a trivial no-op job to a backend
│   ├── estimate       Print a pre-submit cost estimate
│   ├── jobs           List jobs submitted in this CLI session
│   ├── logs JOB_ID    Print a session job's logs
│   └── cancel JOB_ID  Cancel a session job
├── api
│   └── serve [--host 0.0.0.0] [--port 8001]   Serve the rfdf REST API
└── orchestrator   Optional ai-orchestrator integration (needs the [orchestrator] extra)
    ├── status         Show connection state + declared capabilities
    ├── register       Register this rfdf instance as an orchestrator consumer
    ├── hindsight      Write a Hindsight memory entry (debugging helper)
    ├── vault          Write an L5 vault note (debugging helper)
    └── planner        Request a GNU Radio flowgraph from the planner
```

### Key command flags

**`rfdf doa run`**

| Flag | Default | Meaning |
|---|---|---|
| `--algorithm` | `music` | Estimator: `music`, `bartlett`, `mvdr`, `esprit`, `root_music` |
| `--azimuths` | `30` | Comma-separated true emitter bearings to simulate |
| `--duration` | `0.05` | Capture duration in seconds |
| `--num-signals` | `0` | Source count; `0` auto-estimates via MDL |
| `--freq-hz` | `2.4e9` | RF centre frequency |
| `--snr-db` | `20.0` | Scenario signal-to-noise ratio |
| `--channels` | `8` | Number of ULA antennas |

**`rfdf doa benchmark`** — `--algorithms` (default `music,bartlett,mvdr,esprit,root_music`),
`--snr-range start:stop:step` (default `-5:25:10`), `--trials` (default `30`),
`--output report.html`, `--channels`.

**`rfdf doa calibrate`** — `--name` (required), `--directory`, `--freq-hz`, `--channels`.

**`rfdf doa morph-capture`** — `--directory` (required), `--stations` (default `6`),
`--azimuth`, `--freq-hz`, `--rail-span-m`.

**`rfdf ml train`** — `--recipe/-r` (required), `--compute/-c`, `--epochs/-e`,
`--gpu-count/-g` (`0` = CPU), `--yes/-y` (skip the cost-confirmation prompt).

**`rfdf ml export MODEL_ID`** — `--format/-f` (`onnx` default, or `hef`/`tflite`/`coreml`),
`--output/-o`.

**`rfdf compute estimate`** — `--backend/-b`, `--gpu-model`, `--gpu-count`, `--timeout-h`.

**`rfdf orchestrator register`** — `--base-url` (default `http://localhost:8001`),
`--callback-token`.

---

## 7. Direction-of-arrival (DOA) algorithms

DOA estimation takes the IQ from an antenna array and returns the bearing(s) the
signal(s) came from. `rfdf` ships the classic estimator family plus the
extensions you need for hard cases. The full mathematical reference is
[docs/doa-algorithms.md](doa-algorithms.md); this section is the practical map.

### The classical estimators

These five are directly available from `rfdf doa run --algorithm <name>` and
`rfdf doa benchmark`:

| Algorithm | Pick it when… |
|---|---|
| `bartlett` | you want a quick, robust first look; works on any geometry |
| `mvdr` (Capon) | you need sharper resolution than Bartlett |
| `music` | you want the best accuracy and roughly know the source count |
| `root_music` | you have a uniform linear array and want a fast closed-form result |
| `esprit` | uniform linear array, no spectrum search, very fast |

### The extensions (Python API + `docs/doa-algorithms.md`)

- **2-D MUSIC** — estimate azimuth *and* elevation together.
- **Unitary ESPRIT** — ESPRIT with forward-backward averaging for better accuracy.
- **Wideband DOA** — incoherent wideband MUSIC and the Coherent Signal-Subspace
  Method (CSSM) for signals that occupy a wide bandwidth.
- **Synthetic aperture** — fuse captures taken at multiple positions (a moving
  rail) into one large *virtual* array for far finer angular resolution. Capture
  the stations with `rfdf doa morph-capture`.
- **Coherent-source decorrelation** — spatial smoothing for multipath/reflections
  where ordinary MUSIC fails.
- **Model-order estimation** — AIC / MDL / SORTE estimate how many sources are
  present when you do not know (`--num-signals 0` uses MDL).

### CRLB — knowing how good "good" is

The **Cramér-Rao Lower Bound** is the theoretical best accuracy any unbiased
estimator can achieve for a given array, SNR, and snapshot count. `rfdf doa
benchmark` plots every estimator against the CRLB so you can see which algorithm
is near-optimal for your conditions:

```bash
rfdf doa benchmark --snr-range "-5:25:5" --trials 30 --output benchmark.html
```

Open `benchmark.html` for the report. See [docs/crlb.md](crlb.md).

### Pilot-tone calibration

Real arrays have per-channel gain and phase errors that distort the bearing
estimate. The fix is to capture a known reference signal (a *pilot tone*) and
solve for the corrections. Generate a calibration against the mock SDR with:

```bash
rfdf doa calibrate --name my-array --freq-hz 2.4e9 --channels 8
```

For a real B210 this calibration is **mandatory** — see
[§8](#8-hardware-the-hal-and-its-backends) and
[docs/calibration.md](calibration.md).

### Worked Python example

The verified, runnable Python entry point is the bundled demo — it configures an
array, captures synthetic IQ, and runs MUSIC + ESPRIT:

```bash
python examples/01-doa-on-mock-array/demo.py
```

Read [`examples/01-doa-on-mock-array/demo.py`](../examples/01-doa-on-mock-array/demo.py)
as the canonical template for driving the DOA library from Python.

---

## 8. Hardware: the HAL and its backends

### The four abstract interfaces

The HAL defines four Python `Protocol` classes. Your code targets these; concrete
*backends* implement them:

| Interface | Abstracts | Reference doc |
|---|---|---|
| `SdrSource` | the radio / IQ source | [docs/hal.md](hal.md) |
| `RotatorController` | an antenna rotator | [docs/hal.md](hal.md) |
| `GeometryController` | the array geometry | [docs/hal.md](hal.md) |
| `ComputeBackend` | where ML training runs | [docs/ml/compute-backends.md](ml/compute-backends.md) |

### The backends that ship

You select a backend in config (`[sdr] backend = "..."`, etc.) or via an
`RFDF_*` environment variable. `rfdf hw list-backends` prints everything
installed; new backends register automatically through Python entry points.

**SDR (radio) backends**

| Name | What it is | Needs |
|---|---|---|
| `mock` | synthetic IQ generator — no hardware | nothing |
| `file-replay` | plays back recorded SigMF files | nothing |
| `b210` | Ettus USRP B210 / B200mini / X-series | `[sdr-uhd]` + system UHD |

**Rotator backends**

| Name | What it is | Needs |
|---|---|---|
| `mock` | simulated rotator | nothing |
| `antrunner` | AntRunner rotator over HTTP | `[rotator-antrunner]` |

(For Yaesu/SPID rotators, run `rfdf hw rotator-server` to expose the configured
rotator over the Hamlib `rotctld` protocol — handy for Gpredict satellite
tracking.)

**Geometry backends**

| Name | What it is | Needs |
|---|---|---|
| `static` | a fixed list of antenna positions | nothing |
| `mock-morph` | a *simulated* motorised morphing array | nothing |
| `grbl-linear` | real motorised GRBL linear rails over HTTP | `[geometry-grbl]` |

**Compute backends** — see [§10, Renting GPUs](#10-renting-gpus).

### Verifying hardware: `rfdf hw selftest`

`rfdf hw selftest` exercises the HAL contract against every *configured* backend
and runs a device status probe. Exit code 0 means all healthy, 1 means a
failure. Run it after connecting any real hardware:

```bash
rfdf hw selftest                 # human-readable colour report
rfdf hw selftest --format json   # machine-readable
```

### udev rules (non-root USB SDRs)

USB SDRs usually need a udev rule so you can use them without `sudo`:

```bash
rfdf hw udev list        # which devices rfdf ships rules for
rfdf hw udev generate    # print the rules
sudo rfdf hw udev install    # install to /etc/udev/rules.d and reload
```

Then unplug and replug the radio. See [docs/hardware/udev-rules.md](hardware/udev-rules.md).

### Connecting real hardware — operator checklist

**Ettus B210** (full guide: [docs/hardware/sdr-b210.md](hardware/sdr-b210.md))

1. `pip install 'rfdf[sdr-uhd]'` and install the system UHD driver.
2. Check USB topology — each B210 should sit on its own USB-3 controller
   (`lsusb -t`). Add a PCIe USB-3 card if they share one.
3. For 2+ coherent boards, distribute a 10 MHz + 1 PPS reference with an
   OctoClock-G; set `[sdr] reference_clock = "external"` and
   `timing_source = "external"`.
4. Identify boards with `uhd_usrp_probe`; pass their serials via the
   `RFDF_B210_SERIALS` environment variable (comma-separated).
5. Provide a pilot tone at the centre frequency — the B210's fractional-N PLL
   randomises phase on every retune, so **pilot-tone calibration is mandatory**
   before a coherent capture.
6. `rfdf hw selftest` should be green.

**AntRunner rotator** (guide: [docs/hardware/rotator-antrunner.md](hardware/rotator-antrunner.md))

1. `pip install 'rfdf[rotator-antrunner]'`, set `[rotator] backend = "antrunner"`.
2. `rfdf hw rotator status`, then `rfdf hw rotator goto 90 30` to slew.

**GRBL morphing rails** — `pip install 'rfdf[geometry-grbl]'`,
`[geometry] backend = "grbl-linear"`, then `rfdf hw geometry list-presets` /
`goto`. Guide: [docs/hardware/geometry-grbl-rails.md](hardware/geometry-grbl-rails.md).

**RTL-SDR / KrakenSDR** — these are *contrib* backends in
[`contrib/`](../contrib/). Install one and it auto-registers — no core change
needed. See [docs/hardware/contrib-backends.md](hardware/contrib-backends.md).

---

## 9. The machine-learning pipeline

`rfdf` can train neural networks to classify RF signals (modulation, protocol,
emitter fingerprint). The ML feature set needs the `[ml]` extra:

```bash
pip install 'rfdf[ml]'
```

### The pieces

- **Datasets** — `synthetic` (TorchSig-generated, fast, no hardware), `radioml`
  (the public RadioML benchmark), `captured` (your own SigMF recordings).
  See [docs/ml/datasets.md](ml/datasets.md).
- **Models** — `resnet1d` (fast IQ baseline), `resnet2d` (spectrogram input),
  `efficientnet`, `transformer`. See [docs/ml/models.md](ml/models.md).
- **Recipes** — a training run is described by a TOML *recipe*. Six ready-made
  recipes ship in [`recipes/`](../recipes/):
  `sig53-resnet1d-baseline.toml`, `sig53-resnet2d-baseline.toml`,
  `radioml-resnet2d.toml`, `protocol-id-resnet2d.toml`,
  `fingerprint-finetune.toml`, `wideband-detection-detr.toml`.
- **Registry** — every trained model is stored under `~/.local/share/rfdf/models/`
  with a manifest (training params, dataset, metrics).
- **Export** — convert a trained model to a portable runtime format.

### Training a model

```bash
rfdf ml train --recipe recipes/sig53-resnet1d-baseline.toml --compute local
```

`rfdf ml train` loads the recipe, **always prints a cost estimate**, and then
asks for confirmation before submitting (skip the prompt with `--yes`, or set
`compute.require_cost_confirmation = false` in config). Useful overrides:
`--epochs N`, `--gpu-count 0` (force CPU), `--compute runpod` (route to a
rented GPU — see [§10](#10-renting-gpus)).

The fastest way to see the whole loop is the bundled demo — it builds a tiny
synthetic dataset, trains a `resnet1d` on CPU in under a minute, evaluates it,
and exports to ONNX:

```bash
pip install 'rfdf[ml,ml-onnx]'
python examples/02-train-modulation-classifier/demo.py
```

### Managing trained models

```bash
rfdf ml registry list                    # all trained models
rfdf ml registry show <MODEL_ID>         # one model's manifest + metrics
rfdf ml registry export <MODEL_ID>       # bundle into a .tar.gz to move it
rfdf ml registry import model.tar.gz     # restore a bundle
rfdf ml registry delete <MODEL_ID>       # remove a model + its artefacts
```

### Exporting for deployment

```bash
rfdf ml export <MODEL_ID> --format onnx --output model.onnx
```

| Format | Flag value | Extra needed |
|---|---|---|
| ONNX | `onnx` | `ml-onnx` |
| TensorFlow Lite | `tflite` | `ml-tflite` |
| Apple CoreML | `coreml` | `ml-coreml` |
| Hailo HEF (edge TPU) | `hef` | `ml-hailo` + the Hailo Dataflow Compiler on PATH |

See [docs/ml/training.md](ml/training.md), [docs/ml/registry.md](ml/registry.md),
and [docs/ml/export.md](ml/export.md) for the full detail.

---

## 10. Renting GPUs

Training does not have to run on your machine. The `ComputeBackend` abstraction
lets the *same recipe* run locally or on a rented GPU — you only change
`--compute`.

| Backend | Where it runs | Extra | Credentials |
|---|---|---|---|
| `local` | this machine (CPU or GPU) | — | none |
| `runpod` | RunPod | `compute-runpod` | `RUNPOD_API_KEY` |
| `modal` | Modal serverless | `compute-modal` | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` |
| `vastai` | Vast.ai marketplace | `compute-vastai` | `VAST_AI_API_KEY` |
| `skypilot` | AWS / GCP via SkyPilot | `compute-skypilot` | cloud provider credentials |

### Inspect and cost before you spend

```bash
rfdf compute list                                  # discovered backends + availability
rfdf compute test --backend local                  # submit a trivial no-op job
rfdf compute estimate --backend runpod \
  --gpu-model A6000 --gpu-count 1 --timeout-h 6     # pre-submit USD estimate
```

`rfdf compute estimate` works without credentials — it never submits anything.
`rfdf ml train` always prints the estimate and gates submission behind explicit
confirmation. **rfdf never auto-submits a paid cloud job.**

### Train on a rented GPU

```bash
pip install 'rfdf[ml,compute-runpod]'
export RUNPOD_API_KEY=your_key_here
rfdf ml train --recipe recipes/sig53-resnet1d-baseline.toml --compute runpod
```

Jobs submitted in a CLI session can be followed with `rfdf compute jobs`,
`rfdf compute logs <JOB_ID>`, and `rfdf compute cancel <JOB_ID>`. Jobs started in
an earlier session are managed from the provider's own console. Example 04 walks
the whole cost-aware flow without spending money:

```bash
python examples/04-rent-gpu-and-train/demo.py
```

See [docs/ml/compute-backends.md](ml/compute-backends.md).

---

## 11. The REST API

`rfdf` ships an optional HTTP API. It has two jobs: a small standalone REST
surface, and the *callback target* the orchestrator dispatches work to.

```bash
pip install 'rfdf[api]'
rfdf api serve                       # binds 0.0.0.0:8001 by default
rfdf api serve --host 127.0.0.1 --port 8080
```

### Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/healthz` | never | Liveness probe — `{"status":"ok",...}` |
| `GET` | `/` | never | Service banner |
| `GET` | `/capabilities` | token | List dispatchable capabilities |
| `POST` | `/capabilities/{capability}` | token | Dispatch a capability (orchestrator callback) |
| `GET` | `/docs` | never | Auto-generated Swagger UI |

Open `http://localhost:8001/docs` in a browser for the interactive Swagger UI.

The capability routes only do real work when the `[orchestrator]` extra is also
installed; without it `/capabilities` reports
`"dispatch": "unavailable — install rfdf[orchestrator]"`.

### Authentication

Set the `RFDF_API_TOKEN` environment variable to require a bearer token on the
`/capabilities` routes:

```bash
export RFDF_API_TOKEN=$(openssl rand -hex 32)
rfdf api serve
```

Requests then need `Authorization: Bearer <token>`. `/healthz` and `/` stay open
so infrastructure liveness probes always work. With no token set, the API is
open (the development default).

---

## 12. Orchestrator integration (optional)

This whole section is optional. `rfdf` is fully functional without it.

Installing `[orchestrator]` lets rfdf act as a *consumer* of an
[`ai-orchestrator`](https://github.com/ernesto01louis/ai-orchestrator) instance.
It adds, on top of an ordinary run:

- **Evidence bundles** — citation-grade provenance records (inputs, process,
  outputs, a reproducibility hash) for every DOA run, capture, or training job.
- **Hindsight memory writes** — observations persisted to the orchestrator's
  memory layer.
- **L5 vault notes** — research-diary entries in the orchestrator's knowledge
  vault.
- **Planner-dispatched flowgraphs** — ask the orchestrator's LLM planner to
  generate a GNU Radio flowgraph.
- **ntfy alerts** — push notifications on detections / job completion.

The guarantee, demonstrated by example 06: **orchestrator mode produces the
exact same DOA answer as standalone mode.** It adds context and traceability,
never a different result.

### Setup

```bash
pip install 'rfdf[orchestrator]'
export ORCHESTRATOR_URL=http://your-orchestrator:8000
export ORCHESTRATOR_TOKEN=...        # if the orchestrator requires auth

rfdf orchestrator status             # check connectivity + declared capabilities
```

### Registering as a consumer

So the orchestrator's planner can *discover* rfdf and dispatch work to it,
register this instance and run the API server it will call back into:

```bash
rfdf orchestrator register \
  --base-url http://this-host:8001 \
  --callback-token "$(openssl rand -hex 32)"
rfdf api serve --host 0.0.0.0 --port 8001
```

### Debugging helpers

```bash
rfdf orchestrator hindsight --content "Observed strong emitter at 2.45 GHz"
rfdf orchestrator vault --title "Urban survey day 1" --body "## Findings ..."
rfdf orchestrator planner --prompt "Flowgraph: FM receiver at 100.3 MHz" --name fm-rx
```

Full detail lives in [docs/orchestrator/](orchestrator/) and
[docs/standalone-vs-orchestrator.md](standalone-vs-orchestrator.md).

---

## 13. The bundled examples

Each directory under [`examples/`](../examples/) is a self-contained, runnable
demo with its own README.

| Example | Hardware | What it shows |
|---|---|---|
| [`01-doa-on-mock-array`](../examples/01-doa-on-mock-array/) | none | DOA on a mock array — MUSIC + ESPRIT vs. CRLB |
| [`02-train-modulation-classifier`](../examples/02-train-modulation-classifier/) | none | Full ML loop: synthesise → train → evaluate → ONNX export |
| [`04-rent-gpu-and-train`](../examples/04-rent-gpu-and-train/) | none | The cost-aware cloud-GPU rental flow (no real spend) |
| [`05-real-b210-coherent-capture`](../examples/05-real-b210-coherent-capture/) | B210 | Coherent capture + DOA on real hardware |
| [`06-standalone-vs-orchestrator`](../examples/06-standalone-vs-orchestrator/) | none | Proof that orchestrator mode never changes the answer |

Run any no-hardware example directly:

```bash
python examples/01-doa-on-mock-array/demo.py
python examples/02-train-modulation-classifier/demo.py   # needs rfdf[ml,ml-onnx]
python examples/04-rent-gpu-and-train/demo.py
python examples/06-standalone-vs-orchestrator/demo.py
```

Example 05 needs a real B210; with no `RFDF_B210_SERIALS` set it just prints a
banner and exits cleanly, so it is safe to invoke anywhere.

---

## 14. End-to-end workflows

### Workflow A — direction finding, zero hardware

```bash
pip install rfdf
rfdf doa run --algorithm music --azimuths "30,80" --channels 8
rfdf doa benchmark --snr-range "-5:25:5" --trials 30 --output benchmark.html
```

### Workflow B — train a classifier locally

```bash
pip install 'rfdf[ml,ml-onnx]'
rfdf ml train --recipe recipes/sig53-resnet1d-baseline.toml --compute local --gpu-count 0
rfdf ml registry list
rfdf ml export <MODEL_ID> --format onnx --output classifier.onnx
```

### Workflow C — train on a rented GPU

```bash
pip install 'rfdf[ml,compute-runpod]'
export RUNPOD_API_KEY=...
rfdf compute estimate --backend runpod --gpu-model A6000 --timeout-h 6
rfdf ml train --recipe recipes/sig53-resnet2d-baseline.toml --compute runpod
```

### Workflow D — real B210 capture + DOA

```bash
pip install 'rfdf[sdr-uhd]'
# set [sdr] backend = "b210" in ~/.config/rfdf/config.toml
export RFDF_B210_SERIALS=31D5A3B,31D5A40
rfdf hw selftest
RFDF_B210_SERIALS=31D5A3B,31D5A40 python examples/05-real-b210-coherent-capture/demo.py
```

### Workflow E — standalone vs orchestrator

```bash
pip install 'rfdf[orchestrator,api]'
export ORCHESTRATOR_URL=http://orchestrator:8000
rfdf orchestrator status
python examples/06-standalone-vs-orchestrator/demo.py
```

---

## 15. Getting help and troubleshooting

### Built-in help

```bash
rfdf --help                  # all command groups
rfdf doa --help              # one group's subcommands
rfdf ml train --help         # one command's flags
rfdf config validate         # is my config valid?
rfdf config show             # what config is actually in effect?
rfdf hw list-backends        # what hardware backends are installed?
rfdf hw selftest             # are my configured backends healthy?
```

### Common issues

- **`rfdf: command not found`** — the install directory is not on PATH, or you
  installed into a different virtualenv. Re-activate the venv where you ran
  `pip install rfdf`.
- **`ImportError` for torch / uhd / fastapi** — you used a feature whose extra is
  not installed. Install the matching extra (see [§3](#3-installation)).
- **A hardware backend is "not found"** — check `rfdf hw list-backends`, confirm
  the extra is installed, and for USB SDRs install the udev rules
  ([§8](#8-hardware-the-hal-and-its-backends)).
- **A cloud training job will not submit** — confirm the provider credential
  environment variable is set and run `rfdf compute test --backend <name>`.

### Reference documentation

| Topic | Document |
|---|---|
| Why the project exists, design principles | [VISION.md](../VISION.md) |
| Layered design, HAL contracts, config precedence | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| DOA algorithm mathematics | [docs/doa-algorithms.md](doa-algorithms.md) |
| Cramér-Rao lower bound | [docs/crlb.md](crlb.md) |
| Synthetic aperture | [docs/synthetic-aperture.md](synthetic-aperture.md) |
| Array calibration | [docs/calibration.md](calibration.md) |
| Config syntax in depth | [docs/configuration.md](configuration.md) |
| HAL contracts | [docs/hal.md](hal.md) |
| Writing a new backend | [docs/adding-a-backend.md](adding-a-backend.md) |
| Hardware setup (B210, rotator, rails, udev) | [docs/hardware/](hardware/) |
| ML pipeline (datasets, models, training, export) | [docs/ml/](ml/) |
| Orchestrator integration | [docs/orchestrator/](orchestrator/) |
| General troubleshooting | [docs/troubleshooting.md](troubleshooting.md) |
| The hosted open-source software stack | [SOFTWARE-STACK.md](SOFTWARE-STACK.md) |

If you hit a bug, the project's issue tracker is on GitHub at
[`ernesto01louis/rf-direction-finding`](https://github.com/ernesto01louis/rf-direction-finding).

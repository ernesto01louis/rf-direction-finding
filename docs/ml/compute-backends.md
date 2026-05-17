# Compute backends

A compute backend dispatches a training job — a script plus a working
directory, environment, and GPU spec — to an execution host, reports status,
streams logs, and fetches artefacts on completion. Five backends ship, all
implementing the single `ComputeBackend` Protocol from the Stage-2 HAL
(`rfdf.hal.compute`):

| Backend | Extra | GPU | Persistent storage |
|---|---|---|---|
| `local` | none (in `[ml]`) | host `nvidia-smi` probe | local filesystem |
| `runpod` | `[compute-runpod]` | yes | RunPod Network Volumes |
| `vastai` | `[compute-vastai]` | yes | per-instance disk |
| `modal` | `[compute-modal]` | yes | `modal.Volume` |
| `skypilot` | `[compute-skypilot]` | yes | cloud-native bucket (S3 / GCS / R2) |

The same `TrainingRecipe` runs on any of them — pick the backend in the
recipe's `[compute]` section or with `rfdf ml train --compute <name>`. The four
cloud backends are first-class peers; none is "best". Choose by job shape:
RunPod for the common reserved-GPU case, Vast.ai for cheap batch sweeps that
tolerate interruption, Modal for fast iteration and debugging, SkyPilot when a
job benefits from cheapest-pick across several clouds.

## CLI

```
rfdf compute list                       # discovered backends + auth status
rfdf compute test --backend <name>      # submit a trivial no-op job
rfdf compute estimate --backend <name> [--gpu-model M] [--gpu-count N] [--timeout-h H]
rfdf compute jobs                       # jobs from the current CLI session
rfdf compute logs <job_id>              # tail a job's logs
rfdf compute cancel <job_id>            # cancel a job
```

`rfdf compute test --backend <name>` uses the flat verb form (the brief's
original `compute backends:test` was simplified to a plain Typer sub-command).
For `local` it runs a `print(...)` script end-to-end; for the cloud backends it
reports a clear "needs credentials / SDK" status rather than crashing when
those are absent.

Job state in `rfdf compute jobs / logs / cancel` is **per-session** — a fresh
CLI invocation starts empty. For cloud jobs not submitted in the current
session, use the provider's own console.

## Authentication

The platform reads each provider's credentials from environment variables or
the provider's standard locations. It **never** asks you to paste a key into a
config file.

| Backend | Auth mechanism |
|---|---|
| `local` | none |
| `runpod` | `RUNPOD_API_KEY` environment variable |
| `vastai` | `VAST_API_KEY` env var, or `~/.config/vastai/vast_api_key` (`$XDG_CONFIG_HOME`), or the legacy `~/.vast_api_key` — the Vast.ai SDK's standard resolution order |
| `modal` | `~/.modal.toml` (written by running `modal token new` once), or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` env vars |
| `skypilot` | per-cloud native credentials, validated by running `sky check` once per cloud (e.g. `sky check aws`); the backend does not manage credentials directly |

A missing credential surfaces as a clear `RuntimeError` at first use, not a
crash, and `rfdf compute list` reports the backend as needing setup.

## Cost estimation

Before any submission the platform estimates cost. `rfdf ml train` prints the
estimate and requires confirmation; this is the audit guardrail against a
forgotten cloud instance running up a bill.

Each cloud backend exposes `cost_estimate(job)`. It is **SDK-free** — pure
arithmetic on a static `_RATES` table of indicative hourly GPU prices (USD/hr),
so it works without the provider SDK installed and without credentials. The
formula:

```
runtime_h  = job.timeout_s / 3600
gpu_units  = max(job.gpu_count, 1)
hourly_usd = _RATES.get(job.gpu_model, _RATES[None])   # None = cheapest fallback

estimated_usd = hourly_usd × 1.5 × runtime_h × gpu_units
low_usd       = hourly_usd × 1.0 × runtime_h × gpu_units
high_usd      = hourly_usd × 2.0 × runtime_h × gpu_units
```

The 1.5× multiplier pads for pod startup overhead and modest over-run; the
1.0× / 2.0× bounds form a 2× spread covering typical duration variance. Cost is
tracked in **USD** — `CostEstimate.estimated_usd` on the frozen Stage-2 HAL
model. The `_RATES` tables are indicative and drift as providers change market
pricing; treat the estimate as a guess (actual cost can differ by ±30 %) and
verify against the provider console. The `local` backend reports `$0` — local
compute is free against the budget.

## Container strategy

RunPod, Vast.ai, and SkyPilot run the training script inside a container. Two
modes, picked by the recipe's `ComputeSpec`:

1. **`container_image` set** — the backend pulls that OCI image and runs the
   entry script directly inside it.
2. **`pip_requirements` set** (the default, `["rfdf[ml,ml-onnx]"]`) — the
   backend starts from a stock PyTorch base image and `pip install`s the
   requirements before running the script.

The repo ships a [`Dockerfile`](../../Dockerfile) that builds the dedicated
training image (`FROM pytorch/pytorch:2.5-cuda12.1-cudnn8-runtime` +
`pip install .[ml,ml-onnx]`). A `container.yml` workflow that would publish it
to `ghcr.io/ernesto01louis/rfdf-training` is **scaffolded but disabled**
(`if: false`) — this stage makes no live GHCR push. Until that workflow is
enabled, the cloud backends use the `pip_requirements` fallback into a stock
PyTorch base image, which needs no published image to work.

Modal is a special case: it builds its own `modal.Image` from the same pip
requirements rather than pulling a Docker Hub image — `modal.Image.from_registry`
plus `pip_install`.

When no `container_image` is set, each cloud backend has a sensible default
base image (`runpod/pytorch:...`, `pytorch/pytorch:2.1.0-cuda11.8-...`,
`python:3.11-slim` for Modal).

## Persistent storage

Datasets are large (RadioML is ~20 GB); they should not be re-uploaded per job.
Each backend has a persistent-storage mechanism, abstracted behind the
`RemoteStorage` Protocol in `rfdf.backends.compute._remote_storage`:
`RunpodVolumeStorage`, `ModalVolumeStorage`, `VastAiStorage`,
`SkyPilotBucketStorage`, plus the pure-stdlib `LocalFilesystemStorage` default
and test double. The cloud storage classes are **partial** implementations —
`put` records keys in-process so `exists()` / `list()` work within a session;
`get` raises `NotImplementedError` because downloading from a cloud volume
outside a running cluster needs the provider's native SDK. The docstrings say
so plainly.

## Live testing

`rfdf compute test --backend local` is verified end-to-end in CI. The four
cloud backends are unit-tested with their SDKs **fully mocked** — no real
network calls. Real-connectivity smoke tests live in
`tests/integration/test_compute_<name>_live.py` and are **skipped** unless the
relevant provider secret is set (`RUNPOD_API_KEY`, `MODAL_TOKEN_ID` +
`MODAL_TOKEN_SECRET`, `VAST_API_KEY`, `SKYPILOT_TEST_LIVE=1`). A contributor
without cloud credentials runs the full suite with those tests skipped.

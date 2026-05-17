# ML training pipeline

`rfdf.ml.training.train` is the single backend-agnostic training loop. It knows
nothing about which compute backend it runs on — local CPU, local GPU, or a
rented cloud GPU all execute the same code. Conversely the
[compute backends](compute-backends.md) know nothing about what is being
trained: they dispatch a script, stream logs, and fetch artefacts.

## Recipes

A training run is fully described by a `TrainingRecipe` — a frozen,
TOML-backed, **torch-free** Pydantic model in `rfdf.ml.recipes`. It bundles four
sub-specs:

| Sub-spec | What it pins |
|---|---|
| `DatasetSpec` | dataset kind, class count, impairments, augmentation, seed |
| `ModelSpec` | architecture, `num_classes`, `input_shape`, extra kwargs |
| `TrainingSpec` | epochs, batch size, LR, optimizer, scheduler, AMP, DDP, checkpointing, seed |
| `ComputeSpec` | backend, GPU count / VRAM / model, timeout, container image or pip requirements |

`load_recipe(path)` parses a TOML file into a `TrainingRecipe`. Six recipes
ship under `recipes/` — copy one and edit it:

| Recipe | Task |
|---|---|
| `sig53-resnet1d-baseline.toml` | modulation classification, raw IQ, fast |
| `sig53-resnet2d-baseline.toml` | modulation classification, spectrogram, high accuracy |
| `radioml-resnet2d.toml` | modulation classification on RadioML 2018.01A |
| `protocol-id-resnet2d.toml` | protocol ID across LoRa / Zigbee / WiFi / Bluetooth |
| `fingerprint-finetune.toml` | per-device fingerprinting fine-tune from a backbone |
| `wideband-detection-detr.toml` | wideband signal model — see note below |

`wideband-detection-detr.toml` declares `architecture = "transformer"`: the
platform does **not** ship a DETR object-detection model, so the recipe uses
`SignalTransformer` as the closest available time-frequency architecture. The
recipe file states this inline.

## Running a training job

```bash
rfdf ml train --recipe recipes/sig53-resnet1d-baseline.toml --compute local --epochs 2 --yes
```

`rfdf ml train` loads the recipe, applies CLI overrides (`--compute`,
`--epochs`), prints a cost estimate, and — unless `--yes` is passed or
`compute.require_cost_confirmation = false` — **requires explicit confirmation**
before submitting. This cost-confirmation gate is an audit guardrail: the
platform never auto-submits a paid cloud job.

Programmatically:

```python
from pathlib import Path
from rfdf.ml.recipes import load_recipe
from rfdf.ml.datasets import build_datasets
from rfdf.ml.training import train

recipe = load_recipe(Path("recipes/sig53-resnet1d-baseline.toml"))
train_ds, val_ds, _test = build_datasets(recipe.dataset)
result = train(recipe=recipe, train_dataset=train_ds, val_dataset=val_ds,
               output_dir=Path("outputs/run-001"), device="cpu")
print(result.best_val_accuracy, result.best_checkpoint)
```

`train(...)` returns a `TrainingResult` with `best_checkpoint`,
`best_val_accuracy`, `manifest_path`, `final_step`, and `wall_clock_s`.

## What the loop does

| Feature | Detail |
|---|---|
| Deterministic seeding | `random`, `numpy`, `torch`, CUDA, and DataLoader workers are all seeded from `training.seed` |
| Optimizer | AdamW, weight-decay 0.01 |
| LR schedule | linear warmup → cosine annealing |
| Mixed precision | `torch.autocast` + `GradScaler` when `amp=true` **and** CUDA is available |
| Gradient accumulation | effective batch = `batch_size × grad_accum_steps` |
| Checkpointing | saves every `checkpoint_every_steps` steps; keeps the top-`keep_top_k` by validation accuracy; always writes `models/best.pt` |
| DDP | wraps the model in `DistributedDataParallel` with a `DistributedSampler` when `ddp=true` and `torch.distributed` is initialised — for multi-GPU rented instances |
| Manifest | writes `manifest.json` (training config, dataset version, git SHA, seeds, host, GPU model, evaluation) on completion |

Optional Weights & Biases / MLflow logging is off by default; both are
lazy-imported and used only when `training.wandb` / `training.mlflow` are set.

### IQ→tensor adapter

Datasets yield complex64 IQ; models expect real tensors. The loop's adapter
picks the conversion from the length of `ModelSpec.input_shape`:

- **2-tuple `(2, N)`** — 1-D models: real and imaginary parts become two
  channels; the time axis is interpolated to `N`.
- **3-tuple `(2, F, T)`** — 2-D models: a short-time Fourier transform produces
  a spectrogram; real and imaginary parts of the STFT form two channels,
  interpolated to `(F, T)`.

## The entrypoint

`rfdf.ml.train_entrypoint` is the thin (<300-line) script the compute backends
run. Invoked as `python -m rfdf.ml.train_entrypoint <recipe.toml>` or with the
recipe injected via the `$RFDF_RECIPE` environment variable. All logic lives in
`rfdf.ml.training`; the entrypoint only resolves the recipe, calls
`build_datasets` + `train`, and prints a summary line.

`TrainingRecipe.to_compute_job(working_dir)` compiles a recipe into a HAL
`ComputeJob`: it converts `timeout_h` → `timeout_s`, writes a small `train.py`
runpy shim (the HAL contract wants a file path, not a `-m module` string), and
injects the JSON-serialised recipe via `$RFDF_RECIPE`. The Stage-2 HAL
`ComputeJob` is frozen and carries either `container_image` **or**
`pip_requirements` (an XOR validator) — `to_compute_job` honours that. The
Stage-4 `TrainingRecipe` is a *separate* model that compiles down to a plain
`ComputeJob`; there is no `TrainingJob(ComputeJob)` subclass.

## Augmentation and overfitting

The platform's reputation rests on trained models being *useful*, not merely
trained. A model trained without augmentation overfits capture-day artefacts —
a specific SNR, a specific radio — and collapses on anything else. The classic
failure is a train/test SNR mismatch: train only at +20 dB SNR and the model
learns the noise floor, not the modulation; evaluate at 0 dB and accuracy
falls to chance.

The fix is aggressive [augmentation](datasets.md#augmentation): set `add_awgn`
to a wide SNR range so the model sees the full operating envelope during
training, and add `frequency_shift`, `iq_imbalance`, and `multipath` so it
generalises across receiver hardware and propagation. The shipped recipes set
augmentation ranges accordingly.

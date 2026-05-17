# Model registry

`rfdf.ml.registry` is a local filesystem registry for trained models. Every
model the platform produces can be traced back to its dataset, training config,
git SHA, and evaluation results — the same provenance discipline as the
`ai-orchestrator` evidence bundle. A model card a future-you (or a co-author)
can trust.

## Layout

The registry lives under `~/.local/share/rfdf/models/` by default (via
`platformdirs.user_data_dir`), or the path in the `RFDF_MODEL_REGISTRY`
environment variable — pointing tests at a temp directory is one env var.

Each registered model occupies one directory:

```
~/.local/share/rfdf/models/
└── <model_id>/
    ├── manifest.json     # full provenance + evaluation history
    ├── model.pt          # PyTorch checkpoint (torch.save dict)
    ├── classes.json       # list of class-name strings
    ├── model.onnx         # ONNX export (optional)
    ├── model.hef          # Hailo HEF binary (optional)
    └── confusion.png       # confusion-matrix image (optional)
```

## CLI

```
rfdf ml registry list                       # list registered model IDs
rfdf ml registry show <model_id>            # print the manifest
rfdf ml registry export <model_id> --to <dir>   # bundle to a .tar.gz archive
rfdf ml registry import <archive.tar.gz>    # unpack a bundle into the registry
rfdf ml registry delete <model_id>          # remove a model and its artefacts
```

`export` / `import` move models between machines as a single `.tar.gz`; the
manifest stores export filenames (not absolute paths) so a bundle round-trips
cleanly.

## The manifest

`ModelManifest` is a frozen Pydantic model. `manifest.json` is the audit
record:

| Field | Contents |
|---|---|
| `model_id` | unique registry identifier |
| `architecture` | architecture name (`resnet1d`, `resnet2d`, …) |
| `task` | high-level task (`modulation_classification`, …) |
| `dataset` | `{name, version, hash}` — dataset provenance |
| `training` | `{config, compute_backend, gpu_model, duration_h, cost_eur}` |
| `evaluation` | latest `{accuracy_top1, accuracy_top5, confusion_matrix_path, per_class_metrics}` |
| `provenance` | `{git_sha, trained_at, trainer}` |
| `exports` | filenames of exported artefacts (`onnx`, `hef`, `tflite`, `coreml`) |
| `evaluation_history` | append-only list of all prior evaluations |
| `registered_at` | ISO-8601 UTC timestamp of first registration |

## Manifests are append-only

Once a model is registered, its manifest is **never** mutated in place. A new
evaluation run does not overwrite the `evaluation` field — `register_model`
moves the previous `evaluation` snapshot into `evaluation_history` and records
the new one as the current `evaluation`. `evaluation_history` only ever grows.

This makes the registry auditable: every score a model ever earned is
preserved, so you can see whether accuracy improved or regressed across
re-evaluations, not just the last number. `update_exports` follows the same
discipline — it adds export filenames to the `exports` record after a
successful [export](export.md) without touching the rest of the manifest.

## Class taxonomy stability

The class IDs in `classes.json` must be stable across model versions: index 3
should mean the same modulation in every model that claims compatibility. When
a class is added or removed, that is a dataset change — the dataset version in
`manifest.json` (`dataset.version`) bumps to record it, so a reader can tell
two models apart by their dataset version rather than silently mismatching
label indices.

## Python API

```python
from pathlib import Path
from rfdf.ml.registry import register_model, get_manifest, list_models, load_model

# After a training run wrote outputs/run-001/
manifest = register_model("resnet1d-modulation-v1", Path("outputs/run-001"),
                           architecture="resnet1d")
print(list_models())                       # ['resnet1d-modulation-v1']
m = get_manifest("resnet1d-modulation-v1")  # ModelManifest
checkpoint = load_model("resnet1d-modulation-v1")  # torch.load dict (lazy torch)
```

`load_model` is the only function that imports torch — every other registry
function is pure filesystem / JSON, so `import rfdf.ml.registry` stays
torch-free.

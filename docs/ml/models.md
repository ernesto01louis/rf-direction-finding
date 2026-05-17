# ML models

`rfdf.ml.models` ships four neural-network architectures for RF signal
classification. Every architecture subclasses `RfdfClassifier` and is built by
name through one lazy factory.

`import rfdf.ml.models` is torch-free: the package `__init__` only exposes
`ARCHITECTURES` and `build_model`. Each architecture module imports `torch` at
its own top level, but those modules load only when `build_model` reaches them
via `importlib.import_module`. The `zero-domain-deps` CI job and
`test_ml_lazy_import` enforce this.

## The factory

```python
from rfdf.ml.models import build_model

model = build_model("resnet1d", num_classes=53, input_shape=(2, 1024))
logits = model(batch_tensor)            # (batch, num_classes)
```

`build_model(architecture, *, num_classes, input_shape, **kwargs)` maps an
architecture name to its class and instantiates it. Unknown names raise
`ModelError`. Extra keyword arguments are forwarded to the constructor (e.g.
`depth="resnet34"`).

## The `RfdfClassifier` base

Every model subclasses `rfdf.ml.models._base.RfdfClassifier` (an `nn.Module`)
and inherits a common surface:

| Member | Purpose |
|---|---|
| `forward(x) -> Tensor` | class **logits**, shape `(batch, num_classes)` |
| `predict_proba(x) -> Tensor` | softmax over the logits |
| `features(x) -> Tensor` | penultimate-layer activations (transfer learning, RF fingerprinting) |
| `to_onnx(path, sample_shape=None)` | ONNX export with dynamic batch axis (opset 17) |
| `num_classes`, `input_shape` | shape metadata, used by ONNX export |

`features()` is the fingerprinting hook: the penultimate activations are the
feature vector surfaced in `ClassificationResult.feature_vector` by the torch
inference backend (see [export.md](export.md) for what carries through to ONNX).

## The model zoo

| Name | Class | Input | Notes |
|---|---|---|---|
| `resnet1d` | `ResNet1D` | `(2, num_samples)` raw IQ | 1-D residual CNN over raw IQ as two real channels. `depth` ∈ `{"resnet18", "resnet34"}`. The modulation-classification baseline. |
| `resnet2d` | `ResNet2D` | `(channels, freq_bins, time_bins)` | 2-D residual CNN (ResNet-18 layout, 64→128→256→512 channels, global average pool) over a spectrogram. Generally beats 1-D-on-IQ at high SNR; Conv2D is the Hailo-8L sweet spot. |
| `transformer` | `SignalTransformer` | `(channels, freq_bins, time_bins)` | Lightweight Transformer encoder over non-overlapping `patch_size × patch_size` patches with a `[CLS]` token (ViT layout). `patch_size`, `num_heads`, `num_layers`, `embed_dim` are tunable. Strong on hard cases (low SNR, mixed signals). |
| `efficientnet` | `EfficientNetClassifier` | `(channels, freq_bins, time_bins)` | `torchvision.models.efficientnet_b0` adapted for multi-channel RF spectrograms — stem replaced for arbitrary channel count, head replaced for `num_classes`. Pretrained weights are never loaded. |

The training loop's IQ→tensor adapter feeds 1-D models raw IQ stacked as two
channels `(2, N)`, and 2-D models a STFT spectrogram `(2, F, T)` — picked
automatically from the length of `input_shape`. See
[training.md](training.md#iqtensor-adapter).

## Benchmarks

This stage ships the architectures and the training loop; it does not ship a
curated leaderboard of accuracy figures across datasets. Recording benchmark
numbers belongs with reproducible training runs whose provenance lives in the
[model registry](registry.md) — each model's `manifest.json` carries its own
evaluation metrics. The `examples/02-train-modulation-classifier/` demo and the
`tests/demo_no_hardware/test_ml_pipeline_smoke.py` gate are the load-bearing
"the architectures genuinely learn on synthetic data" checks; for the small,
fast configurations they use, `resnet1d` clears chance comfortably.

The architecture choice is task-driven, not "best"-driven: `resnet1d` for fast
raw-IQ baselines, `resnet2d` for high-SNR spectrogram accuracy and easy Hailo
deployment, `transformer` for difficult low-SNR / mixed-signal cases,
`efficientnet` as the TorchSig-style benchmark backbone.

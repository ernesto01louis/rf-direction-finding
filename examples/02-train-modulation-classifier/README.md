# 02 — Train a signal classifier

A small, fast, end-to-end walk through the `rfdf.ml` pipeline. It needs no
hardware and no GPU: TorchSig synthesises the training data, and the whole run
finishes on a CPU in well under a minute.

```sh
python examples/02-train-modulation-classifier/demo.py
```

[`demo.py`](demo.py) does the full loop:

1. Configures the `local` compute backend.
2. Builds a small synthetic dataset (four distinct signal families — LoRa,
   WiFi, Bluetooth, noise) and splits a held-out validation set.
3. Trains a `resnet1d` classifier through `rfdf.ml.training.train` — CPU-only,
   fixed-seed, a handful of epochs.
4. Evaluates on the held-out split and asserts the model clears chance.
5. Exports the trained model to ONNX with `rfdf.ml.export.export_onnx`.
6. Runs `onnxruntime` inference on a synthetic capture.

It exits `0` and prints `demo: ML pipeline PASS` once every step succeeds.

The demo wraps the TorchSig dataset in a per-sample RMS-normalisation step —
standard RF-ML preprocessing that puts every signal on a common power scale so
a small model converges quickly. The brief frames example 02 as a "modulation
classifier"; protocol-ID is the same kind of signal-classification task, and
the protocol dataset (unlike `make_modulation_dataset`, which yields one
recording per class) gives the many fast-to-generate samples a learnable demo
needs. See [`STAGE-4-OUTPUTS.md`](../../STAGE-4-OUTPUTS.md) §4.

For the full ML surface — datasets and augmentation, the model zoo, the
training loop, the compute backends, the registry, and export targets — see
[`docs/ml/`](../../docs/ml/). To train at scale on a rented cloud GPU, see
[`examples/04-rent-gpu-and-train/`](../04-rent-gpu-and-train/) and the
`rfdf ml` / `rfdf compute` CLI groups.

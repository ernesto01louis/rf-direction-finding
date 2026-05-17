"""Train a signal classifier end-to-end — the canonical "is the ML pipeline working?" demo.

Builds a small synthetic dataset with TorchSig (four distinct signal families:
LoRa, WiFi, Bluetooth, and noise), trains a ResNet1D classifier through the real
``rfdf.ml`` training loop, evaluates on a held-out split, exports the trained
model to ONNX, and runs inference on a synthetic capture. No hardware, no GPU —
the whole run finishes on a CPU in well under a minute.

Requires the ``[ml]`` and ``[ml-onnx]`` extras (``pip install rfdf[ml,ml-onnx]``).

Run it from the repository root::

    python examples/02-train-modulation-classifier/demo.py
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from rfdf.hal.discovery import discover_backends
from rfdf.ml.datasets.synthetic import make_protocol_dataset
from rfdf.ml.export import export_onnx
from rfdf.ml.models import build_model
from rfdf.ml.recipes import TrainingRecipe
from rfdf.ml.training import train

# Four signal families that map to *distinct* TorchSig generators, so the
# classes are genuinely separable: LoRa -> chirp-spread-spectrum, WiFi -> OFDM,
# Bluetooth -> GFSK/QPSK, plus a Gaussian-noise class.
PROTOCOLS = ["lora", "wifi", "bluetooth", "noise"]
IQ_SAMPLES = 512
SAMPLES_PER_CLASS = 48  # kept small so the demo finishes well under 60 s on a CPU
SEED = 0


class NormalizedIQDataset(Dataset):  # type: ignore[type-arg]
    """Per-sample RMS-normalised IQ — standard RF-ML preprocessing.

    TorchSig signals span a wide dynamic range; scaling every recording to unit
    RMS power puts them on a common scale so a small model converges quickly.
    """

    def __init__(self, base: Dataset) -> None:  # type: ignore[type-arg]
        """Wrap *base*, normalising each item on access.

        Args:
            base: The underlying dataset yielding ``(complex64 IQ, int label)``.
        """
        self._base = base
        self.class_names: list[str] = list(base.class_names)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return the RMS-normalised ``(iq, label)`` for item *idx*."""
        iq, label = self._base[idx]
        rms = torch.sqrt(torch.mean(torch.abs(iq) ** 2))
        if rms > 0:
            iq = iq / rms
        return iq, label


def main() -> None:
    """Run the modulation-classifier training demo and print PASS on success."""
    # 1. Configure the local compute backend (the demo trains in-process; this
    #    shows the same discovery path a cloud run would use).
    compute = discover_backends("rfdf.backends.compute")
    if "local" not in compute:
        raise RuntimeError("local compute backend not discovered")
    backend = compute["local"]()
    print(f"compute backend: {backend.name} (supports_gpu={backend.supports_gpu})")

    # 2. Build a small synthetic dataset and a held-out validation split.
    print(f"generating synthetic signals for {PROTOCOLS} ...")
    base = make_protocol_dataset(
        protocols=PROTOCOLS,
        num_samples_per_protocol=SAMPLES_PER_CLASS,
        num_iq_samples=IQ_SAMPLES,
        impairments="cabled",
        seed=SEED,
    )
    dataset = NormalizedIQDataset(base)
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    split = int(0.8 * len(indices))
    train_ds: Subset = Subset(dataset, indices[:split])  # type: ignore[type-arg]
    val_ds: Subset = Subset(dataset, indices[split:])  # type: ignore[type-arg]
    train_ds.class_names = dataset.class_names  # type: ignore[attr-defined]  # for classes.json
    print(f"  dataset: {len(train_ds)} train / {len(val_ds)} val, {len(PROTOCOLS)} classes")

    # 3. Train a small ResNet1D through the real rfdf.ml training loop.
    recipe = TrainingRecipe.model_validate(
        {
            "name": "example-02-modulation-classifier",
            "dataset": {
                "kind": "protocol",
                "num_samples_per_signal": SAMPLES_PER_CLASS,
                "num_iq_samples": IQ_SAMPLES,
                "seed": SEED,
            },
            "model": {
                "architecture": "resnet1d",
                "num_classes": len(PROTOCOLS),
                "input_shape": [2, IQ_SAMPLES],
                "extra_kwargs": {"depth": "resnet18"},
            },
            "training": {
                "epochs": 12,
                "batch_size": 32,
                "learning_rate": 2e-3,
                "warmup_steps": 4,
                "checkpoint_every_steps": 0,
                "keep_top_k": 1,
                "seed": SEED,
            },
            "compute": {"backend": "local"},
        }
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        print(f"training resnet1d for {recipe.training.epochs} epochs on CPU ...")
        result = train(
            recipe=recipe,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=out_dir,
            device="cpu",
        )
        print(
            f"  trained: best val accuracy {result.best_val_accuracy:.3f} "
            f"in {result.wall_clock_s:.1f}s ({result.final_step} steps)"
        )

        # 4. The held-out accuracy must clear chance (4 classes -> 0.25).
        chance = 1.0 / len(PROTOCOLS)
        if result.best_val_accuracy <= chance + 0.05:
            raise AssertionError(
                f"validation accuracy {result.best_val_accuracy:.3f} did not clear "
                f"chance ({chance:.3f}) by a clear margin"
            )

        # 5. Export the best checkpoint to ONNX.
        model = build_model(
            "resnet1d",
            num_classes=len(PROTOCOLS),
            input_shape=(2, IQ_SAMPLES),
            depth="resnet18",
        )
        checkpoint = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        onnx_path = out_dir / "model.onnx"
        export_onnx(model, torch.zeros(1, 2, IQ_SAMPLES), onnx_path)
        print(f"  exported ONNX model -> {onnx_path.name}")

        # 6. Run inference on a synthetic capture via ONNX Runtime.
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # silence CPU-affinity warnings
        session = ort.InferenceSession(str(onnx_path), sess_options)
        sample_iq, true_label = val_ds[0]
        x = torch.stack([sample_iq.real, sample_iq.imag], dim=0).unsqueeze(0).float().numpy()
        logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
        predicted = int(np.argmax(logits))
        print(
            f"  inference on a synthetic capture: predicted '{PROTOCOLS[predicted]}' "
            f"(true '{PROTOCOLS[int(true_label)]}')"
        )

    print("demo: ML pipeline PASS")


if __name__ == "__main__":
    main()

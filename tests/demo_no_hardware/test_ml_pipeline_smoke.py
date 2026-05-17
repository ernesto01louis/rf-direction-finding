"""Demo-no-hardware ML pipeline smoke test.

The ML half of the load-bearing "every algorithm works on synthetic data"
gate (the DOA half lives in :mod:`test_pipeline_smoke`). Trains a tiny
``resnet1d`` classifier on TorchSig-synthesised signals through the real
``rfdf.ml`` training loop and asserts the model genuinely learns — validation
accuracy must clear chance.

No hardware, no GPU: TorchSig synthesises the data, training runs on CPU with
a fixed seed, and the whole test budget is under a minute.

**This test MUST stay green for the rest of the project's life** — it is the
ML counterpart of the DOA smoke gate. It runs in the ``test-demo-no-hardware``
CI job, which installs the ``[ml,ml-onnx]`` extras.
"""

from __future__ import annotations

import pytest

# The whole module needs torch / torchsig. Skip cleanly when the [ml] extra is
# absent so the torch-free DOA smoke tests in this directory still run.
pytest.importorskip("torch")

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from rfdf.ml.datasets.synthetic import make_protocol_dataset
from rfdf.ml.recipes import TrainingRecipe
from rfdf.ml.training import train

# Four signal families mapping to *distinct* TorchSig generators (LoRa ->
# chirp-spread-spectrum, WiFi -> OFDM, Bluetooth -> GFSK/QPSK, plus noise) so
# the classes are genuinely separable.
_PROTOCOLS = ["lora", "wifi", "bluetooth", "noise"]
_IQ_SAMPLES = 512
_SAMPLES_PER_CLASS = 64  # 4 * 64 = 256 items -> ~205 train / ~51 val; budget < 60 s
_SEED = 0


class _NormalizedIQDataset(Dataset):  # type: ignore[type-arg]
    """Per-sample RMS-normalised IQ — standard RF-ML preprocessing.

    TorchSig signals span a wide dynamic range; scaling each recording to unit
    RMS power puts the classes on a common scale so a small model converges.
    """

    def __init__(self, base: Dataset) -> None:  # type: ignore[type-arg]
        """Wrap *base*, normalising each item on access."""
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


def test_ml_pipeline_trains_on_synthetic_data(tmp_path: Path) -> None:
    """A tiny resnet1d trained on synthetic signals clears chance accuracy.

    The load-bearing ML gate: the training loop must genuinely learn from
    TorchSig synthetic data. Four separable classes give a 0.25 chance level;
    the run must comfortably exceed 0.30.
    """
    base = make_protocol_dataset(
        protocols=_PROTOCOLS,
        num_samples_per_protocol=_SAMPLES_PER_CLASS,
        num_iq_samples=_IQ_SAMPLES,
        impairments="cabled",
        seed=_SEED,
    )
    dataset = _NormalizedIQDataset(base)
    indices = list(range(len(dataset)))
    random.Random(_SEED).shuffle(indices)
    split = int(0.8 * len(indices))
    train_ds: Subset = Subset(dataset, indices[:split])  # type: ignore[type-arg]
    val_ds: Subset = Subset(dataset, indices[split:])  # type: ignore[type-arg]

    recipe = TrainingRecipe.model_validate(
        {
            "name": "ml-pipeline-smoke",
            "dataset": {
                "kind": "protocol",
                "num_samples_per_signal": _SAMPLES_PER_CLASS,
                "num_iq_samples": _IQ_SAMPLES,
                "seed": _SEED,
            },
            "model": {
                "architecture": "resnet1d",
                "num_classes": len(_PROTOCOLS),
                "input_shape": [2, _IQ_SAMPLES],
                "extra_kwargs": {"depth": "resnet18"},
            },
            "training": {
                "epochs": 25,
                "batch_size": 32,
                "learning_rate": 2e-3,
                "warmup_steps": 4,
                "checkpoint_every_steps": 0,
                "keep_top_k": 1,
                "seed": _SEED,
            },
            "compute": {"backend": "local"},
        }
    )

    # Pin torch to a single CPU thread for the duration of training. CPU
    # training of this tiny model is not bit-reproducible run-to-run (TorchSig
    # synthesis + multi-threaded float-reduction order both drift), and the
    # original 12-epoch / 48-per-class recipe landed close enough to the 0.30
    # gate that a bad run dipped to ~0.28 and failed CI. Single-threading
    # narrows the spread; the recipe below (longer training, more data — see
    # `_SAMPLES_PER_CLASS` / `epochs`) lifts the whole distribution so every
    # run clears the gate with a wide margin. The thread count is saved +
    # restored so the rest of the pytest process is unaffected.
    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train(
            recipe=recipe,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=tmp_path / "run",
            device="cpu",
        )
    finally:
        torch.set_num_threads(_prev_threads)

    # 4 separable classes -> 0.25 chance. The strengthened recipe lands every
    # run at >= 0.50 (six local runs spanned 0.50-0.62); the 0.30 gate keeps a
    # wide >= 0.20 margin so genuine run-to-run variance never trips it.
    assert result.best_val_accuracy > 0.30, (
        f"resnet1d only reached val accuracy {result.best_val_accuracy:.3f} — "
        "the ML pipeline did not learn from synthetic data"
    )
    # The training loop must have written its artefacts.
    assert result.best_checkpoint.exists()
    assert result.manifest_path.exists()

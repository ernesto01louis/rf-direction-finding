"""Rent a GPU and train — the cost-aware cloud-compute walkthrough.

Discovers the RunPod cloud compute backend, builds a representative training
job, prints its cost estimate, shows the cost-confirmation gate, and then runs
the training on the *local* backend as the compare-with-baseline step.

It does NOT submit a real cloud job. A live RunPod submission needs the SDK and
credentials::

    pip install rfdf[compute-runpod]
    export RUNPOD_API_KEY=...

With no credentials configured, this demo runs the cost-estimate + local
baseline path. RunPod's ``cost_estimate`` is pure arithmetic (no SDK, no
credentials required), so the estimate always prints.

Requires the ``[ml]`` and ``[ml-onnx]`` extras (``pip install rfdf[ml,ml-onnx]``).

Run it from the repository root::

    python examples/04-rent-gpu-and-train/demo.py
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from rfdf.hal.discovery import discover_backends
from rfdf.ml.datasets.synthetic import make_protocol_dataset
from rfdf.ml.recipes import TrainingRecipe
from rfdf.ml.training import train

PROTOCOLS = ["lora", "wifi", "bluetooth", "noise"]
IQ_SAMPLES = 512
SAMPLES_PER_CLASS = 48  # kept small so the local baseline finishes under 60 s on a CPU
SEED = 0


class NormalizedIQDataset(Dataset):  # type: ignore[type-arg]
    """Per-sample RMS-normalised IQ — standard RF-ML preprocessing."""

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


def _build_recipe(backend_name: str) -> TrainingRecipe:
    """Build a representative GPU training recipe targeting *backend_name*."""
    return TrainingRecipe.model_validate(
        {
            "name": "example-04-rent-gpu-and-train",
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
            "compute": {
                "backend": backend_name,
                "gpu_count": 1,
                "gpu_min_vram_gb": 16.0,
                "timeout_h": 3.0,
            },
        }
    )


def main() -> None:
    """Run the GPU-rental walkthrough and print PASS on success."""
    catalog = discover_backends("rfdf.backends.compute")

    # 1. Discover the RunPod cloud backend (import-safe — no SDK needed yet).
    if "runpod" not in catalog:
        raise RuntimeError("runpod compute backend not discovered")
    runpod_backend = catalog["runpod"]()
    print(f"cloud backend: {runpod_backend.name} (supports_gpu={runpod_backend.supports_gpu})")

    # 2. Build a representative GPU training job for RunPod.
    cloud_recipe = _build_recipe("runpod")
    with tempfile.TemporaryDirectory() as job_tmp:
        job = cloud_recipe.to_compute_job(Path(job_tmp))
        print(
            f"  job: {cloud_recipe.model.architecture}, "
            f"{cloud_recipe.training.epochs} epochs, {job.gpu_count}x GPU "
            f">= {job.gpu_min_vram_gb:.0f} GB VRAM, timeout {job.timeout_s / 3600:.1f} h"
        )

        # 3. Cost estimate — pure arithmetic; works with no SDK / credentials.
        estimate = asyncio.run(runpod_backend.cost_estimate(job))
        print(
            f"  RunPod cost estimate: ${estimate.estimated_usd:.4f} "
            f"(low ${estimate.low_usd:.4f} / high ${estimate.high_usd:.4f})"
        )
        print(f"    rationale: {estimate.rationale}")

    # 4. The cost-confirmation gate. The platform NEVER auto-submits a paid
    #    cloud job — submission needs explicit confirmation (`rfdf ml train`
    #    prompts, or pass `--yes`). No credentials are configured here, so the
    #    demo does not submit; it falls through to the local baseline.
    print("  cost-confirmation gate: a live submission requires explicit operator")
    print("    confirmation (rfdf ml train --compute runpod). No RUNPOD_API_KEY")
    print("    configured -> skipping cloud submission, running the local baseline.")

    # 5. Run the same recipe on the local backend as the compare-with-baseline
    #    step — exactly what a real workflow does before committing cloud spend.
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
    train_ds.class_names = dataset.class_names  # type: ignore[attr-defined]

    local_recipe = _build_recipe("local")
    with tempfile.TemporaryDirectory() as out_tmp:
        print(f"training the local baseline for {local_recipe.training.epochs} epochs on CPU ...")
        result = train(
            recipe=local_recipe,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=Path(out_tmp),
            device="cpu",
        )
        print(
            f"  local baseline: best val accuracy {result.best_val_accuracy:.3f} "
            f"in {result.wall_clock_s:.1f}s"
        )
        chance = 1.0 / len(PROTOCOLS)
        if result.best_val_accuracy <= chance + 0.05:
            raise AssertionError(
                f"local baseline accuracy {result.best_val_accuracy:.3f} did not clear "
                f"chance ({chance:.3f}) by a clear margin"
            )

    print("demo: GPU-rental walkthrough PASS")


if __name__ == "__main__":
    main()

"""Behavioural tests for the rfdf.ml training loop.

Requires torch (``[ml]`` extra).  The test module skips automatically on a
base install without torch.

The test builds a tiny in-code TrainingRecipe (4 classes, 1024 IQ samples,
resnet1d, 20 epochs, batch 16, CPU, fixed seed) and runs ``train(...)`` to
verify:

* A :class:`~rfdf.ml.training.TrainingResult` is returned.
* The best checkpoint file exists on disk.
* ``manifest.json`` was written and parses correctly.
* ``best_val_accuracy > 0.30`` (clear of the 0.25 chance level for 4 classes).

The test is designed to be CPU-fast (< 90 s wall-clock).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _make_tiny_recipe(num_classes: int = 4, num_iq_samples: int = 1024) -> TrainingRecipe:  # type: ignore[name-defined]  # noqa: F821
    """Build a minimal TrainingRecipe for fast CPU tests."""
    from rfdf.ml.recipes import (
        ComputeSpec,
        DatasetSpec,
        ModelSpec,
        TrainingRecipe,
        TrainingSpec,
    )

    return TrainingRecipe(
        name="test-tiny-resnet1d",
        dataset=DatasetSpec(
            kind="modulation",
            num_signals=num_classes,
            num_samples_per_signal=64,
            num_iq_samples=num_iq_samples,
            impairments="clean",
            seed=42,
        ),
        model=ModelSpec(
            architecture="resnet1d",
            num_classes=num_classes,
            input_shape=(2, num_iq_samples),
            extra_kwargs={"depth": "resnet18"},
        ),
        training=TrainingSpec(
            epochs=20,
            batch_size=16,
            learning_rate=1e-3,
            warmup_steps=0,
            checkpoint_every_steps=9999,  # only end-of-epoch checkpoint
            keep_top_k=2,
            seed=42,
            amp=False,
            ddp=False,
            wandb=False,
            mlflow=False,
        ),
        compute=ComputeSpec(backend="local", gpu_count=0),
    )


def test_train_returns_result(tmp_path: Path) -> None:
    """train() returns a TrainingResult with sensible values."""
    from rfdf.ml.datasets import build_datasets
    from rfdf.ml.training import TrainingResult, train

    recipe = _make_tiny_recipe()
    train_ds, val_ds, _ = build_datasets(recipe.dataset)

    result = train(
        recipe=recipe,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=tmp_path,
        device="cpu",
    )

    assert isinstance(result, TrainingResult)
    assert result.final_step > 0
    assert result.wall_clock_s > 0.0


def test_best_checkpoint_exists(tmp_path: Path) -> None:
    """The best checkpoint file is written to disk."""
    from rfdf.ml.datasets import build_datasets
    from rfdf.ml.training import train

    recipe = _make_tiny_recipe()
    train_ds, val_ds, _ = build_datasets(recipe.dataset)

    result = train(
        recipe=recipe,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=tmp_path,
        device="cpu",
    )

    assert result.best_checkpoint.exists(), (
        f"best_checkpoint={result.best_checkpoint} does not exist"
    )


def test_manifest_written_and_parseable(tmp_path: Path) -> None:
    """manifest.json is written and contains required keys."""
    from rfdf.ml.datasets import build_datasets
    from rfdf.ml.training import train

    recipe = _make_tiny_recipe()
    train_ds, val_ds, _ = build_datasets(recipe.dataset)

    result = train(
        recipe=recipe,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=tmp_path,
        device="cpu",
    )

    assert result.manifest_path.exists(), "manifest.json not written"
    with open(result.manifest_path) as fh:
        manifest = json.load(fh)

    assert "training_config" in manifest
    assert "git_sha" in manifest
    assert "evaluation" in manifest
    assert "best_val_accuracy" in manifest["evaluation"]


def test_val_accuracy_above_chance(tmp_path: Path) -> None:
    """best_val_accuracy > 0.30 (well above the 0.25 chance level for 4 classes).

    This sanity-checks that the loop trains rather than producing random
    logits.  The threshold is conservative to avoid flaky failures on CPU.
    """
    from rfdf.ml.datasets import build_datasets
    from rfdf.ml.training import train

    recipe = _make_tiny_recipe(num_classes=4)
    train_ds, val_ds, _ = build_datasets(recipe.dataset)

    result = train(
        recipe=recipe,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=tmp_path,
        device="cpu",
    )

    assert result.best_val_accuracy > 0.30, (
        f"best_val_accuracy={result.best_val_accuracy:.4f} <= 0.30 — "
        "the training loop may not be learning."
    )


def test_progress_callback_called(tmp_path: Path) -> None:
    """progress_cb is called during training."""
    from rfdf.ml.datasets import build_datasets
    from rfdf.ml.training import train

    calls: list[dict[str, object]] = []

    def _cb(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    recipe = _make_tiny_recipe()
    train_ds, val_ds, _ = build_datasets(recipe.dataset)

    # Use a very small checkpoint_every_steps so the callback fires.
    from rfdf.ml.recipes import TrainingSpec

    recipe2 = recipe.model_copy(
        update={
            "training": TrainingSpec(
                **{**recipe.training.model_dump(), "checkpoint_every_steps": 1}
            )
        }
    )
    train(
        recipe=recipe2,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=tmp_path,
        device="cpu",
        progress_cb=_cb,
    )

    # At least one call should have happened.
    assert len(calls) >= 1
    first = calls[0]
    assert "step" in first
    assert "val_acc" in first

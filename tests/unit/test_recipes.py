"""Tests for the recipe system (rfdf.ml.recipes).

Verifies that:

1. All 6 shipped TOML recipes parse into valid :class:`TrainingRecipe` objects.
2. :meth:`TrainingRecipe.to_compute_job` produces a :class:`ComputeJob` that:
   * passes Pydantic validation (XOR: exactly one of container_image / pip_requirements);
   * has ``timeout_s == timeout_h * 3600``;
   * has the expected artifact_globs;
   * writes a ``train.py`` runpy shim into ``working_dir``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Locate the repo-root recipes/ directory relative to this test file.
_RECIPES_DIR = Path(__file__).parent.parent.parent / "recipes"
_RECIPE_NAMES = [
    "sig53-resnet1d-baseline",
    "sig53-resnet2d-baseline",
    "radioml-resnet2d",
    "wideband-detection-detr",
    "protocol-id-resnet2d",
    "fingerprint-finetune",
]


def _recipe_path(name: str) -> Path:
    return _RECIPES_DIR / f"{name}.toml"


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _RECIPE_NAMES)
def test_load_recipe_valid(name: str) -> None:
    """Each shipped TOML parses into a valid TrainingRecipe."""
    from rfdf.ml.recipes import TrainingRecipe, load_recipe

    path = _recipe_path(name)
    assert path.exists(), f"Recipe file not found: {path}"
    recipe = load_recipe(path)
    assert isinstance(recipe, TrainingRecipe)
    assert recipe.name == name
    assert recipe.dataset.kind in {"modulation", "protocol", "radioml", "captured"}
    assert recipe.model.architecture in {"resnet1d", "resnet2d", "transformer", "efficientnet"}
    assert recipe.training.epochs > 0
    assert recipe.training.batch_size > 0
    assert recipe.compute.timeout_h > 0


# ---------------------------------------------------------------------------
# to_compute_job tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _RECIPE_NAMES)
def test_to_compute_job(name: str, tmp_path: Path) -> None:
    """to_compute_job produces a valid ComputeJob for each recipe."""
    from rfdf.hal.compute import ComputeJob
    from rfdf.ml.recipes import load_recipe

    recipe = load_recipe(_recipe_path(name))
    job = recipe.to_compute_job(working_dir=tmp_path)

    assert isinstance(job, ComputeJob)

    # XOR: exactly one of container_image / pip_requirements must be populated.
    has_image = job.container_image is not None
    has_reqs = len(job.pip_requirements) > 0
    assert has_image ^ has_reqs, (
        f"Expected exactly one of container_image/pip_requirements; "
        f"got container_image={job.container_image!r}, pip_requirements={job.pip_requirements!r}"
    )


@pytest.mark.parametrize("name", _RECIPE_NAMES)
def test_timeout_conversion(name: str, tmp_path: Path) -> None:
    """timeout_s == timeout_h * 3600."""
    from rfdf.ml.recipes import load_recipe

    recipe = load_recipe(_recipe_path(name))
    job = recipe.to_compute_job(working_dir=tmp_path)
    expected = recipe.compute.timeout_h * 3600.0
    assert abs(job.timeout_s - expected) < 1e-6, f"timeout_s={job.timeout_s} != {expected}"


@pytest.mark.parametrize("name", _RECIPE_NAMES)
def test_artifact_globs(name: str, tmp_path: Path) -> None:
    """The expected artifact patterns are always included."""
    from rfdf.ml.recipes import load_recipe

    recipe = load_recipe(_recipe_path(name))
    job = recipe.to_compute_job(working_dir=tmp_path)
    expected_globs = {
        "models/*.pt",
        "models/*.onnx",
        "manifest.json",
        "classes.json",
        "confusion.png",
    }
    assert expected_globs.issubset(set(job.artifact_globs)), (
        f"Missing globs: {expected_globs - set(job.artifact_globs)}"
    )


@pytest.mark.parametrize("name", _RECIPE_NAMES)
def test_runpy_shim_written(name: str, tmp_path: Path) -> None:
    """The train.py runpy shim is written into working_dir."""
    from rfdf.ml.recipes import load_recipe

    recipe = load_recipe(_recipe_path(name))
    recipe.to_compute_job(working_dir=tmp_path)

    shim = tmp_path / "train.py"
    assert shim.exists(), "train.py shim was not written"
    content = shim.read_text()
    assert "runpy" in content
    assert "rfdf.ml.train_entrypoint" in content


def test_container_image_recipe(tmp_path: Path) -> None:
    """When container_image is set, pip_requirements must be empty."""
    from rfdf.ml.recipes import (
        ComputeSpec,
        DatasetSpec,
        ModelSpec,
        TrainingRecipe,
        TrainingSpec,
    )

    recipe = TrainingRecipe(
        name="test-container",
        dataset=DatasetSpec(kind="modulation", num_signals=4, num_samples_per_signal=8),
        model=ModelSpec(architecture="resnet1d", num_classes=4, input_shape=(2, 256)),
        training=TrainingSpec(epochs=1, batch_size=4),
        compute=ComputeSpec(
            container_image="python:3.11-slim",
            pip_requirements=[],  # must be empty when container_image is set
        ),
    )
    job = recipe.to_compute_job(working_dir=tmp_path)
    assert job.container_image == "python:3.11-slim"
    assert job.pip_requirements == []


def test_recipe_env_contains_rfdf_recipe(tmp_path: Path) -> None:
    """RFDF_RECIPE env var is set in the ComputeJob."""
    from rfdf.ml.recipes import load_recipe

    recipe = load_recipe(_recipe_path("sig53-resnet1d-baseline"))
    job = recipe.to_compute_job(working_dir=tmp_path)
    assert "RFDF_RECIPE" in job.env
    import json

    parsed = json.loads(job.env["RFDF_RECIPE"])
    assert parsed["name"] == "sig53-resnet1d-baseline"

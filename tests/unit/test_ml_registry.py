"""Tests for the model registry (rfdf.ml.registry).

All tests use a temporary registry root via the RFDF_MODEL_REGISTRY environment
variable to avoid touching the real user data directory.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rfdf.ml.errors import RegistryError  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tiny_training_output(tmp_path: Path, num_classes: int = 4) -> Path:
    """Create a minimal training output_dir that register_model can ingest."""
    import torch

    from rfdf.ml.models import build_model

    out = tmp_path / "train_out"
    models_dir = out / "models"
    models_dir.mkdir(parents=True)

    model = build_model("resnet1d", num_classes=num_classes, input_shape=(2, 64))
    # Save as a full checkpoint dict (the format training.py writes).
    torch.save(
        {
            "step": 1,
            "val_acc": 0.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
        },
        models_dir / "best.pt",
    )

    # classes.json
    classes = [f"mod_{i}" for i in range(num_classes)]
    with open(out / "classes.json", "w") as fh:
        json.dump(classes, fh)

    # manifest.json (TrainingManifest format)
    manifest_data = {
        "training_config": {"input_shape": [2, 64], "epochs": 1},
        "dataset_version": "test",
        "git_sha": "abc123",
        "completed_at": "2026-01-01T00:00:00+00:00",
        "evaluation": {"best_val_accuracy": 0.5, "final_step": 1},
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest_data, fh)

    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the registry at a temp directory."""
    root = tmp_path / "registry"
    root.mkdir()
    monkeypatch.setenv("RFDF_MODEL_REGISTRY", str(root))
    return root


# ---------------------------------------------------------------------------
# Tests: register / get_manifest / list / delete
# ---------------------------------------------------------------------------


def test_register_creates_manifest(tmp_path: Path, registry_root: Path) -> None:
    """register_model writes manifest.json and model.pt into the registry."""
    from rfdf.ml.registry import get_manifest, register_model

    out = _build_tiny_training_output(tmp_path)
    manifest = register_model("test-model-1", out, architecture="resnet1d")

    assert manifest.model_id == "test-model-1"
    assert manifest.architecture == "resnet1d"
    assert (registry_root / "test-model-1" / "manifest.json").exists()
    assert (registry_root / "test-model-1" / "model.pt").exists()

    # get_manifest round-trip
    loaded = get_manifest("test-model-1")
    assert loaded.model_id == "test-model-1"
    assert loaded.architecture == "resnet1d"


def test_register_copies_classes(tmp_path: Path, registry_root: Path) -> None:
    """register_model copies classes.json into the registry."""
    from rfdf.ml.registry import register_model

    out = _build_tiny_training_output(tmp_path)
    register_model("test-model-classes", out, architecture="resnet1d")

    classes_path = registry_root / "test-model-classes" / "classes.json"
    assert classes_path.exists()
    with open(classes_path) as fh:
        classes = json.load(fh)
    assert classes == ["mod_0", "mod_1", "mod_2", "mod_3"]


def test_list_models(tmp_path: Path, registry_root: Path) -> None:
    """list_models returns sorted model IDs."""
    from rfdf.ml.registry import list_models, register_model

    assert list_models() == []

    for name in ("beta", "alpha", "gamma"):
        out = _build_tiny_training_output(tmp_path / name)
        register_model(name, out, architecture="resnet1d")

    assert list_models() == ["alpha", "beta", "gamma"]


def test_list_models_empty_registry(registry_root: Path) -> None:
    """list_models returns [] when the registry is empty."""
    from rfdf.ml.registry import list_models

    assert list_models() == []


def test_delete_model(tmp_path: Path, registry_root: Path) -> None:
    """delete_model removes the model directory."""
    from rfdf.ml.registry import delete_model, list_models, register_model

    out = _build_tiny_training_output(tmp_path)
    register_model("to-delete", out, architecture="resnet1d")
    assert "to-delete" in list_models()

    delete_model("to-delete")
    assert "to-delete" not in list_models()
    assert not (registry_root / "to-delete").exists()


def test_delete_model_not_found(registry_root: Path) -> None:
    """delete_model raises RegistryError for unknown models."""
    from rfdf.ml.registry import delete_model

    with pytest.raises(RegistryError, match="not found"):
        delete_model("nonexistent-model")


def test_get_manifest_not_found(registry_root: Path) -> None:
    """get_manifest raises RegistryError for unknown models."""
    from rfdf.ml.registry import get_manifest

    with pytest.raises(RegistryError, match="not found"):
        get_manifest("does-not-exist")


# ---------------------------------------------------------------------------
# Tests: append-only evaluation history
# ---------------------------------------------------------------------------


def test_evaluation_history_append_only(tmp_path: Path, registry_root: Path) -> None:
    """Re-registering a model appends to evaluation_history, never overwrites."""
    from rfdf.ml.registry import _EvaluationInfo, get_manifest, register_model

    out = _build_tiny_training_output(tmp_path)
    register_model("ev-model", out, architecture="resnet1d")

    # First registration: no history yet.
    m1 = get_manifest("ev-model")
    assert len(m1.evaluation_history) == 0

    # Second registration: prior evaluation is appended to history.
    register_model(
        "ev-model",
        out,
        architecture="resnet1d",
        evaluation=_EvaluationInfo(accuracy_top1=0.75),
    )
    m2 = get_manifest("ev-model")
    assert m2.evaluation.accuracy_top1 == pytest.approx(0.75)
    # The first registration's evaluation is now in history.
    assert len(m2.evaluation_history) >= 1

    # Third registration: both prior evaluations preserved.
    register_model(
        "ev-model",
        out,
        architecture="resnet1d",
        evaluation=_EvaluationInfo(accuracy_top1=0.90),
    )
    m3 = get_manifest("ev-model")
    assert m3.evaluation.accuracy_top1 == pytest.approx(0.90)
    assert len(m3.evaluation_history) >= 2
    # Older entries are intact (not overwritten).
    assert any(abs(e.accuracy_top1 - 0.75) < 1e-6 for e in m3.evaluation_history)


# ---------------------------------------------------------------------------
# Tests: load_model
# ---------------------------------------------------------------------------


def test_load_model(tmp_path: Path, registry_root: Path) -> None:
    """load_model returns the checkpoint dict."""
    from rfdf.ml.registry import load_model, register_model

    out = _build_tiny_training_output(tmp_path)
    register_model("load-test", out, architecture="resnet1d")

    ckpt = load_model("load-test")
    assert "model_state_dict" in ckpt
    assert ckpt["step"] == 1


def test_load_model_not_found(registry_root: Path) -> None:
    """load_model raises RegistryError for unknown models."""
    from rfdf.ml.registry import load_model

    with pytest.raises(RegistryError):
        load_model("no-such-model")


# ---------------------------------------------------------------------------
# Tests: export / import round-trip
# ---------------------------------------------------------------------------


def test_export_import_round_trip(tmp_path: Path, registry_root: Path) -> None:
    """export_model produces a .tar.gz; import_model restores it correctly."""
    from rfdf.ml.registry import (
        delete_model,
        export_model,
        get_manifest,
        import_model,
        list_models,
        register_model,
    )

    out = _build_tiny_training_output(tmp_path)
    register_model("export-test", out, architecture="resnet1d")

    archive = export_model("export-test", tmp_path / "archives")
    assert archive.exists()
    assert archive.suffix == ".gz" or archive.name.endswith(".tar.gz")

    # Verify it is a valid tar.gz.
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getnames()
    assert any("manifest.json" in m for m in members)

    # Delete from registry, then import back.
    delete_model("export-test")
    assert "export-test" not in list_models()

    restored_id = import_model(archive)
    assert restored_id == "export-test"
    assert "export-test" in list_models()

    restored = get_manifest("export-test")
    assert restored.architecture == "resnet1d"


def test_import_invalid_archive_raises(tmp_path: Path, registry_root: Path) -> None:
    """import_model raises RegistryError for a non-model archive."""
    from rfdf.ml.registry import import_model

    # Create a tar.gz without a manifest.json inside.
    bad_archive = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_archive, "w:gz") as tar:
        dummy = tmp_path / "dummy.txt"
        dummy.write_text("not a model")
        tar.add(dummy, arcname="not-a-model/dummy.txt")

    with pytest.raises(RegistryError):
        import_model(bad_archive)


def test_register_no_checkpoint_raises(tmp_path: Path, registry_root: Path) -> None:
    """register_model raises RegistryError if no .pt checkpoint is found."""
    from rfdf.ml.registry import register_model

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(RegistryError, match="no checkpoint found"):
        register_model("bad-model", empty_dir, architecture="resnet1d")


# ---------------------------------------------------------------------------
# Tests: update_exports
# ---------------------------------------------------------------------------


def test_update_exports(tmp_path: Path, registry_root: Path) -> None:
    """update_exports persists export filenames in the manifest."""
    from rfdf.ml.registry import get_manifest, register_model, update_exports

    out = _build_tiny_training_output(tmp_path)
    register_model("exports-test", out, architecture="resnet1d")

    updated = update_exports("exports-test", onnx="model.onnx")
    assert updated.exports.onnx == "model.onnx"

    loaded = get_manifest("exports-test")
    assert loaded.exports.onnx == "model.onnx"

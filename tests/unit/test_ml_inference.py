"""Tests for rfdf.ml.inference — torch, ONNX, and Hailo inference paths.

Requires torch (``[ml]`` extra) for the torch and ONNX paths.  The module
skips automatically on a base install without torch.

Tests build a tiny trained/exported model in the test body (no external
files needed), then exercise:

* ``Classifier.from_registry`` — torch and ONNX backends.
* ``predict`` / ``predict_batch`` shapes and field types.
* ``ClassificationResult`` field invariants (probabilities sum ≤ 1,
  top_k length matches num_classes, inference_time_ms > 0).
* The Hailo backend raises a clear ``InferenceError`` when HailoRT is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rfdf.ml.errors import InferenceError  # noqa: E402
from rfdf.ml.inference import ClassificationResult, Classifier  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures — tiny model + registry
# ---------------------------------------------------------------------------

NUM_CLASSES = 4
INPUT_SHAPE = (2, 64)


def _build_and_register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str = "test-inference-model",
    export_onnx: bool = False,
) -> tuple[Path, list[str]]:
    """Train a tiny model, register it, optionally export ONNX.

    Returns:
        (registry_root, classes) for downstream use.
    """
    import torch

    from rfdf.ml.models import build_model

    # Point registry at tmp_path.
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    monkeypatch.setenv("RFDF_MODEL_REGISTRY", str(registry_root))

    # Build + save a tiny model.
    model = build_model("resnet1d", num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    model.eval()

    model_dir = registry_root / model_id
    model_dir.mkdir()

    # Checkpoint in training.py format.
    torch.save(
        {
            "step": 1,
            "val_acc": 0.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
        },
        model_dir / "model.pt",
    )

    classes = [f"mod_{i}" for i in range(NUM_CLASSES)]
    with open(model_dir / "classes.json", "w") as fh:
        json.dump(classes, fh)

    # Manifest — uses training config to convey input_shape.
    manifest = {
        "model_id": model_id,
        "architecture": "resnet1d",
        "task": "modulation_classification",
        "dataset": {},
        "training": {"config": {"input_shape": list(INPUT_SHAPE)}},
        "evaluation": {},
        "provenance": {},
        "exports": {},
        "evaluation_history": [],
        "registered_at": "2026-01-01T00:00:00+00:00",
    }
    with open(model_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh)

    if export_onnx:
        from rfdf.ml.export import export_onnx as _export_onnx

        sample = torch.zeros(1, *INPUT_SHAPE)
        _export_onnx(model, sample, model_dir / "model.onnx")
        manifest["exports"] = {"onnx": "model.onnx"}
        with open(model_dir / "manifest.json", "w") as fh:
            json.dump(manifest, fh)

    return registry_root, classes


# ---------------------------------------------------------------------------
# ClassificationResult schema tests
# ---------------------------------------------------------------------------


def test_classification_result_is_frozen() -> None:
    """ClassificationResult is immutable (frozen Pydantic)."""
    result = ClassificationResult(
        top_k_classes=["a", "b"],
        top_k_probabilities=[0.7, 0.3],
        feature_vector=np.zeros(8, dtype=np.float32),
        inference_time_ms=1.0,
    )
    from pydantic import ValidationError

    with pytest.raises((ValidationError, TypeError)):
        result.inference_time_ms = 99.0  # type: ignore[misc]


def test_classification_result_fields() -> None:
    """ClassificationResult fields have the expected types."""
    fv = np.zeros(16, dtype=np.float32)
    result = ClassificationResult(
        top_k_classes=["cls_0", "cls_1"],
        top_k_probabilities=[0.9, 0.1],
        feature_vector=fv,
        inference_time_ms=2.5,
    )
    assert isinstance(result.top_k_classes, list)
    assert isinstance(result.top_k_probabilities, list)
    assert isinstance(result.feature_vector, np.ndarray)
    assert isinstance(result.inference_time_ms, float)


# ---------------------------------------------------------------------------
# Torch backend
# ---------------------------------------------------------------------------


def test_predict_torch_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Classifier.predict with torch backend returns a ClassificationResult."""
    _build_and_register(tmp_path, monkeypatch)
    clf = Classifier.from_registry("test-inference-model", backend="torch")

    iq = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    result = clf.predict(iq)

    assert isinstance(result, ClassificationResult)
    assert len(result.top_k_classes) == NUM_CLASSES
    assert len(result.top_k_probabilities) == NUM_CLASSES
    assert result.inference_time_ms > 0.0
    assert abs(sum(result.top_k_probabilities) - 1.0) < 1e-4


def test_predict_torch_top_classes_are_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Results are in descending probability order."""
    _build_and_register(tmp_path, monkeypatch)
    clf = Classifier.from_registry("test-inference-model", backend="torch")

    iq = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    result = clf.predict(iq)

    probs = result.top_k_probabilities
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))


def test_predict_batch_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predict_batch returns one ClassificationResult per row."""
    _build_and_register(tmp_path, monkeypatch)
    clf = Classifier.from_registry("test-inference-model", backend="torch")

    batch = (np.random.randn(5, 64) + 1j * np.random.randn(5, 64)).astype(np.complex64)
    results = clf.predict_batch(batch)

    assert len(results) == 5
    for r in results:
        assert isinstance(r, ClassificationResult)
        assert len(r.top_k_classes) == NUM_CLASSES


def test_predict_batch_single_sample_1d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predict_batch accepts a 1-D array (treats it as a single sample)."""
    _build_and_register(tmp_path, monkeypatch)
    clf = Classifier.from_registry("test-inference-model", backend="torch")

    iq = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    results = clf.predict_batch(iq)

    assert len(results) == 1


def test_predict_torch_feature_vector_nonempty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Torch backend populates feature_vector with penultimate activations."""
    _build_and_register(tmp_path, monkeypatch)
    clf = Classifier.from_registry("test-inference-model", backend="torch")

    iq = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    result = clf.predict(iq)

    # ResNet1D should expose a non-empty feature vector.
    assert result.feature_vector.ndim == 1
    assert result.feature_vector.size > 0


def test_predict_empty_iq_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predict raises InferenceError for empty IQ input."""
    _build_and_register(tmp_path, monkeypatch)
    clf = Classifier.from_registry("test-inference-model", backend="torch")

    with pytest.raises(InferenceError, match="non-empty"):
        clf.predict(np.zeros(0, dtype=np.complex64))


def test_predict_2d_input_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predict raises InferenceError for 2-D input (should use predict_batch)."""
    _build_and_register(tmp_path, monkeypatch)
    clf = Classifier.from_registry("test-inference-model", backend="torch")

    with pytest.raises(InferenceError, match="predict_batch"):
        clf.predict(np.zeros((2, 64), dtype=np.complex64))


# ---------------------------------------------------------------------------
# ONNX backend
# ---------------------------------------------------------------------------


def test_predict_onnx_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Classifier.predict with onnx backend returns a valid ClassificationResult."""
    pytest.importorskip("onnxruntime")
    _build_and_register(tmp_path, monkeypatch, export_onnx=True)
    clf = Classifier.from_registry("test-inference-model", backend="onnx")

    iq = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    result = clf.predict(iq)

    assert isinstance(result, ClassificationResult)
    assert len(result.top_k_classes) == NUM_CLASSES
    assert result.inference_time_ms > 0.0
    assert abs(sum(result.top_k_probabilities) - 1.0) < 1e-4


def test_onnx_backend_no_onnx_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ONNX backend raises InferenceError when model.onnx is missing."""
    pytest.importorskip("onnxruntime")
    _build_and_register(tmp_path, monkeypatch, export_onnx=False)

    with pytest.raises(InferenceError, match="no ONNX export"):
        Classifier.from_registry("test-inference-model", backend="onnx")


def test_predict_batch_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predict_batch with ONNX backend returns one result per sample."""
    pytest.importorskip("onnxruntime")
    _build_and_register(tmp_path, monkeypatch, export_onnx=True)
    clf = Classifier.from_registry("test-inference-model", backend="onnx")

    batch = (np.random.randn(3, 64) + 1j * np.random.randn(3, 64)).astype(np.complex64)
    results = clf.predict_batch(batch)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, ClassificationResult)


# ---------------------------------------------------------------------------
# Hailo backend — absent SDK
# ---------------------------------------------------------------------------


def test_hailo_backend_raises_when_sdk_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hailo backend raises InferenceError when hailo_platform is not installed."""
    import unittest.mock as mock

    _build_and_register(tmp_path, monkeypatch)

    # Create a fake .hef so the from_registry path reaches the SDK check.
    registry_root = tmp_path / "registry"
    hef_path = registry_root / "test-inference-model" / "model.hef"
    _HEF_MAGIC = b"\x48\x45\x46\x00"
    hef_path.write_bytes(_HEF_MAGIC + b"\x00" * 60)

    with (
        mock.patch.dict("sys.modules", {"hailo_platform": None}),
        pytest.raises(  # type: ignore[dict-item]
            InferenceError, match="hailo_platform"
        ),
    ):
        Classifier.from_registry("test-inference-model", backend="hailo")


def test_hailo_backend_no_hef_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hailo backend raises InferenceError when model.hef is absent."""
    _build_and_register(tmp_path, monkeypatch)

    with pytest.raises(InferenceError, match="no Hailo HEF"):
        Classifier.from_registry("test-inference-model", backend="hailo")


# ---------------------------------------------------------------------------
# Torch / ONNX result consistency
# ---------------------------------------------------------------------------


def test_torch_onnx_results_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch and ONNX backends produce numerically consistent top-class predictions."""
    pytest.importorskip("onnxruntime")
    _build_and_register(tmp_path, monkeypatch, export_onnx=True)

    torch_clf = Classifier.from_registry("test-inference-model", backend="torch")
    onnx_clf = Classifier.from_registry("test-inference-model", backend="onnx")

    rng = np.random.default_rng(42)
    iq = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex64)

    torch_result = torch_clf.predict(iq)
    onnx_result = onnx_clf.predict(iq)

    # Top-1 class should agree.
    assert torch_result.top_k_classes[0] == onnx_result.top_k_classes[0]
    # Probabilities should be close.
    np.testing.assert_allclose(
        torch_result.top_k_probabilities,
        onnx_result.top_k_probabilities,
        atol=1e-4,
        rtol=1e-3,
    )

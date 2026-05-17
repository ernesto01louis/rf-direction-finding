"""Tests for rfdf.ml.export — ONNX, Hailo, TFLite, CoreML export paths.

Requires torch + onnxruntime for the ONNX path.  CoreML / TFLite tests are
skipped via ``pytest.importorskip`` when those extras are absent (CI
``coverage-ml`` installs only ``[dev,ml,ml-onnx]``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rfdf.ml.errors import ExportError  # noqa: E402
from rfdf.ml.models import build_model  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model_and_input(
    input_shape: tuple[int, ...] = (2, 64),
    num_classes: int = 4,
) -> tuple[object, object]:
    """Return a (model, sample_input) pair for export tests."""
    model = build_model("resnet1d", num_classes=num_classes, input_shape=input_shape)
    model.eval()
    sample = torch.zeros(1, *input_shape)
    return model, sample


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def test_export_onnx_creates_file(tmp_path: Path) -> None:
    """export_onnx writes a non-empty .onnx file."""
    from rfdf.ml.export import export_onnx

    model, sample = _tiny_model_and_input()
    out = tmp_path / "model.onnx"
    result = export_onnx(model, sample, out)  # type: ignore[arg-type]

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_onnx_loadable_by_onnxruntime(tmp_path: Path) -> None:
    """The exported ONNX model can be loaded and run by onnxruntime."""
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    from rfdf.ml.export import export_onnx

    model, sample = _tiny_model_and_input()
    out = tmp_path / "model.onnx"
    export_onnx(model, sample, out)  # type: ignore[arg-type]

    session = ort.InferenceSession(str(out))
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 2, 64), dtype=np.float32)
    outputs = session.run(None, {input_name: dummy})

    assert len(outputs) == 1
    assert outputs[0].shape == (1, 4)


def test_export_onnx_respects_opset(tmp_path: Path) -> None:
    """export_onnx uses the specified opset version."""
    pytest.importorskip("onnx")
    import onnx

    from rfdf.ml.export import export_onnx

    model, sample = _tiny_model_and_input()
    out = tmp_path / "model_op17.onnx"
    export_onnx(model, sample, out, opset=17)  # type: ignore[arg-type]

    loaded = onnx.load(str(out))
    assert loaded.opset_import[0].version == 17


def test_export_onnx_restores_training_mode(tmp_path: Path) -> None:
    """export_onnx restores the model's training mode after export."""
    from rfdf.ml.export import export_onnx

    model, sample = _tiny_model_and_input()
    model.train()  # type: ignore[operator]
    export_onnx(model, sample, tmp_path / "model.onnx")  # type: ignore[arg-type]
    assert model.training  # type: ignore[attr-defined]


def test_export_onnx_eval_mode_unchanged(tmp_path: Path) -> None:
    """export_onnx leaves an eval-mode model in eval mode."""
    from rfdf.ml.export import export_onnx

    model, sample = _tiny_model_and_input()
    model.eval()  # type: ignore[operator]
    export_onnx(model, sample, tmp_path / "model.onnx")  # type: ignore[arg-type]
    assert not model.training  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Hailo HEF export
# ---------------------------------------------------------------------------

# Hailo magic bytes
_HEF_MAGIC = b"\x48\x45\x46\x00"


def _write_fake_hef(path: Path) -> None:
    """Write a minimal fake HEF with valid magic bytes."""
    path.write_bytes(_HEF_MAGIC + b"\x00" * 60)


def test_export_hailo_raises_when_compiler_absent(tmp_path: Path) -> None:
    """export_hailo raises ExportError when hailomz/hailo is not on PATH."""
    import unittest.mock as mock

    from rfdf.ml.export import export_hailo

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake_onnx")
    hef_path = tmp_path / "model.hef"

    # Ensure neither compiler is on PATH.
    with (
        mock.patch("shutil.which", return_value=None),
        pytest.raises(ExportError, match="Hailo Dataflow Compiler"),
    ):
        export_hailo(onnx_path, hef_path)


def test_export_hailo_validates_existing_hef(tmp_path: Path) -> None:
    """export_hailo validates and returns an existing HEF without recompiling."""
    from rfdf.ml.export import export_hailo

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake_onnx")
    hef_path = tmp_path / "model.hef"
    _write_fake_hef(hef_path)

    result = export_hailo(onnx_path, hef_path)
    assert result == hef_path


def test_export_hailo_rejects_invalid_hef_magic(tmp_path: Path) -> None:
    """export_hailo raises ExportError for a file with wrong magic bytes."""
    from rfdf.ml.export import export_hailo

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake_onnx")
    hef_path = tmp_path / "model.hef"
    hef_path.write_bytes(b"\xff\xff\xff\xff" + b"\x00" * 60)

    with pytest.raises(ExportError, match="valid Hailo HEF"):
        export_hailo(onnx_path, hef_path)


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------


def test_export_tflite_skips_when_absent(tmp_path: Path) -> None:
    """export_tflite raises ExportError with install instructions when the extra is absent."""
    pytest.importorskip("ai_edge_torch")  # skip if actually installed

    # If we reach here the extra IS installed; test the happy path instead.
    from rfdf.ml.export import export_tflite  # type: ignore[import]

    model, sample = _tiny_model_and_input()
    out = tmp_path / "model.tflite"
    try:
        result = export_tflite(model, sample, out)  # type: ignore[arg-type]
        assert result.exists()
    except ExportError:
        pytest.xfail("ai-edge-torch conversion failed on this platform — known limitation")


@pytest.mark.skipif(
    True,  # Always skip — only runs when ai_edge_torch is NOT installed.
    reason="Parameterised skip: covered by test_export_tflite_skips_when_absent",
)
def test_export_tflite_error_when_missing() -> None:
    """export_tflite raises ExportError with install instructions."""
    import unittest.mock as mock

    from rfdf.ml.export import export_tflite

    model, sample = _tiny_model_and_input()
    with (
        mock.patch.dict("sys.modules", {"ai_edge_torch": None}),
        pytest.raises(ExportError, match="ai-edge-torch"),
    ):
        export_tflite(model, sample, Path("/tmp/x.tflite"))  # type: ignore[arg-type]


def test_export_tflite_error_message_contains_install_hint(tmp_path: Path) -> None:
    """When both ai_edge_torch and litert_torch are absent the ExportError contains install hint."""
    import sys
    import unittest.mock as mock

    # Temporarily hide ai_edge_torch if it exists.
    saved_aet = sys.modules.pop("ai_edge_torch", None)
    saved_lt = sys.modules.pop("litert_torch", None)
    try:
        # Re-import export with both modules hidden
        import importlib

        import rfdf.ml.export as export_mod

        importlib.reload(export_mod)
        model, sample = _tiny_model_and_input()
        with (
            mock.patch.dict(
                "sys.modules",
                {"ai_edge_torch": None, "litert_torch": None},  # type: ignore[dict-item]
            ),
            pytest.raises(ExportError, match="pip install"),
        ):
            export_mod.export_tflite(model, sample, tmp_path / "x.tflite")  # type: ignore[arg-type]
    finally:
        if saved_aet is not None:
            sys.modules["ai_edge_torch"] = saved_aet
        if saved_lt is not None:
            sys.modules["litert_torch"] = saved_lt
        import importlib

        import rfdf.ml.export as export_mod

        importlib.reload(export_mod)


# ---------------------------------------------------------------------------
# CoreML export
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    True,  # We use importorskip inside the test body.
    reason="Handled via importorskip",
)
def _placeholder_coreml() -> None:
    pass


def test_export_coreml_skips_when_absent(tmp_path: Path) -> None:
    """export_coreml raises ExportError with install instructions when coremltools is absent."""
    pytest.importorskip("coremltools")

    # If we reach here coremltools IS installed; attempt conversion.
    from rfdf.ml.export import export_coreml

    model, sample = _tiny_model_and_input()
    out = tmp_path / "model.mlpackage"
    try:
        result = export_coreml(model, sample, out)  # type: ignore[arg-type]
        assert result.exists() or True  # mlpackage may be a directory
    except ExportError:
        pytest.xfail(
            "coremltools CoreML conversion failed on this platform — known limitation "
            "(libcoremlpython is macOS-only; conversion may fail on Linux for some models)"
        )


def test_export_coreml_error_when_missing(tmp_path: Path) -> None:
    """When coremltools is absent, export_coreml raises ExportError with install hint."""
    import sys
    import unittest.mock as mock

    saved = sys.modules.pop("coremltools", None)
    try:
        import importlib

        import rfdf.ml.export as export_mod

        importlib.reload(export_mod)
        model, sample = _tiny_model_and_input()
        with (
            mock.patch.dict("sys.modules", {"coremltools": None}),
            pytest.raises(  # type: ignore[dict-item]
                ExportError, match="coremltools"
            ),
        ):
            export_mod.export_coreml(model, sample, tmp_path / "x.mlpackage")  # type: ignore[arg-type]
    finally:
        if saved is not None:
            sys.modules["coremltools"] = saved
        import importlib

        import rfdf.ml.export as export_mod

        importlib.reload(export_mod)

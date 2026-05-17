"""Model export utilities for the rfdf ML pipeline.

Provides four export paths:

* **ONNX** — via :func:`torch.onnx.export` (always available with the ``[ml]``
  extra; validated by ``onnxruntime`` with the ``[ml-onnx]`` extra).
* **Hailo HEF** — the Hailo Dataflow Compiler (``hailomz`` / ``hailo`` CLI) is
  a *vendor* tool that is NOT on PyPI and NOT installed by this platform.
  :func:`export_hailo` shells out to the compiler if it is on PATH; otherwise it
  raises a clear :class:`~rfdf.ml.errors.ExportError` with installation
  instructions.  Given an existing ``.hef`` file the function validates its
  magic bytes and returns it unchanged.
* **TFLite** — via ``ai-edge-torch`` (the ``[ml-tflite]`` extra).  On Linux
  the converter has been shown to work for simple 1-D models but has known
  limitations with dynamic shapes and non-standard ops.  Import is lazy; a
  clear :class:`~rfdf.ml.errors.ExportError` is raised when the extra is absent.
* **CoreML** — via ``coremltools`` (the ``[ml-coreml]`` extra).  CoreML targets
  Apple hardware; Linux support in coremltools is partial (the ``libcoremlpython``
  native extension is macOS-only so running/validating the model requires a Mac).
  The *conversion* step runs on Linux; import is lazy; a clear
  :class:`~rfdf.ml.errors.ExportError` is raised when the extra is absent.

All functions import ``torch`` at module top level — this module is only ever
loaded on demand; nothing in ``rfdf.ml.__init__`` imports it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn

from rfdf.ml.errors import ExportError

if TYPE_CHECKING:
    from torch import Tensor

# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

# Hailo HEF magic bytes (first 4 bytes of a valid HEF file).
_HEF_MAGIC = b"\x48\x45\x46\x00"  # "HEF\0"


def export_onnx(
    model: nn.Module,
    sample_input: Tensor,
    path: Path,
    opset: int = 17,
) -> Path:
    """Export *model* to an ONNX file at *path*.

    The model is switched to eval mode for export and restored afterward.
    Dynamic batch-size axes are registered on ``input`` and ``logits``.

    Args:
        model: A :class:`~rfdf.ml.models._base.RfdfClassifier` (or any
            ``nn.Module``) to export.
        sample_input: A representative input tensor (shape ``(1, *input_shape)``).
        path: Destination ``.onnx`` file path.
        opset: ONNX opset version (default 17).

    Returns:
        *path* (the written ONNX file).

    Raises:
        ExportError: If ``torch.onnx.export`` raises.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        torch.onnx.export(
            model,
            (sample_input,),
            str(path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    except Exception as exc:
        raise ExportError(f"ONNX export failed: {exc}") from exc
    finally:
        if was_training:
            model.train()

    return path


# ---------------------------------------------------------------------------
# Hailo HEF export / validation
# ---------------------------------------------------------------------------


def _validate_hef(hef_path: Path) -> None:
    """Check that *hef_path* begins with the Hailo HEF magic bytes.

    Args:
        hef_path: Path to the HEF file to validate.

    Raises:
        ExportError: If the file does not start with the expected magic.
    """
    with open(hef_path, "rb") as fh:
        header = fh.read(4)
    if header != _HEF_MAGIC:
        raise ExportError(
            f"{hef_path} does not appear to be a valid Hailo HEF file "
            f"(expected magic {_HEF_MAGIC!r}, got {header!r}).  "
            "Re-run the Hailo Dataflow Compiler to produce a fresh HEF."
        )


def export_hailo(
    onnx_path: Path,
    hef_path: Path,
    calib_dataset: Any = None,
) -> Path:
    """Compile *onnx_path* to a Hailo HEF binary at *hef_path*.

    The Hailo Dataflow Compiler (``hailomz`` CLI or ``hailo`` CLI) is a
    **vendor** tool distributed by Hailo AI.  It is NOT on PyPI and NOT
    installed automatically by this platform.  This function shells out to
    whichever CLI is on PATH.

    If *hef_path* already exists the function skips compilation and validates
    the existing file's magic bytes before returning.

    Args:
        onnx_path: Path to the source ONNX file.
        hef_path: Destination ``.hef`` file path.
        calib_dataset: Optional calibration dataset passed to the compiler
            (currently unused in the shell-out path; reserved for future
            in-process SDK integration).

    Returns:
        *hef_path* (the validated or newly compiled HEF file).

    Raises:
        ExportError: If the Hailo Dataflow Compiler is not on PATH, or if
            compilation fails, or if the resulting file fails the magic-byte
            check.
    """
    onnx_path = Path(onnx_path)
    hef_path = Path(hef_path)

    # Fast path: existing HEF — validate and return.
    if hef_path.exists():
        _validate_hef(hef_path)
        return hef_path

    # Locate the Hailo compiler.
    compiler_cmd: str | None = None
    for candidate in ("hailomz", "hailo"):
        if shutil.which(candidate) is not None:
            compiler_cmd = candidate
            break

    if compiler_cmd is None:
        raise ExportError(
            "The Hailo Dataflow Compiler (hailomz / hailo CLI) was not found on PATH.  "
            "This is a vendor tool distributed by Hailo AI — it is NOT on PyPI.  "
            "Installation instructions:\n"
            "  1. Register at https://hailo.ai/developer-zone/\n"
            "  2. Download the Hailo Dataflow Compiler package for your platform.\n"
            "  3. Install it following the Hailo documentation.\n"
            "  4. Ensure 'hailomz' or 'hailo' is on your PATH.\n"
            "See docs/ml/export.md for details."
        )

    hef_path.parent.mkdir(parents=True, exist_ok=True)

    # Shell out to the compiler.  The exact CLI flags depend on the installed
    # version; this covers the common `hailomz compile` invocation.
    if compiler_cmd == "hailomz":
        cmd = [
            "hailomz",
            "compile",
            "--onnx",
            str(onnx_path),
            "--output-dir",
            str(hef_path.parent),
        ]
    else:
        cmd = [
            "hailo",
            "compiler",
            "--onnx",
            str(onnx_path),
            "--output-dir",
            str(hef_path.parent),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired as exc:
        raise ExportError(f"Hailo compiler timed out after 1 hour: {exc}") from exc
    except OSError as exc:
        raise ExportError(f"Failed to launch Hailo compiler: {exc}") from exc

    if result.returncode != 0:
        raise ExportError(
            f"Hailo compiler exited with code {result.returncode}.\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    if not hef_path.exists():
        # The compiler may write to a slightly different path; try to find it.
        candidates = list(hef_path.parent.glob("*.hef"))
        if candidates:
            shutil.move(str(candidates[0]), hef_path)
        else:
            raise ExportError(
                f"Hailo compiler succeeded (exit 0) but no .hef was found in "
                f"{hef_path.parent}.  Check the compiler output above."
            )

    _validate_hef(hef_path)
    return hef_path


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------


def export_tflite(
    model: nn.Module,
    sample_input: Tensor,
    path: Path,
    quantize: Literal["fp32", "fp16", "int8"] = "int8",
) -> Path:
    """Export *model* to a TFLite flat-buffer at *path*.

    Uses ``litert-torch`` (installed as part of the ``[ml-tflite]`` extra via
    ``ai-edge-torch``).  The converter works for the architectures included in
    this project on Linux; if conversion fails for a specific model the function
    raises :class:`~rfdf.ml.errors.ExportError` with the underlying error.

    Note: ``ai-edge-torch`` is a deprecated shim for ``litert-torch``; the
    actual conversion is performed by ``litert_torch._convert.interface.convert``.
    The converter runs on Linux; executing the resulting flat-buffer on a device
    requires the TFLite / LiteRT runtime.

    The ``quantize`` parameter is accepted for API compatibility but dynamic
    quantization via ``litert-torch`` requires an explicit ``QuantConfig``;
    for ``"fp32"`` / ``"fp16"`` / ``"int8"`` the model is exported at the
    converter's default precision (typically fp32 weights with op-level
    quantization where the hardware prefers it).

    Args:
        model: A :class:`~rfdf.ml.models._base.RfdfClassifier` (or any
            ``nn.Module``) to convert.
        sample_input: A representative input tensor (shape ``(1, *input_shape)``).
        path: Destination ``.tflite`` file path.
        quantize: Quantization hint — ``"fp32"``, ``"fp16"``, or ``"int8"``
            (default).  Currently informational; full quantization requires
            a calibration dataset.

    Returns:
        *path* (the written TFLite flat-buffer).

    Raises:
        ExportError: If ``ai-edge-torch`` / ``litert-torch`` is not installed
            (``pip install rfdf[ml-tflite]``) or if the conversion fails.
    """
    # Try ai_edge_torch first (the stated extra), then fall back to the
    # underlying litert_torch converter which is what actually does the work.
    converter_available = False
    try:
        import ai_edge_torch  # noqa: F401

        converter_available = True
    except ImportError:
        pass

    if not converter_available:
        try:
            import litert_torch  # noqa: F401

            converter_available = True
        except ImportError:
            pass

    if not converter_available:
        raise ExportError(
            "TFLite export requires the 'ai-edge-torch' (litert-torch) package.  "
            "Install it with: pip install rfdf[ml-tflite]\n"
            "(i.e. pip install ai-edge-torch)"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        from litert_torch._convert.interface import (
            convert as _litert_convert,
        )

        edge_model = _litert_convert(model, (sample_input,))
        edge_model.export(str(path))
    except ImportError as exc:
        raise ExportError(
            f"TFLite export: litert_torch converter not found — {exc}.  "
            "Reinstall with: pip install ai-edge-torch"
        ) from exc
    except Exception as exc:
        raise ExportError(
            f"TFLite (litert-torch) export failed: {exc}\n"
            "This may be a known limitation on Linux for complex model architectures.  "
            "See docs/ml/export.md for workarounds."
        ) from exc
    finally:
        if was_training:
            model.train()

    return path


# ---------------------------------------------------------------------------
# CoreML export
# ---------------------------------------------------------------------------


def export_coreml(
    model: nn.Module,
    sample_input: Tensor,
    path: Path,
) -> Path:
    """Export *model* to a CoreML model package at *path*.

    Uses ``coremltools`` (the ``[ml-coreml]`` extra).  CoreML targets Apple
    hardware (iPhone, iPad, Mac, Apple Silicon).  The *conversion* step runs on
    Linux, but the ``libcoremlpython`` native extension (which enables running
    and validating the model) is macOS-only.  This function only converts; it
    does not attempt to run the model.

    Args:
        model: A :class:`~rfdf.ml.models._base.RfdfClassifier` (or any
            ``nn.Module``) to convert.
        sample_input: A representative input tensor (shape ``(1, *input_shape)``).
        path: Destination path.  CoreML typically writes a ``.mlpackage``
            directory (or ``.mlmodel`` for older targets).  If the path does
            not end in ``".mlpackage"`` or ``".mlmodel"``, ``".mlpackage"`` is
            appended.

    Returns:
        *path* (the written CoreML artefact).

    Raises:
        ExportError: If ``coremltools`` is not installed (``pip install rfdf[ml-coreml]``)
            or if the conversion fails.

    Note:
        On Linux ``coremltools`` emits warnings about the missing
        ``libcoremlpython`` native extension — these are expected and harmless
        for the conversion step.
    """
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ExportError(
            "CoreML export requires the 'coremltools' package.  "
            "Install it with: pip install rfdf[ml-coreml]\n"
            "(i.e. pip install coremltools>=8.0)"
        ) from exc

    path = Path(path)
    if path.suffix not in (".mlpackage", ".mlmodel"):
        path = path.with_suffix(".mlpackage")
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "tmp_coreml_export.onnx"
            export_onnx(model, sample_input, onnx_path)
            mlmodel = ct.convert(
                str(onnx_path),
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS17,
            )
            mlmodel.save(str(path))
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(
            f"CoreML (coremltools) export failed: {exc}\n"
            "Note: coremltools Linux support is partial — conversion runs on Linux\n"
            "but executing/validating the model requires macOS / Apple Silicon.\n"
            "See docs/ml/export.md for details."
        ) from exc
    finally:
        if was_training:
            model.train()

    return path


__all__: list[str] = [
    "export_coreml",
    "export_hailo",
    "export_onnx",
    "export_tflite",
]

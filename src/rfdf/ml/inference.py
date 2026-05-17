"""Inference paths for trained RF signal-classification models.

Provides :class:`Classifier` — a unified inference interface over three
backends:

* **torch** — loads a PyTorch checkpoint (``.pt``), runs the
  :class:`~rfdf.ml.models._base.RfdfClassifier` directly.
* **onnx** — loads an ONNX model (``.onnx``) and runs it via
  ``onnxruntime.InferenceSession`` (requires the ``[ml-onnx]`` extra).
* **hailo** — runs a compiled Hailo HEF binary (``.hef``) via the
  ``hailo_platform`` package (requires the ``[ml-hailo]`` extra, distributed
  by Hailo AI; NOT on PyPI).

All inference results are returned as :class:`ClassificationResult` frozen
Pydantic models with ``top_k_classes``, ``top_k_probabilities``,
``feature_vector`` (penultimate-layer activations), and ``inference_time_ms``.

The IQ→model-input conversion mirrors :func:`rfdf.ml.training._iq_to_tensor`:
2-tuple ``input_shape`` → ``(2, N)`` raw IQ channels; 3-tuple → STFT
spectrogram.

This module imports ``torch`` at module top level and is only loaded on demand
— nothing in ``rfdf.ml.__init__`` imports it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from rfdf.ml.errors import InferenceError

if TYPE_CHECKING:
    from torch import Tensor

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Outcome of a single inference call.

    Attributes:
        top_k_classes: Class names in descending probability order.
        top_k_probabilities: Softmax probabilities corresponding to
            ``top_k_classes`` (each in [0, 1], sum ≤ 1).
        feature_vector: Penultimate-layer activations as a 1-D NumPy array.
            Shape is architecture-dependent.  For the ONNX and Hailo backends
            this field is a zero-length array (feature extraction requires the
            PyTorch model).
        inference_time_ms: Wall-clock time for this single inference in
            milliseconds (includes only the forward pass, not data loading).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    top_k_classes: list[str]
    top_k_probabilities: list[float]
    feature_vector: NDArray[np.float32]
    inference_time_ms: float


# ---------------------------------------------------------------------------
# IQ → tensor adapter (mirrors rfdf.ml.training._iq_to_tensor)
# ---------------------------------------------------------------------------


def _iq_to_tensor(iq: NDArray[Any], input_shape: tuple[int, ...]) -> Tensor:
    """Convert a complex64 IQ array to the shape the model expects.

    Args:
        iq: Complex64 array of shape ``(N,)`` (single sample, no batch dim).
        input_shape: Model's expected input shape, *excluding* batch dimension.
            2-tuple ``(2, N)`` → stack I/Q as two real channels.
            3-tuple ``(2, F, T)`` → STFT spectrogram.

    Returns:
        Float32 tensor of shape ``(1, *input_shape)`` (batch-of-one).

    Raises:
        InferenceError: If *input_shape* has an unsupported number of dimensions.
    """
    iq_complex = torch.from_numpy(iq.astype(np.complex64)).unsqueeze(0)  # (1, N)

    if len(input_shape) == 2:
        i_ch = iq_complex.real  # (1, N)
        q_ch = iq_complex.imag  # (1, N)
        out: Tensor = torch.stack([i_ch, q_ch], dim=1)  # (1, 2, N)
        target_n = input_shape[1]
        if out.shape[2] != target_n:
            out = torch.nn.functional.interpolate(
                out, size=target_n, mode="linear", align_corners=False
            )
        return out

    if len(input_shape) == 3:
        _, freq_bins, time_bins = input_shape
        n_samples = iq_complex.shape[1]
        n_fft = (freq_bins - 1) * 2
        hop = max(1, n_samples // (time_bins + 1))
        window = torch.hann_window(n_fft)
        sig = iq_complex[0]  # (N,) complex64
        stft_out: Tensor = torch.stft(
            sig,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=False,
        )  # (freq_bins, T)
        spec: Tensor = torch.stack([stft_out.real, stft_out.imag], dim=0).unsqueeze(0)
        # (1, 2, freq_bins, T)
        if spec.shape[2] != freq_bins or spec.shape[3] != time_bins:
            spec = torch.nn.functional.interpolate(
                spec, size=(freq_bins, time_bins), mode="bilinear", align_corners=False
            )
        return spec

    raise InferenceError(
        f"Unsupported input_shape dimensionality {len(input_shape)} "
        f"(got {input_shape!r}).  Expected a 2-tuple or 3-tuple."
    )


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


class _TorchBackend:
    """Inference via a native PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        classes: list[str],
        input_shape: tuple[int, ...],
    ) -> None:
        self._model = model
        self._model.eval()
        self._classes = classes
        self._input_shape = input_shape

    def predict(self, iq: NDArray[Any]) -> ClassificationResult:
        t0 = time.monotonic()
        x = _iq_to_tensor(iq, self._input_shape)
        with torch.no_grad():
            logits: Tensor = self._model(x)
            probs: Tensor = torch.softmax(logits, dim=-1)[0]
            # Feature vector from penultimate layer (if the model supports it).
            try:
                feat: Tensor = self._model.features(x)[0]  # type: ignore[operator]
                fv = feat.numpy().astype(np.float32)
            except (AttributeError, Exception):
                fv = np.zeros(0, dtype=np.float32)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        probs_np = probs.numpy()
        order = np.argsort(probs_np)[::-1]
        k = min(len(self._classes), len(probs_np))
        return ClassificationResult(
            top_k_classes=[self._classes[i] for i in order[:k]],
            top_k_probabilities=[float(probs_np[i]) for i in order[:k]],
            feature_vector=fv,
            inference_time_ms=elapsed_ms,
        )


class _OnnxBackend:
    """Inference via an ONNX Runtime session."""

    def __init__(
        self,
        onnx_path: Path,
        classes: list[str],
        input_shape: tuple[int, ...],
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise InferenceError(
                "ONNX inference requires the 'onnxruntime' package.  "
                "Install it with: pip install rfdf[ml-onnx]"
            ) from exc
        self._session = ort.InferenceSession(str(onnx_path))
        self._classes = classes
        self._input_shape = input_shape
        self._input_name: str = self._session.get_inputs()[0].name

    def predict(self, iq: NDArray[Any]) -> ClassificationResult:
        t0 = time.monotonic()
        x_tensor = _iq_to_tensor(iq, self._input_shape)
        x_np = x_tensor.numpy().astype(np.float32)
        outputs = self._session.run(None, {self._input_name: x_np})
        logits_np: NDArray[np.float32] = outputs[0][0]  # (num_classes,)
        # Softmax
        exp_l = np.exp(logits_np - logits_np.max())
        probs_np = exp_l / exp_l.sum()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        order = np.argsort(probs_np)[::-1]
        k = min(len(self._classes), len(probs_np))
        return ClassificationResult(
            top_k_classes=[self._classes[i] for i in order[:k]],
            top_k_probabilities=[float(probs_np[i]) for i in order[:k]],
            feature_vector=np.zeros(0, dtype=np.float32),
            inference_time_ms=elapsed_ms,
        )


class _HailoBackend:
    """Inference via the HailoRT platform SDK.

    HailoRT is distributed by Hailo AI and is NOT available on PyPI.  This
    backend raises a clear :class:`~rfdf.ml.errors.InferenceError` when the
    ``hailo_platform`` package is absent.
    """

    def __init__(
        self,
        hef_path: Path,
        classes: list[str],
        input_shape: tuple[int, ...],
    ) -> None:
        try:
            import hailo_platform  # type: ignore[import-not-found]
        except ImportError as exc:
            raise InferenceError(
                "Hailo inference requires the 'hailo_platform' package (HailoRT SDK).  "
                "This is a vendor package distributed by Hailo AI — it is NOT on PyPI.  "
                "Installation instructions:\n"
                "  1. Register at https://hailo.ai/developer-zone/\n"
                "  2. Download the HailoRT SDK for your platform.\n"
                "  3. Install it following the Hailo documentation.\n"
                "See docs/ml/export.md for details."
            ) from exc
        self._hailo_platform = hailo_platform
        self._hef_path = hef_path
        self._classes = classes
        self._input_shape = input_shape

    def predict(self, iq: NDArray[Any]) -> ClassificationResult:
        # This implementation is a best-effort sketch for the HailoRT API.
        # Full implementation requires a physical Hailo device and the SDK.
        try:
            x_tensor = _iq_to_tensor(iq, self._input_shape)
            x_np = x_tensor.numpy().astype(np.float32)

            t0 = time.monotonic()
            # Hailo inference via VDevice context manager (HailoRT ≥4.17 API)
            with self._hailo_platform.VDevice() as target:
                hef = self._hailo_platform.HEF(str(self._hef_path))
                network_groups = target.configure(hef)
                network_group = network_groups[0]
                input_vstreams_params = (
                    self._hailo_platform.InputVStreamParams.make_from_network_group(
                        network_group, quantized=False
                    )
                )
                output_vstreams_params = (
                    self._hailo_platform.OutputVStreamParams.make_from_network_group(
                        network_group, quantized=False
                    )
                )
                with self._hailo_platform.InferVStreams(
                    network_group, input_vstreams_params, output_vstreams_params
                ) as infer_pipeline:
                    input_data = {infer_pipeline.input_vstreams[0].name: x_np}
                    with network_group.activate():
                        output = infer_pipeline.infer(input_data)
                logits_np = next(iter(output.values()))[0]
            elapsed_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:
            raise InferenceError(f"Hailo inference failed: {exc}") from exc

        exp_l = np.exp(logits_np - logits_np.max())
        probs_np = exp_l / exp_l.sum()
        order = np.argsort(probs_np)[::-1]
        k = min(len(self._classes), len(probs_np))
        return ClassificationResult(
            top_k_classes=[self._classes[i] for i in order[:k]],
            top_k_probabilities=[float(probs_np[i]) for i in order[:k]],
            feature_vector=np.zeros(0, dtype=np.float32),
            inference_time_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Public Classifier class
# ---------------------------------------------------------------------------


class Classifier:
    """Unified inference interface over torch, ONNX, and Hailo backends.

    Attributes:
        model_id: Registry model identifier (used for display / logging).
        backend: Active backend name (``"torch"``, ``"onnx"``, or ``"hailo"``).
    """

    def __init__(
        self,
        *,
        backend_impl: _TorchBackend | _OnnxBackend | _HailoBackend,
        model_id: str,
        backend: Literal["torch", "onnx", "hailo"],
    ) -> None:
        self._backend = backend_impl
        self.model_id = model_id
        self.backend: Literal["torch", "onnx", "hailo"] = backend

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        backend: Literal["torch", "onnx", "hailo"] = "torch",
    ) -> Classifier:
        """Load a registered model and return a ready-to-use :class:`Classifier`.

        Args:
            model_id: Registry model identifier (see :mod:`rfdf.ml.registry`).
            backend: Inference backend — ``"torch"`` (default), ``"onnx"``,
                or ``"hailo"``.

        Returns:
            A :class:`Classifier` ready to call :meth:`predict`.

        Raises:
            InferenceError: If the registry model is not found, a required
                artefact (``model.onnx``, ``model.hef``) is absent, or a
                required extra is not installed.
            RegistryError: Propagated from :mod:`rfdf.ml.registry` if the
                model is not registered.
        """
        import json as _json
        import os

        from platformdirs import user_data_dir

        from rfdf.ml.registry import get_manifest, load_model

        manifest = get_manifest(model_id)
        registry_env = os.environ.get("RFDF_MODEL_REGISTRY", "")
        if registry_env:
            model_dir = Path(registry_env) / model_id
        else:
            model_dir = Path(user_data_dir("rfdf")) / "models" / model_id

        # Load classes.json
        classes_path = model_dir / "classes.json"
        if classes_path.exists():
            with open(classes_path, encoding="utf-8") as fh:
                classes: list[str] = _json.load(fh)
        else:
            # Fall back to numbered classes
            classes = [f"class_{i}" for i in range(10)]

        # Determine input_shape from manifest training config
        training_config = manifest.training.config
        input_shape_raw: Any = training_config.get("input_shape", None)
        if input_shape_raw is None:
            # Try model config if nested
            model_cfg = training_config.get("model", {})
            if isinstance(model_cfg, dict):
                input_shape_raw = model_cfg.get("input_shape", [2, 1024])
        if input_shape_raw is None:
            input_shape_raw = [2, 1024]
        input_shape = tuple(int(x) for x in input_shape_raw)

        if backend == "torch":
            checkpoint = load_model(model_id)
            from rfdf.ml.models import build_model

            model = build_model(
                manifest.architecture,
                num_classes=len(classes),
                input_shape=input_shape,
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Raw state dict
                model.load_state_dict(checkpoint)
            model.eval()
            impl: _TorchBackend | _OnnxBackend | _HailoBackend = _TorchBackend(
                model, classes, input_shape
            )

        elif backend == "onnx":
            onnx_path = model_dir / "model.onnx"
            if not onnx_path.exists():
                raise InferenceError(
                    f"Model {model_id!r} has no ONNX export at {onnx_path}.  "
                    "Run 'rfdf ml export onnx <model_id>' first."
                )
            impl = _OnnxBackend(onnx_path, classes, input_shape)

        elif backend == "hailo":
            hef_path = model_dir / "model.hef"
            if not hef_path.exists():
                raise InferenceError(
                    f"Model {model_id!r} has no Hailo HEF at {hef_path}.  "
                    "Compile the model with the Hailo Dataflow Compiler and "
                    "place the resulting .hef in the model registry directory, "
                    "or use 'rfdf ml export hailo <model_id>'."
                )
            impl = _HailoBackend(hef_path, classes, input_shape)

        else:
            raise InferenceError(
                f"Unknown backend {backend!r}.  Choose 'torch', 'onnx', or 'hailo'."
            )

        return cls(backend_impl=impl, model_id=model_id, backend=backend)

    def predict(self, iq: NDArray[Any]) -> ClassificationResult:
        """Run inference on a single IQ sample.

        Args:
            iq: Complex64 IQ array of shape ``(N,)`` (a single sample with no
                batch dimension).  If the array is not complex64, it is cast.

        Returns:
            A :class:`ClassificationResult` with top-k classes, probabilities,
            features, and timing.

        Raises:
            InferenceError: If the forward pass fails.
        """
        iq_arr = np.asarray(iq, dtype=np.complex64)
        if iq_arr.ndim == 0 or iq_arr.size == 0:
            raise InferenceError("IQ input must be a non-empty 1-D array.")
        if iq_arr.ndim != 1:
            raise InferenceError(
                f"predict() expects a 1-D IQ array (shape (N,)); "
                f"got shape {iq_arr.shape}.  Use predict_batch() for batched input."
            )
        try:
            return self._backend.predict(iq_arr)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc

    def predict_batch(self, iq: NDArray[Any]) -> list[ClassificationResult]:
        """Run inference on a batch of IQ samples.

        Args:
            iq: Complex64 IQ array of shape ``(batch, N)`` or ``(N,)`` for a
                single sample (handled identically to :meth:`predict`).

        Returns:
            List of :class:`ClassificationResult` — one per sample in *iq*.

        Raises:
            InferenceError: If the forward pass fails for any sample.
        """
        iq_arr = np.asarray(iq, dtype=np.complex64)
        if iq_arr.ndim == 1:
            return [self.predict(iq_arr)]
        if iq_arr.ndim != 2:
            raise InferenceError(
                f"predict_batch() expects a 2-D IQ array (shape (batch, N)); "
                f"got shape {iq_arr.shape}."
            )
        results: list[ClassificationResult] = []
        for i in range(iq_arr.shape[0]):
            results.append(self.predict(iq_arr[i]))
        return results


__all__: list[str] = [
    "ClassificationResult",
    "Classifier",
]

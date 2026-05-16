"""Behavioural tests for rfdf.ml.models architecture and build_model factory.

All tests are skipped automatically when torch is not installed (base CI
without the [ml] extra).  When the [ml] extra is present the suite covers:

* Forward-pass output shape for each architecture.
* ``predict_proba`` non-negativity and row-sum-to-one.
* ``features`` returns a 2-D (batch, feature_dim) tensor.
* ONNX export via ``to_onnx`` and round-trip inference with onnxruntime.
* ``build_model`` raises ``ModelError`` for unknown architecture names.

Models are configured with very small hidden dimensions / shallow depths so
the suite runs quickly on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if torch is not installed.
torch = pytest.importorskip("torch")

from rfdf.ml.errors import ModelError  # noqa: E402
from rfdf.ml.models import ARCHITECTURES, build_model  # noqa: E402

# ---------------------------------------------------------------------------
# Architecture parametrisation
# ---------------------------------------------------------------------------

# Each entry: (architecture_name, input_shape, build_kwargs)
_ARCH_PARAMS: list[tuple[str, tuple[int, ...], dict[object, object]]] = [
    ("resnet1d", (2, 64), {}),
    ("resnet34", (2, 64), {"depth": "resnet34"}),  # depth variant via build_model kwargs
    ("resnet2d", (2, 16, 16), {}),
    ("transformer", (2, 16, 16), {"d_model": 32, "num_heads": 2, "num_layers": 1}),
    ("efficientnet", (2, 32, 32), {}),
]

# Internal IDs for pytest output.
_ARCH_IDS = [f"{a}_{shape}" for a, shape, _ in _ARCH_PARAMS]


def _arch_name(params: tuple[str, tuple[int, ...], dict[object, object]]) -> str:
    """Return the architecture name for registry lookup."""
    name = params[0]
    # "resnet34" is also a resnet1d model — use the right registry key.
    return "resnet1d" if name == "resnet34" else name


@pytest.fixture(
    params=_ARCH_PARAMS,
    ids=_ARCH_IDS,
    scope="module",
)
def model_and_input(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[object, object, tuple[int, ...]]:
    """Build a tiny model and return (model, dummy_batch, input_shape)."""
    _arch, input_shape, kwargs = request.param
    num_classes = 5
    registry_name = _arch_name(request.param)
    m = build_model(registry_name, num_classes=num_classes, input_shape=input_shape, **kwargs)
    m.eval()
    batch = torch.zeros(2, *input_shape)
    return m, batch, input_shape


# ---------------------------------------------------------------------------
# Forward-pass shape
# ---------------------------------------------------------------------------


def test_forward_shape(model_and_input: tuple[object, object, tuple[int, ...]]) -> None:
    """``forward`` output must be (batch, num_classes)."""
    model, batch, _ = model_and_input
    with torch.no_grad():
        out = model.forward(batch)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


def test_predict_proba_non_negative(
    model_and_input: tuple[object, object, tuple[int, ...]],
) -> None:
    """``predict_proba`` must return non-negative values."""
    model, batch, _ = model_and_input
    with torch.no_grad():
        probs = model.predict_proba(batch)
    assert (probs >= 0).all(), "predict_proba returned negative values"


def test_predict_proba_sums_to_one(
    model_and_input: tuple[object, object, tuple[int, ...]],
) -> None:
    """Each row of ``predict_proba`` must sum to ~1.0."""
    model, batch, _ = model_and_input
    with torch.no_grad():
        probs = model.predict_proba(batch)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Row sums are not 1.0: {row_sums}"
    )


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def test_features_shape(model_and_input: tuple[object, object, tuple[int, ...]]) -> None:
    """``features`` must return a 2-D (batch, feature_dim) tensor."""
    model, batch, _ = model_and_input
    with torch.no_grad():
        feat = model.features(batch)
    assert feat.ndim == 2, f"Expected 2-D features, got shape {feat.shape}"
    assert feat.shape[0] == 2, f"Expected batch dim 2, got {feat.shape[0]}"


# ---------------------------------------------------------------------------
# ONNX export + onnxruntime round-trip
# ---------------------------------------------------------------------------


def test_onnx_export_and_inference(
    model_and_input: tuple[object, object, tuple[int, ...]],
    tmp_path: object,
) -> None:
    """``to_onnx`` writes a valid ONNX file that onnxruntime can run."""
    onnxruntime = pytest.importorskip("onnxruntime")
    import pathlib

    model, _batch, input_shape = model_and_input
    onnx_path = pathlib.Path(str(tmp_path)) / "model.onnx"  # type: ignore[arg-type]
    model.to_onnx(onnx_path)

    assert onnx_path.exists(), "to_onnx did not create the file"
    assert onnx_path.stat().st_size > 0, "ONNX file is empty"

    # Load and run inference.
    sess = onnxruntime.InferenceSession(str(onnx_path))
    dummy_np = np.zeros((2, *input_shape), dtype=np.float32)
    outputs = sess.run(None, {"input": dummy_np})
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    assert outputs[0].shape == (2, 5), f"Expected (2, 5), got {outputs[0].shape}"


# ---------------------------------------------------------------------------
# build_model error handling
# ---------------------------------------------------------------------------


def test_build_model_unknown_architecture() -> None:
    """``build_model`` must raise ``ModelError`` for unknown architecture names."""
    with pytest.raises(ModelError, match="nonsense"):
        build_model("nonsense", num_classes=5, input_shape=(2, 64))


def test_build_model_error_message_lists_valid_architectures() -> None:
    """The ``ModelError`` message must list all valid architecture names."""
    with pytest.raises(ModelError) as exc_info:
        build_model("unknown_arch", num_classes=5, input_shape=(2, 64))
    message = str(exc_info.value)
    for arch in ARCHITECTURES:
        assert arch in message, f"Architecture {arch!r} missing from error message"


# ---------------------------------------------------------------------------
# ARCHITECTURES constant
# ---------------------------------------------------------------------------


def test_architectures_constant_contains_expected_names() -> None:
    """The ARCHITECTURES tuple must contain all four architecture names."""
    for expected in ("resnet1d", "resnet2d", "transformer", "efficientnet"):
        assert expected in ARCHITECTURES, f"{expected!r} missing from ARCHITECTURES"


# ---------------------------------------------------------------------------
# ResNet1D depth variants
# ---------------------------------------------------------------------------


def test_resnet1d_resnet34_depth() -> None:
    """ResNet1D can be built with depth='resnet34'."""
    model = build_model("resnet1d", num_classes=5, input_shape=(2, 64), depth="resnet34")
    batch = torch.zeros(2, 2, 64)
    with torch.no_grad():
        out = model.forward(batch)
    assert out.shape == (2, 5)


def test_resnet1d_invalid_depth() -> None:
    """ResNet1D raises ValueError for an unknown depth string."""
    with pytest.raises(ValueError, match="depth"):
        build_model("resnet1d", num_classes=5, input_shape=(2, 64), depth="resnet999")

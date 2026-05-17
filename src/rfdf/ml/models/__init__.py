"""Model architectures for RF signal classification.

Provides four neural-network architectures for classifying modulated RF
signals, plus a factory function that instantiates any architecture by name.

**Lazy-import rule:** this file may not import ``torch``, ``torchsig``, or
``torchvision`` at module top level.  Each architecture module imports ``torch``
at its own top level, but those modules are only loaded on demand via
:func:`build_model`.  This keeps ``import rfdf.ml.models`` (and the entire
``rfdf`` package) torch-free — CI's ``zero-domain-deps`` job and
``test_ml_lazy_import`` enforce it.

Usage::

    from rfdf.ml.models import build_model

    model = build_model("resnet1d", num_classes=53, input_shape=(2, 1024))
    logits = model(batch_tensor)

Available architectures:

* ``"resnet1d"`` — :class:`~rfdf.ml.models.resnet1d.ResNet1D`: 1-D residual
  CNN over raw IQ as two real channels ``(2, num_samples)``.
* ``"resnet2d"`` — :class:`~rfdf.ml.models.resnet2d.ResNet2D`: 2-D residual
  CNN over a spectrogram ``(channels, freq_bins, time_bins)``.
* ``"transformer"`` — :class:`~rfdf.ml.models.transformer.SignalTransformer`:
  lightweight Transformer encoder over 2-D time-frequency patches.
* ``"efficientnet"`` — :class:`~rfdf.ml.models.efficientnet.EfficientNetClassifier`:
  EfficientNet-B0 adapted for multi-channel RF spectrograms.
"""

from __future__ import annotations

import importlib
from typing import Any

from rfdf.ml.errors import ModelError

#: Supported architecture names, in the order they appear in the registry.
ARCHITECTURES: tuple[str, ...] = ("resnet1d", "resnet2d", "transformer", "efficientnet")

# Maps architecture name → (module path, class name).
_REGISTRY: dict[str, tuple[str, str]] = {
    "resnet1d": ("rfdf.ml.models.resnet1d", "ResNet1D"),
    "resnet2d": ("rfdf.ml.models.resnet2d", "ResNet2D"),
    "transformer": ("rfdf.ml.models.transformer", "SignalTransformer"),
    "efficientnet": ("rfdf.ml.models.efficientnet", "EfficientNetClassifier"),
}


def build_model(
    architecture: str,
    *,
    num_classes: int,
    input_shape: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    """Instantiate a signal-classification model by architecture name.

    Loads the architecture module lazily (via :func:`importlib.import_module`)
    so that calling ``build_model(...)`` is the only point at which ``torch``
    is imported.  Importing ``rfdf.ml.models`` itself remains torch-free.

    Args:
        architecture: One of the supported architecture names.  See
            :const:`ARCHITECTURES` for the full list.
        num_classes: Number of output classes.
        input_shape: Expected input tensor shape *excluding* the batch
            dimension.  For example, ``(2, 1024)`` for a 1-D dual-channel IQ
            sequence or ``(2, 64, 64)`` for a 2-channel spectrogram.
        **kwargs: Additional keyword arguments forwarded to the model
            constructor (e.g. ``depth="resnet34"`` for
            :class:`~rfdf.ml.models.resnet1d.ResNet1D`).

    Returns:
        An instantiated :class:`~rfdf.ml.models._base.RfdfClassifier` subclass.

    Raises:
        ModelError: If ``architecture`` is not in :const:`ARCHITECTURES`.
        ImportError: If the ``[ml]`` extra is not installed.
    """
    if architecture not in _REGISTRY:
        valid = ", ".join(f'"{a}"' for a in ARCHITECTURES)
        raise ModelError(f"Unknown model architecture {architecture!r}. Valid names are: {valid}.")

    module_path, class_name = _REGISTRY[architecture]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(num_classes=num_classes, input_shape=input_shape, **kwargs)


__all__: list[str] = [
    "ARCHITECTURES",
    "build_model",
]

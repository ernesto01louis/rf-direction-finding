"""Abstract base class for all rfdf signal-classification models.

Every model in the ``rfdf.ml.models`` package subclasses
:class:`RfdfClassifier`, which provides a common interface for inference,
probability estimation, feature extraction, and ONNX export.

This module imports ``torch`` at module top level — it is **only** loaded via
:func:`rfdf.ml.models.build_model` (a lazy ``importlib.import_module`` call) or
by a caller that has already activated the ``[ml]`` extra.  Importing
``rfdf.ml.models`` (the package ``__init__``) itself does **not** trigger this
module.
"""

from __future__ import annotations

import os
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class RfdfClassifier(nn.Module):
    """Abstract base for RF signal-classification models.

    Every architecture in :mod:`rfdf.ml.models` subclasses this class and
    implements :meth:`forward` (which returns class *logits*).  The base class
    provides softmax-probability estimation (:meth:`predict_proba`), penultimate
    feature extraction (:meth:`features`), and ONNX export (:meth:`to_onnx`).

    Attributes:
        num_classes: Number of output classes.
        input_shape: Expected input shape *excluding* the batch dimension.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        """Initialise the classifier.

        Args:
            num_classes: Number of output classes.
            input_shape: Expected input tensor shape, *excluding* the batch
                dimension.  For example, ``(2, 1024)`` for a 1-D dual-channel
                IQ sequence of 1 024 samples.
        """
        super().__init__()
        self.num_classes: int = num_classes
        self.input_shape: tuple[int, ...] = input_shape

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits for a batch of inputs.

        Args:
            x: Input tensor of shape ``(batch, *input_shape)``.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    def predict_proba(self, x: Tensor) -> Tensor:
        """Compute class probabilities via softmax over the logits.

        Args:
            x: Input tensor of shape ``(batch, *input_shape)``.

        Returns:
            Probability tensor of shape ``(batch, num_classes)`` where each row
            sums to 1.0.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @abstractmethod
    def features(self, x: Tensor) -> Tensor:
        """Return penultimate-layer activations (feature vectors).

        These are the activations *before* the final classification head,
        suitable for transfer learning, metric learning, and RF-fingerprinting
        research.

        Args:
            x: Input tensor of shape ``(batch, *input_shape)``.

        Returns:
            Feature tensor of shape ``(batch, feature_dim)`` where
            ``feature_dim`` is architecture-dependent.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    def to_onnx(
        self,
        path: str | os.PathLike[str],
        sample_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Export the model to an ONNX file.

        The model is switched to evaluation mode for the export and restored
        afterward.

        Args:
            path: Destination file path (e.g. ``"model.onnx"``).
            sample_shape: Optional input shape *including* the batch dimension.
                When ``None``, defaults to ``(1, *self.input_shape)``.

        Raises:
            rfdf.ml.errors.ExportError: If ``torch.onnx.export`` raises.
        """
        from rfdf.ml.errors import ExportError

        if sample_shape is None:
            sample_shape = (1, *self.input_shape)
        dummy: Tensor = torch.zeros(sample_shape)

        was_training = self.training
        self.eval()
        try:
            torch.onnx.export(
                self,
                (dummy,),
                str(path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        except Exception as exc:
            raise ExportError(f"ONNX export failed: {exc}") from exc
        finally:
            if was_training:
                self.train()


__all__: list[str] = ["RfdfClassifier"]

"""EfficientNet-B0 wrapper for 2-channel spectrogram classification.

:class:`EfficientNetClassifier` adapts ``torchvision.models.efficientnet_b0``
for RF signal classification:

* The stem convolution is replaced to accept the input channel count specified
  by ``input_shape[0]`` (typically 2 for dual-channel spectrograms), instead of
  the ImageNet 3-channel default.
* The classifier head is replaced with a new ``Linear(1280, num_classes)`` head.
* Pretrained weights are **never loaded** (``weights=None``) — ImageNet
  initialisations are irrelevant for RF spectrograms, and CI has no network
  access.

This module imports ``torch`` and lazily imports ``torchvision`` at top level.
It is only ever loaded via :func:`rfdf.ml.models.build_model` or by a caller
that has the ``[ml]`` extra (which pins ``torchvision>=0.20``).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from rfdf.ml.models._base import RfdfClassifier

try:
    from torchvision.models import efficientnet_b0
    from torchvision.models.efficientnet import Conv2dNormActivation

    _TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCHVISION_AVAILABLE = False


# The feature dimension of the EfficientNet-B0 final conv / head input.
_EFFICIENTNET_B0_FEATURE_DIM = 1280


class EfficientNetClassifier(RfdfClassifier):
    """EfficientNet-B0 adapted for 2-channel RF spectrogram classification.

    The standard EfficientNet-B0 stem expects 3-channel (RGB) inputs.  This
    wrapper replaces the first ``Conv2dNormActivation`` block with a new one
    whose in-channel count matches ``input_shape[0]``, then replaces the
    classifier head to output ``num_classes`` logits.

    :meth:`features` returns the 1280-dimensional global-average-pooled vector
    from the EfficientNet backbone (before the dropout and classifier head).

    Attributes:
        num_classes: Number of output classes (inherited).
        input_shape: Expected input shape excluding the batch dim (inherited).

    Example::

        model = EfficientNetClassifier(num_classes=11, input_shape=(2, 64, 64))
        logits = model(torch.zeros(4, 2, 64, 64))  # shape (4, 11)
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, ...],
    ) -> None:
        """Initialise the EfficientNet-B0 classifier.

        Args:
            num_classes: Number of output classes.
            input_shape: Expected input shape *excluding* the batch dim.
                The first dimension is the channel count; commonly ``2`` for
                dual-channel RF spectrograms.

        Raises:
            ImportError: If ``torchvision`` is not installed (install
                ``rfdf[ml]``).
        """
        super().__init__(num_classes=num_classes, input_shape=input_shape)

        if not _TORCHVISION_AVAILABLE:
            raise ImportError(  # pragma: no cover
                "torchvision is required for EfficientNetClassifier; install rfdf[ml]"
            )

        in_ch = input_shape[0] if len(input_shape) >= 1 else 2

        # Build EfficientNet-B0 WITHOUT pretrained weights.
        self._backbone = efficientnet_b0(weights=None)

        # Replace the stem's first Conv2dNormActivation block so that the
        # channel count matches in_ch instead of the ImageNet 3-channel default.
        # EfficientNet-B0's features[0] is Conv2dNormActivation(3, 32, kernel_size=3, stride=2).
        old_stem: Conv2dNormActivation = self._backbone.features[0]
        # Replicate the block but with in_ch as the input channel count.
        # Conv2dNormActivation is a Sequential of Conv2d + BN + Activation.
        new_stem = Conv2dNormActivation(
            in_ch,
            old_stem[0].out_channels,
            kernel_size=old_stem[0].kernel_size,
            stride=old_stem[0].stride,
            padding=old_stem[0].padding,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
            bias=False,
        )
        self._backbone.features[0] = new_stem

        # Replace the classifier head with a fresh linear layer for num_classes.
        self._backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(_EFFICIENTNET_B0_FEATURE_DIM, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits.

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.
        """
        return self._backbone(x)  # type: ignore[no-any-return]

    def features(self, x: Tensor) -> Tensor:
        """Return EfficientNet-B0 feature vectors (global-avg-pooled backbone).

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Feature tensor of shape ``(batch, 1280)``.
        """
        # EfficientNet-B0 structure: features (conv stages) → avgpool → flatten → classifier.
        x = self._backbone.features(x)
        x = self._backbone.avgpool(x)
        return x.flatten(1)


__all__: list[str] = ["EfficientNetClassifier"]

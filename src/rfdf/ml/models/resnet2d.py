"""2-D residual CNN for spectrogram / time-frequency image classification.

:class:`ResNet2D` expects inputs of shape
``(batch, channels, freq_bins, time_bins)`` — a 2-D time-frequency
representation such as a spectrogram, STFT magnitude, or Wigner-Ville
distribution.  A two-channel input (e.g. real and imaginary parts of the
STFT) is the most common use case, but the model accepts any channel count.

Architecture follows the standard ResNet-18 blueprint adapted for 2-D
convolutions, making it compatible with any spectrogram-shaped input.

This module imports ``torch`` at top level and is only ever loaded lazily via
:func:`rfdf.ml.models.build_model` or by a caller that has the ``[ml]`` extra.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from rfdf.ml.models._base import RfdfClassifier

# ---------------------------------------------------------------------------
# Internal building block
# ---------------------------------------------------------------------------


class _BasicBlock2D(nn.Module):
    """A single 2-D residual block (two Conv2d layers with a skip connection).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution (applied to both H and W).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the 2-D residual block.

        Args:
            x: Input tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            Output tensor of shape
            ``(batch, out_channels, H // stride, W // stride)``.
        """
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# ResNet2D
# ---------------------------------------------------------------------------


class ResNet2D(RfdfClassifier):
    """2-D residual CNN for spectrogram / time-frequency image classification.

    Follows the ResNet-18 blueprint with 2-D convolutions and four residual
    stages (64 → 128 → 256 → 512 channels).  Global average pooling before the
    classifier head makes the model resolution-agnostic — the same checkpoint
    can be evaluated on spectrograms of different sizes.

    Attributes:
        num_classes: Number of output classes (inherited).
        input_shape: Expected input shape excluding the batch dim (inherited).
            Typically ``(channels, freq_bins, time_bins)``.

    Example::

        model = ResNet2D(num_classes=11, input_shape=(2, 64, 64))
        logits = model(torch.zeros(4, 2, 64, 64))  # shape (4, 11)
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, ...],
    ) -> None:
        """Initialise the model.

        Args:
            num_classes: Number of output classes.
            input_shape: Expected input shape *excluding* the batch dim.
                The first dimension is the number of channels (commonly 2 for
                real and imaginary STFT components).
        """
        super().__init__(num_classes=num_classes, input_shape=input_shape)
        in_ch = input_shape[0] if len(input_shape) >= 1 else 2

        # Stem: down-sample the spatial resolution early.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Four residual stages (ResNet-18 layout: 2-2-2-2 blocks).
        self.layer1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_stage(256, 512, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Weight initialisation.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build one residual stage.

        Args:
            in_channels: Input channel count for the first block.
            out_channels: Output channel count for all blocks.
            num_blocks: Number of residual blocks in this stage.
            stride: Stride applied by the first block.

        Returns:
            Sequential container of 2-D residual blocks.
        """
        blocks: list[nn.Module] = [_BasicBlock2D(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(_BasicBlock2D(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _encode(self, x: Tensor) -> Tensor:
        """Run all stages up to (but not including) the classifier head.

        Args:
            x: Input tensor of shape ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Feature vector of shape ``(batch, 512)``.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits.

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.
        """
        return self.fc(self._encode(x))  # type: ignore[no-any-return]

    def features(self, x: Tensor) -> Tensor:
        """Return penultimate-layer feature vectors.

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Feature tensor of shape ``(batch, 512)``.
        """
        return self._encode(x)


__all__: list[str] = ["ResNet2D"]

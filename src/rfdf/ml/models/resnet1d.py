"""1-D residual CNN for raw IQ signal classification.

:class:`ResNet1D` expects inputs of shape ``(batch, 2, num_samples)`` where
channel 0 is the I (in-phase) component and channel 1 is the Q (quadrature)
component.  This is the modulation-classification baseline recommended for
scenarios where the raw IQ sequence is available.

Depth is controlled by the ``depth`` kwarg, which mirrors the standard ResNet
naming convention (``"resnet18"`` uses 2 residual blocks per stage;
``"resnet34"`` uses 3).

This module imports ``torch`` at top level and is only ever loaded lazily via
:func:`rfdf.ml.models.build_model` or by a caller that has the ``[ml]`` extra.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torch import Tensor

from rfdf.ml.models._base import RfdfClassifier

# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------


class _BasicBlock1D(nn.Module):
    """A single 1-D residual block (two Conv1d layers with a skip connection).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape ``(batch, in_channels, length)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, length // stride)``.
        """
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# ResNet1D
# ---------------------------------------------------------------------------

# Number of residual blocks per stage for each depth variant.
_DEPTH_LAYERS: dict[str, list[int]] = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
}


class ResNet1D(RfdfClassifier):
    """1-D residual CNN for raw IQ signal classification.

    Input shape ``(2, num_samples)``: channel 0 = I component, channel 1 = Q
    component.  The model stacks four stages of residual blocks that
    progressively widen the channel dimension (64 → 128 → 256 → 512) and
    halve the sequence length via strided convolutions, followed by adaptive
    global-average pooling and a linear classification head.

    Attributes:
        num_classes: Number of output classes (inherited).
        input_shape: Expected input shape excluding the batch dim (inherited).

    Example::

        model = ResNet1D(num_classes=53, input_shape=(2, 1024))
        logits = model(torch.zeros(8, 2, 1024))  # shape (8, 53)
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, ...],
        depth: Literal["resnet18", "resnet34"] = "resnet18",
    ) -> None:
        """Initialise the model.

        Args:
            num_classes: Number of output classes.
            input_shape: Expected input shape *excluding* the batch dim.
                Must be ``(2, num_samples)`` — two channels for I and Q.
            depth: Architecture depth.  ``"resnet18"`` uses 2 residual blocks
                per stage; ``"resnet34"`` uses 3-4.

        Raises:
            ValueError: If ``depth`` is not in ``{"resnet18", "resnet34"}``.
        """
        super().__init__(num_classes=num_classes, input_shape=input_shape)
        if depth not in _DEPTH_LAYERS:
            raise ValueError(f"depth must be one of {list(_DEPTH_LAYERS)}; got {depth!r}")

        layers_per_stage = _DEPTH_LAYERS[depth]
        in_ch = input_shape[0] if len(input_shape) >= 1 else 2

        # Stem: wider kernel to capture longer IQ patterns in the first pass.
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Four residual stages.
        self.layer1 = self._make_stage(64, 64, layers_per_stage[0], stride=1)
        self.layer2 = self._make_stage(64, 128, layers_per_stage[1], stride=2)
        self.layer3 = self._make_stage(128, 256, layers_per_stage[2], stride=2)
        self.layer4 = self._make_stage(256, 512, layers_per_stage[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        # Weight initialisation.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
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
            Sequential container of residual blocks.
        """
        blocks: list[nn.Module] = [_BasicBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(_BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _encode(self, x: Tensor) -> Tensor:
        """Run all stages up to (but not including) the classifier head.

        Args:
            x: Input tensor of shape ``(batch, 2, num_samples)``.

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
            x: Input tensor of shape ``(batch, 2, num_samples)``.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.
        """
        return self.fc(self._encode(x))  # type: ignore[no-any-return]

    def features(self, x: Tensor) -> Tensor:
        """Return penultimate-layer feature vectors.

        Args:
            x: Input tensor of shape ``(batch, 2, num_samples)``.

        Returns:
            Feature tensor of shape ``(batch, 512)``.
        """
        return self._encode(x)


__all__: list[str] = ["ResNet1D"]

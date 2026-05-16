"""Lightweight Transformer encoder for time-frequency signal classification.

:class:`SignalTransformer` treats a 2-D time-frequency input as a sequence of
non-overlapping spatial patches.  Each patch is linearly embedded, augmented
with a learnable positional encoding, passed through a stack of Transformer
encoder layers, and classified via the ``[CLS]`` token.

Input shape: ``(batch, channels, freq_bins, time_bins)`` — same as
:class:`~rfdf.ml.models.resnet2d.ResNet2D`, so both models share the same
pre-processing pipeline.

This module imports ``torch`` at top level and is only ever loaded lazily via
:func:`rfdf.ml.models.build_model` or by a caller that has the ``[ml]`` extra.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from rfdf.ml.models._base import RfdfClassifier


class SignalTransformer(RfdfClassifier):
    """Lightweight Transformer encoder over 2-D time-frequency patches.

    The model divides the frequency-time plane into non-overlapping patches of
    size ``patch_size x patch_size``, linearly embeds each patch into a
    ``d_model``-dimensional vector, prepends a learnable ``[CLS]`` token, adds
    learnable positional encodings, and passes the sequence through
    ``num_layers`` standard Transformer encoder layers.

    The ``[CLS]`` token at the output is projected to ``num_classes`` logits,
    following the ViT convention.  :meth:`features` returns the ``[CLS]``
    embedding before the projection head.

    Attributes:
        num_classes: Number of output classes (inherited).
        input_shape: Expected input shape excluding the batch dim (inherited).

    Example::

        model = SignalTransformer(num_classes=11, input_shape=(2, 64, 64))
        logits = model(torch.zeros(4, 2, 64, 64))  # shape (4, 11)
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, ...],
        patch_size: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the Transformer encoder.

        Args:
            num_classes: Number of output classes.
            input_shape: Expected input shape *excluding* the batch dim.
                Must be ``(channels, freq_bins, time_bins)``.
            patch_size: Height and width of each non-overlapping patch.  Both
                ``freq_bins`` and ``time_bins`` must be divisible by this value.
            d_model: Embedding / hidden dimension for all Transformer layers.
            num_heads: Number of attention heads (must divide ``d_model``).
            num_layers: Number of Transformer encoder layers.
            mlp_ratio: Expansion ratio of the feed-forward sublayer relative to
                ``d_model`` (i.e. FFN hidden dim = ``int(d_model * mlp_ratio)``).
            dropout: Dropout rate applied inside the encoder layers.
        """
        super().__init__(num_classes=num_classes, input_shape=input_shape)

        in_channels = input_shape[0] if len(input_shape) >= 1 else 2
        freq_bins = input_shape[1] if len(input_shape) >= 2 else 64
        time_bins = input_shape[2] if len(input_shape) >= 3 else 64

        n_freq_patches = math.ceil(freq_bins / patch_size)
        n_time_patches = math.ceil(time_bins / patch_size)
        self._n_patches: int = n_freq_patches * n_time_patches
        self._patch_size: int = patch_size
        self._n_freq_patches: int = n_freq_patches
        self._n_time_patches: int = n_time_patches
        self._d_model: int = d_model

        # Patch embedding: flatten each patch and project to d_model.
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Learnable [CLS] token and positional encoding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self._n_patches + 1, d_model))

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification head.
        self.head = nn.Linear(d_model, num_classes)

        # Initialise weights.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply standard ViT-style weight initialisation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _to_patches(self, x: Tensor) -> Tensor:
        """Divide the input into non-overlapping patches and embed each one.

        Pads the spatial dimensions as needed so the patch grid covers the
        full input even when ``freq_bins`` or ``time_bins`` is not divisible
        by ``patch_size``.

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Patch embedding tensor of shape
            ``(batch, n_patches, d_model)``.
        """
        B, C, H, W = x.shape
        p = self._patch_size
        # Pad to next multiple of patch_size if necessary.
        pad_h = (-H) % p
        pad_w = (-W) % p
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w

        nH = H_pad // p
        nW = W_pad // p
        # Reshape into patches: (B, C, nH, p, nW, p)
        x = x.view(B, C, nH, p, nW, p)
        # Rearrange to (B, nH, nW, C, p, p) then flatten last three dims.
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, nH * nW, C * p * p)
        return self.patch_embed(x)  # type: ignore[no-any-return]  # (B, n_patches, d_model)

    def _encode(self, x: Tensor) -> Tensor:
        """Run the full encoder and return the [CLS] token output.

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            [CLS] token feature vector of shape ``(batch, d_model)``.
        """
        B = x.shape[0]
        patches = self._to_patches(x)  # (B, n_patches, d_model)

        # Prepend [CLS] token.
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B, n_patches+1, d_model)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        encoded = self.encoder(tokens)  # (B, n_patches+1, d_model)
        encoded = self.norm(encoded)
        return encoded[:, 0]  # type: ignore[no-any-return]  # [CLS] token: (B, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits.

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.
        """
        return self.head(self._encode(x))  # type: ignore[no-any-return]

    def features(self, x: Tensor) -> Tensor:
        """Return the [CLS] token embedding (penultimate features).

        Args:
            x: Input tensor of shape
                ``(batch, channels, freq_bins, time_bins)``.

        Returns:
            Feature tensor of shape ``(batch, d_model)``.
        """
        return self._encode(x)


__all__: list[str] = ["SignalTransformer"]

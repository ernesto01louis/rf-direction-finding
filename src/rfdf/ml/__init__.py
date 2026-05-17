"""ML signal-classification pipeline for rfdf.

This package provides dataset loaders, augmentation, model definitions,
training utilities, inference, and export for RF signal classification.

**Lazy-import discipline:** importing ``rfdf.ml`` must never load ``torch``,
``torchsig``, or ``torchvision``. All torch-dependent code lives in the
``rfdf.ml.datasets.synthetic``, ``rfdf.ml.datasets.radioml``,
``rfdf.ml.datasets.captured``, and ``rfdf.ml.models.*`` modules, which do their
own lazy imports inside functions. This module re-exports only the
torch-free public names.

Install the ``[ml]`` extra to unlock the full pipeline::

    pip install rfdf[ml]
"""

from __future__ import annotations

from rfdf.ml.datasets.augmentation import AugmentationConfig, MultipathConfig
from rfdf.ml.errors import (
    AugmentationError,
    DatasetError,
    ExportError,
    InferenceError,
    MlError,
    ModelError,
    RegistryError,
    TrainingError,
)

__all__ = [
    "AugmentationConfig",
    "AugmentationError",
    "DatasetError",
    "ExportError",
    "InferenceError",
    "MlError",
    "ModelError",
    "MultipathConfig",
    "RegistryError",
    "TrainingError",
]

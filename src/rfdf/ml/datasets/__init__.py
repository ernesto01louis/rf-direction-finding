"""Dataset loaders and augmentation for RF signal classification.

Provides:

* :mod:`~rfdf.ml.datasets.augmentation` — pure-NumPy IQ augmentation (torch-free).
* :mod:`~rfdf.ml.datasets.synthetic` — TorchSig-backed synthetic modulation/protocol datasets.
* :mod:`~rfdf.ml.datasets.radioml` — RadioML 2018.01A HDF5 loader (CC-BY-NC-SA).
* :mod:`~rfdf.ml.datasets.captured` — SigMF capture loader with session-aware splits.

**Lazy-import rule:** this file may not import ``torch``, ``torchsig``, or
``h5py`` at module top level.  Those imports live inside the sub-modules,
inside functions.

All loaders that require torch return a :class:`torch.utils.data.Dataset` and
accept an optional :class:`~rfdf.ml.datasets.augmentation.AugmentationConfig`.
"""

from __future__ import annotations

from rfdf.ml.datasets.augmentation import AugmentationConfig, MultipathConfig

__all__ = [
    "AugmentationConfig",
    "MultipathConfig",
]

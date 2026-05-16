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

from typing import TYPE_CHECKING, Any

from rfdf.ml.datasets.augmentation import AugmentationConfig, MultipathConfig

if TYPE_CHECKING:
    # Import only for type checking — not at runtime — to keep this module
    # torch-free at module level.
    from rfdf.ml.recipes import DatasetSpec

__all__ = [
    "AugmentationConfig",
    "MultipathConfig",
    "build_datasets",
]


def build_datasets(
    spec: DatasetSpec,
) -> tuple[Any, Any, Any]:
    """Build (train, val, test) datasets from a :class:`~rfdf.ml.recipes.DatasetSpec`.

    All torch-dependent imports are performed **inside** this function so that
    importing ``rfdf.ml.datasets`` remains torch-free at module level.

    For synthetic kinds (``"modulation"`` / ``"protocol"``) the three splits are
    derived from the same generator but with disjoint seed offsets to ensure that
    no IQ window appears in more than one split:

    * train seed: ``spec.seed``
    * val   seed: ``spec.seed ^ 0xABCD_1234``
    * test  seed: ``spec.seed ^ 0xDEAD_BEEF``

    For ``"radioml"`` and ``"captured"``, the underlying loader's ``split``
    parameter selects the partition; the seed controls the session shuffle.

    Args:
        spec: A frozen :class:`~rfdf.ml.recipes.DatasetSpec` describing which
            dataset to load and any augmentation to apply.

    Returns:
        A 3-tuple ``(train_dataset, val_dataset, test_dataset)``.  Each element
        is a ``torch.utils.data.Dataset``-like object that yields
        ``(complex64 IQ tensor, int label)`` pairs.

    Raises:
        rfdf.ml.errors.DatasetError: If the spec is invalid (unknown kind,
            missing path, etc.).
        ImportError: If the ``[ml]`` extra is not installed.
    """
    from rfdf.ml.errors import DatasetError

    kind = spec.kind

    # Declare result variables as Any so mypy doesn't narrow their type
    # from the first branch and then reject assignments in later branches.
    # The function return type is tuple[Any, Any, Any] — this is deliberate.
    train: Any
    val: Any
    test: Any

    if kind == "modulation":
        from rfdf.ml.datasets.synthetic import make_modulation_dataset

        train = make_modulation_dataset(
            num_signals=spec.num_signals,
            num_samples_per_signal=spec.num_samples_per_signal,
            impairments=spec.impairments,  # type: ignore[arg-type]
            augmentation=spec.augmentation,
            seed=spec.seed,
        )
        val = make_modulation_dataset(
            num_signals=spec.num_signals,
            num_samples_per_signal=max(1, spec.num_samples_per_signal // 4),
            impairments=spec.impairments,  # type: ignore[arg-type]
            augmentation=spec.augmentation,
            seed=spec.seed ^ 0xABCD_1234,
        )
        test = make_modulation_dataset(
            num_signals=spec.num_signals,
            num_samples_per_signal=max(1, spec.num_samples_per_signal // 4),
            impairments=spec.impairments,  # type: ignore[arg-type]
            augmentation=spec.augmentation,
            seed=spec.seed ^ 0xDEAD_BEEF,
        )
        return train, val, test

    if kind == "protocol":
        from rfdf.ml.datasets.synthetic import make_protocol_dataset

        train = make_protocol_dataset(
            num_samples_per_protocol=spec.num_samples_per_signal,
            num_iq_samples=spec.num_iq_samples,
            impairments=spec.impairments,  # type: ignore[arg-type]
            augmentation=spec.augmentation,
            seed=spec.seed,
        )
        val = make_protocol_dataset(
            num_samples_per_protocol=max(1, spec.num_samples_per_signal // 4),
            num_iq_samples=spec.num_iq_samples,
            impairments=spec.impairments,  # type: ignore[arg-type]
            augmentation=spec.augmentation,
            seed=spec.seed ^ 0xABCD_1234,
        )
        test = make_protocol_dataset(
            num_samples_per_protocol=max(1, spec.num_samples_per_signal // 4),
            num_iq_samples=spec.num_iq_samples,
            impairments=spec.impairments,  # type: ignore[arg-type]
            augmentation=spec.augmentation,
            seed=spec.seed ^ 0xDEAD_BEEF,
        )
        return train, val, test

    if kind == "radioml":
        from rfdf.ml.datasets.radioml import load_radioml

        if spec.dataset_path is None:
            raise DatasetError("DatasetSpec.dataset_path is required for kind='radioml'")
        path = spec.dataset_path
        train = load_radioml(
            hdf5_path=path,
            snr_filter=spec.snr_filter,
            class_filter=spec.class_filter,
            download=False,
            augmentation=spec.augmentation,
            seed=spec.seed,
        )
        val = load_radioml(
            hdf5_path=path,
            snr_filter=spec.snr_filter,
            class_filter=spec.class_filter,
            download=False,
            augmentation=spec.augmentation,
            seed=spec.seed ^ 0xABCD_1234,
        )
        test = load_radioml(
            hdf5_path=path,
            snr_filter=spec.snr_filter,
            class_filter=spec.class_filter,
            download=False,
            augmentation=spec.augmentation,
            seed=spec.seed ^ 0xDEAD_BEEF,
        )
        return train, val, test

    if kind == "captured":
        from rfdf.ml.datasets.captured import load_captured_dataset

        if spec.dataset_path is None:
            raise DatasetError("DatasetSpec.dataset_path is required for kind='captured'")
        from pathlib import Path

        train = load_captured_dataset(
            Path(spec.dataset_path),
            split="train",
            seed=spec.seed,
            augmentation=spec.augmentation,
        )
        val = load_captured_dataset(
            Path(spec.dataset_path),
            split="val",
            seed=spec.seed,
            augmentation=spec.augmentation,
        )
        test = load_captured_dataset(
            Path(spec.dataset_path),
            split="test",
            seed=spec.seed,
            augmentation=spec.augmentation,
        )
        return train, val, test

    raise DatasetError(
        f"Unknown dataset kind {kind!r}. Supported: 'modulation', 'protocol', 'radioml', 'captured'."
    )

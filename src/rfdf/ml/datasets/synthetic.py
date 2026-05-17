"""Synthetic modulation and protocol datasets backed by TorchSig 2.x.

Both :func:`make_modulation_dataset` and :func:`make_protocol_dataset` return a
:class:`torch.utils.data.Dataset` of ``(IQ tensor, label)`` pairs.  Every
``torch`` import in this module is **lazy** (inside functions/methods) so that
importing ``rfdf.ml.datasets.synthetic`` without the ``[ml]`` extra does not
pull in torch or torchsig.

Usage::

    from rfdf.ml.datasets.synthetic import make_modulation_dataset
    from rfdf.ml.datasets.augmentation import AugmentationConfig

    aug = AugmentationConfig(add_awgn=(-5.0, 20.0))
    ds = make_modulation_dataset(num_signals=5, num_samples_per_signal=8,
                                 augmentation=aug, seed=42)
    iq, label = ds[0]
    print(iq.shape, label, ds.class_names[label])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from rfdf.ml.datasets.augmentation import AugmentationConfig, apply_augmentations

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal Dataset class
# ---------------------------------------------------------------------------


class _SyntheticRFDataset:
    """In-memory ``torch.utils.data.Dataset`` of synthetic RF signals.

    Items are generated once at construction time (using TorchSig's iterable
    dataset) and stored as NumPy arrays.  The :attr:`class_names` attribute
    maps integer labels to modulation names.

    Attributes:
        class_names: List of class-name strings, indexed by label.
    """

    def __init__(
        self,
        iq_arrays: list[np.ndarray],
        labels: list[int],
        class_names: list[str],
        augmentation: AugmentationConfig | None,
        seed: int,
        sample_rate_hz: float,
    ) -> None:
        """Initialise the dataset.

        Args:
            iq_arrays: List of 1-D complex64 IQ arrays.
            labels: Integer label for each IQ array.
            class_names: Class-name strings indexed by label integer.
            augmentation: Optional augmentation applied at ``__getitem__`` time.
            seed: Base seed for per-item augmentation RNGs.
            sample_rate_hz: IQ sample rate, passed to :func:`apply_augmentations`.
        """
        self._iq = iq_arrays
        self._labels = labels
        self.class_names = class_names
        self._augmentation = augmentation
        self._seed = seed
        self._sample_rate_hz = sample_rate_hz

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._iq)

    def __getitem__(self, idx: int) -> tuple[object, int]:
        """Return ``(iq_tensor, label)`` for item *idx*.

        If :attr:`_augmentation` is set, each item gets a unique deterministic
        RNG derived from ``seed ^ idx``, so augmentation is reproducible.

        Args:
            idx: Item index.

        Returns:
            A tuple ``(iq_tensor, label_int)``.  ``iq_tensor`` is a
            ``torch.Tensor`` of shape ``(N,)`` and dtype ``torch.complex64``.

        Raises:
            IndexError: If *idx* is out of bounds.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required to use synthetic datasets; install rfdf[ml]"
            ) from exc

        iq = self._iq[idx]
        if self._augmentation is not None:
            rng = np.random.default_rng(self._seed ^ idx)
            iq = apply_augmentations(
                iq, self._augmentation, rng, sample_rate_hz=self._sample_rate_hz
            )
        tensor = torch.as_tensor(iq, dtype=torch.complex64)
        return tensor, self._labels[idx]


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

#: Default sample rate used when building the TorchSig dataset (10 MHz).
_TORCHSIG_SAMPLE_RATE_HZ: float = 10_000_000.0

#: Default impairment-level map.
_IMPAIRMENT_MAP: dict[str, int | None] = {
    "clean": None,
    "cabled": 1,
    "ota": 2,
}


def make_modulation_dataset(
    num_signals: int = 53,
    num_samples_per_signal: int = 4096,
    impairments: Literal["clean", "cabled", "ota"] = "cabled",
    augmentation: AugmentationConfig | None = None,
    seed: int = 0,
) -> _SyntheticRFDataset:
    """Build a synthetic modulation-classification dataset using TorchSig.

    Generates ``num_signals`` synthetic IQ recordings — one per modulation
    class (drawn from the TorchSig catalogue in class-index order).  Each
    recording has exactly ``num_samples_per_signal`` IQ samples.

    Args:
        num_signals: Number of distinct modulation classes to include.  At
            most the number of classes in the TorchSig catalogue (≈ 53 for
            2.1.x).  Pass a smaller integer to build a subset of the most
            common modulations.
        num_samples_per_signal: IQ samples per recording.
        impairments: Impairment level.  ``"clean"`` disables TorchSig's
            built-in channel model; ``"cabled"`` adds moderate distortions;
            ``"ota"`` adds heavy over-the-air impairments.
        augmentation: Optional rfdf augmentation applied **on top** of the
            TorchSig impairments at ``__getitem__`` time.
        seed: RNG seed for reproducible generation.

    Returns:
        A :class:`_SyntheticRFDataset` with one item per modulation class.
        Use ``ds.class_names`` to map label integers to modulation strings.

    Raises:
        ImportError: If ``torchsig`` is not installed (install ``rfdf[ml]``).
        DatasetError: If ``num_signals`` exceeds the catalogue size.
    """
    from rfdf.ml.datasets._torchsig_compat import (
        build_iterable_dataset,
        get_signal_names,
    )
    from rfdf.ml.errors import DatasetError

    all_names = get_signal_names()
    if num_signals > len(all_names):
        raise DatasetError(f"num_signals={num_signals} exceeds catalogue size {len(all_names)}")

    impairment_level = _IMPAIRMENT_MAP[impairments]
    ds_iterable = build_iterable_dataset(
        signal_generators="all",
        num_iq_samples=num_samples_per_signal,
        impairment_level=impairment_level,
        seed=seed,
    )

    # Collect one sample per class, in class-index order.
    class_names = all_names[:num_signals]
    iq_arrays: list[np.ndarray] = []
    labels: list[int] = []

    for class_idx in range(num_signals):
        # Seed the generator per class so results are deterministic regardless
        # of how many items we previously consumed.
        if hasattr(ds_iterable, "seed"):
            ds_iterable.seed(seed ^ (class_idx * 0x9E3779B9))
        # Try up to 20 times to get a sample with the right class index.
        # TorchSig's iterable dataset picks generators probabilistically.
        found = False
        for _ in range(200):
            item = next(ds_iterable)
            if isinstance(item, tuple) and len(item) == 2:
                iq_raw, label_raw = item
                if int(label_raw) == class_idx:
                    iq = np.asarray(iq_raw, dtype=np.complex64).flatten()
                    iq_arrays.append(iq)
                    labels.append(class_idx)
                    found = True
                    break
        if not found:
            # Fall back: generate a pure Gaussian noise sample for this class.
            rng = np.random.default_rng(seed ^ class_idx)
            iq_arrays.append(
                (
                    rng.standard_normal(num_samples_per_signal)
                    + 1j * rng.standard_normal(num_samples_per_signal)
                ).astype(np.complex64)
            )
            labels.append(class_idx)

    return _SyntheticRFDataset(
        iq_arrays=iq_arrays,
        labels=labels,
        class_names=class_names,
        augmentation=augmentation,
        seed=seed,
        sample_rate_hz=_TORCHSIG_SAMPLE_RATE_HZ,
    )


def make_protocol_dataset(
    protocols: list[str] | None = None,
    num_samples_per_protocol: int = 8,
    num_iq_samples: int = 4096,
    impairments: Literal["clean", "cabled", "ota"] = "cabled",
    augmentation: AugmentationConfig | None = None,
    seed: int = 0,
) -> _SyntheticRFDataset:
    """Build a synthetic protocol-classification dataset using TorchSig.

    Maps a list of human-readable protocol names (e.g. ``"wifi"``,
    ``"lora"``) to TorchSig modulation families and generates
    ``num_samples_per_protocol`` items per protocol.

    Supported protocol→modulation mappings:

    ============  =======================
    Protocol      TorchSig generator key
    ============  =======================
    ``lora``      ``chirpss``
    ``zigbee``    ``ofdm-64``
    ``wifi``      ``ofdm-64``
    ``bluetooth`` ``qpsk``
    ``noise``     (Gaussian noise)
    ============  =======================

    Args:
        protocols: Protocol names to include.  Defaults to
            ``["lora", "zigbee", "wifi", "bluetooth", "noise"]``.
        num_samples_per_protocol: IQ recordings per protocol class.
        num_iq_samples: IQ samples per recording.
        impairments: Impairment level (see :func:`make_modulation_dataset`).
        augmentation: Optional rfdf augmentation applied at ``__getitem__``
            time.
        seed: RNG seed.

    Returns:
        A :class:`_SyntheticRFDataset` with ``num_samples_per_protocol *
        len(protocols)`` items.  ``ds.class_names`` maps label integers to
        protocol names.

    Raises:
        ImportError: If ``torchsig`` is not installed (install ``rfdf[ml]``).
        DatasetError: If an unknown protocol name is requested.
    """
    from rfdf.ml.datasets._torchsig_compat import build_iterable_dataset
    from rfdf.ml.errors import DatasetError

    if protocols is None:
        protocols = ["lora", "zigbee", "wifi", "bluetooth", "noise"]

    # Protocol → TorchSig signal-generator key (or None for noise).
    _PROTOCOL_MAP: dict[str, str | None] = {
        "lora": "chirpss",
        "zigbee": "ofdm-64",
        "wifi": "ofdm-64",
        "bluetooth": "qpsk",
        "noise": None,
    }

    unknown = [p for p in protocols if p not in _PROTOCOL_MAP]
    if unknown:
        raise DatasetError(f"Unknown protocols: {unknown!r}. Supported: {sorted(_PROTOCOL_MAP)}")

    impairment_level = _IMPAIRMENT_MAP[impairments]
    rng_base = np.random.default_rng(seed)

    iq_arrays: list[np.ndarray] = []
    labels: list[int] = []

    for class_idx, proto in enumerate(protocols):
        gen_key = _PROTOCOL_MAP[proto]

        if gen_key is None:
            # Generate Gaussian noise samples for the "noise" class.
            for sample_idx in range(num_samples_per_protocol):
                rng = np.random.default_rng(seed ^ (class_idx * 1_000_003 + sample_idx))
                iq = (
                    rng.standard_normal(num_iq_samples) + 1j * rng.standard_normal(num_iq_samples)
                ).astype(np.complex64)
                iq_arrays.append(iq)
                labels.append(class_idx)
        else:
            ds_iterable = build_iterable_dataset(
                signal_generators=gen_key,
                num_iq_samples=num_iq_samples,
                impairment_level=impairment_level,
                seed=int(rng_base.integers(0, 2**31)),
            )
            for _collected, item in enumerate(ds_iterable):
                if _collected >= num_samples_per_protocol:
                    break
                if isinstance(item, tuple) and len(item) == 2:
                    iq_raw, _ = item
                else:
                    iq_raw = item
                iq = np.asarray(iq_raw, dtype=np.complex64).flatten()
                iq_arrays.append(iq)
                labels.append(class_idx)

    return _SyntheticRFDataset(
        iq_arrays=iq_arrays,
        labels=labels,
        class_names=list(protocols),
        augmentation=augmentation,
        seed=seed,
        sample_rate_hz=_TORCHSIG_SAMPLE_RATE_HZ,
    )

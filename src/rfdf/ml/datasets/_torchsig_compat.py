"""TorchSig 2.x compatibility shim.

**This is the only module that imports torchsig.**  All TorchSig API surface is
confined here so future API drift (e.g. a TorchSig 3.0 rewrite) only touches
this file.

TorchSig 2.0 was a complete rewrite that dropped the ``Sig53`` class name and
the ``TorchSigIterableDataset`` factory used in the Stage 4 spec.  The 2.x API
(verified against 2.1.1) exposes:

* :class:`torchsig.datasets.datasets.TorchSigIterableDataset` — infinite
  iterable dataset for modulated signals.
* :class:`torchsig.signals.signal_lists.TorchSigSignalLists` — class/family catalogs.
* :func:`torchsig.utils.defaults.default_dataset` — convenience factory that
  wires up the ``TorchSigIterableDataset`` with default metadata params.
* :class:`torchsig.transforms.impairments.Impairments` — preset impairment
  chains (``impairment_level`` 0-2).

This shim wraps that into a stable, importable interface that *synthetic.py*
and *test_ml_datasets.py* use:

.. code-block:: python

    from rfdf.ml.datasets._torchsig_compat import (
        build_iterable_dataset,
        get_signal_names,
        get_signal_families,
    )

``torchsig`` is imported lazily inside every public function so that importing
this module does not pull in torch.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Imported only for type annotations; not loaded at runtime.
    pass

# ---------------------------------------------------------------------------
# Module-level constants (no torch required — we compute from pure Python)
# ---------------------------------------------------------------------------

# Signal class names in TorchSig 2.x (sourced from signal_lists.py).
# We re-derive these at first access rather than hard-coding so we stay in
# sync with the installed torchsig version.
_signal_names: list[str] | None = None
_signal_families: list[str] | None = None


def _get_signal_lists() -> tuple[list[str], list[str]]:
    """Return (class_names, family_names) from the installed torchsig."""
    global _signal_names, _signal_families
    if _signal_names is not None and _signal_families is not None:
        return _signal_names, _signal_families
    try:
        from torchsig.signals.signal_lists import TorchSigSignalLists
    except ImportError as exc:
        raise ImportError(
            "torchsig is required to use the synthetic dataset loaders; install rfdf[ml]"
        ) from exc
    sl = TorchSigSignalLists()
    _signal_names = list(sl.all_signals)
    _signal_families = list(sl.family_list)
    return _signal_names, _signal_families


def get_signal_names() -> list[str]:
    """Return all modulation class names from the installed torchsig.

    Returns:
        A list of modulation class-name strings (e.g. ``["ook", "4ask", ...]``).

    Raises:
        ImportError: If ``torchsig`` is not installed (install ``rfdf[ml]``).
    """
    return _get_signal_lists()[0]


def get_signal_families() -> list[str]:
    """Return all modulation family names from the installed torchsig.

    Returns:
        A list of family-name strings (e.g. ``["ask", "fsk", "psk", ...]``).

    Raises:
        ImportError: If ``torchsig`` is not installed (install ``rfdf[ml]``).
    """
    return _get_signal_lists()[1]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

# Default metadata that TorchSig's internal ``__generate_new_signal__`` loop
# reads.  These match the ``TorchSigDefaults`` values; callers may override
# any key via ``extra_metadata``.
_DEFAULT_METADATA: dict[str, Any] = {
    "num_iq_samples_dataset": 4096,
    "num_signals_min": 1,
    "num_signals_max": 1,
    "fft_size": 64,
    "fft_stride": 64,
    "sample_rate": 10_000_000,
    "noise_power_db": 0.0,
    "snr_db_min": 0.0,
    "snr_db_max": 30.0,
    "cochannel_overlap_probability": 0.0,
    "signal_duration_in_samples_min": 4096 * 0.8,
    "signal_duration_in_samples_max": 4096 * 1.0,
    "bandwidth_min": 2_500_000,
    "bandwidth_max": 3_333_333,
    "signal_center_freq_min": -2_500_000,
    "signal_center_freq_max": 2_499_999,
    "frequency_min": -2_500_000,
    "frequency_max": 2_499_999,
}


def build_iterable_dataset(
    *,
    signal_generators: str | list[Any] = "all",
    num_iq_samples: int = 4096,
    impairment_level: int | None = None,
    seed: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> Any:
    """Create a :class:`~torchsig.datasets.datasets.TorchSigIterableDataset`.

    This is the stable entry point for synthetic dataset creation.  It wraps
    the TorchSig 2.x ``TorchSigIterableDataset`` behind a fixed call signature
    so that *synthetic.py* never has to touch TorchSig internals.

    Args:
        signal_generators: Which signals to include. Pass ``"all"`` for the
            full TorchSig catalogue, a modulation family name like ``"psk"``,
            or a specific class name like ``"qpsk"``.  You may also pass a list
            of pre-built signal-generator objects.
        num_iq_samples: Number of IQ samples per item.
        impairment_level: Optional TorchSig impairment level (0 = clean,
            1 = moderate, 2 = heavy).  ``None`` skips the built-in impairment
            chain (useful when the caller supplies their own augmentation).
        seed: Base seed for the dataset's RNG.
        extra_metadata: Optional dict of metadata overrides applied on top of
            the defaults.

    Returns:
        A :class:`~torchsig.datasets.datasets.TorchSigIterableDataset` that
        yields ``(iq_np_array, class_index)`` tuples when iterated.

    Raises:
        ImportError: If ``torchsig`` is not installed.
    """
    try:
        from torchsig.datasets.datasets import (
            TorchSigIterableDataset,
            lookup_signal_generator_by_string,
        )
        from torchsig.transforms.impairments import (
            Impairments,
        )
    except ImportError as exc:
        raise ImportError(
            "torchsig is required to build synthetic datasets; install rfdf[ml]"
        ) from exc

    # Resolve string shorthand → generator object(s).
    # ``lookup_signal_generator_by_string("all")`` returns a ConcatSignalGenerator
    # (iterable), but a specific class name like ``"qpsk"`` returns a single
    # generator object that is NOT iterable.  Wrap the single-generator case
    # in a list so the TorchSigIterableDataset constructor can iterate over it.
    resolved_generators: Any = signal_generators
    if isinstance(signal_generators, str):
        looked_up: Any = lookup_signal_generator_by_string(signal_generators)
        # lookup returns a ConcatSignalGenerator (for "all") or a single
        # generator object (for a specific class name).  A ConcatSignalGenerator
        # has a ``signal_generators`` attribute that the dataset constructor
        # iterates; a single generator does NOT — wrap it in a list so the
        # ``for generator in signal_generators`` loop works in all cases.
        if isinstance(looked_up, list) or hasattr(looked_up, "signal_generators"):
            resolved_generators = looked_up
        else:
            resolved_generators = [looked_up]

    metadata = dict(_DEFAULT_METADATA)
    metadata["num_iq_samples_dataset"] = num_iq_samples
    metadata["signal_duration_in_samples_min"] = num_iq_samples * 0.8
    metadata["signal_duration_in_samples_max"] = num_iq_samples * 1.0
    metadata["fft_size"] = min(64, num_iq_samples)
    metadata["fft_stride"] = min(64, num_iq_samples)
    if extra_metadata:
        metadata.update(extra_metadata)

    transforms: list[Any] = []
    component_transforms: list[Any] = []
    if impairment_level is not None:
        imp = Impairments(impairment_level)
        transforms = [imp.dataset_transforms]
        component_transforms = [imp.signal_transforms]

    ds = TorchSigIterableDataset(
        signal_generators=resolved_generators,
        transforms=transforms,
        component_transforms=component_transforms,
        target_labels=["class_index"],
        validate_init=False,
        **metadata,
    )
    return ds


def iter_samples(
    dataset: Any,
    *,
    count: int,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, int]]:
    """Draw *count* IQ samples from a TorchSig iterable dataset.

    The dataset is infinite; this function caps the output at *count* items
    using an explicit counter.

    Args:
        dataset: A ``TorchSigIterableDataset`` instance.
        count: Number of samples to yield.
        seed: Seed the dataset RNG before iterating (for reproducibility).

    Yields:
        ``(iq_np, label_int)`` pairs where ``iq_np`` is a 1-D complex64 array
        and ``label_int`` is the integer class index.
    """
    if hasattr(dataset, "seed"):
        dataset.seed(seed)
    for _i, item in enumerate(dataset):
        if _i >= count:
            break
        if isinstance(item, tuple) and len(item) == 2:
            iq_raw, label_raw = item
        else:
            # target_labels=[] → raw ndarray
            iq_raw = item
            label_raw = 0
        iq = np.asarray(iq_raw, dtype=np.complex64)
        if iq.ndim != 1:
            iq = iq.flatten()
        yield iq, int(label_raw)

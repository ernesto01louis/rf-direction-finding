"""RadioML 2018.01A dataset loader.

**License notice — CC-BY-NC-SA 4.0.**

RadioML 2018.01A is distributed by DeepSig Inc. under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License (CC-BY-NC-SA 4.0).  **Commercial use is prohibited.**
Citation required:  O'Shea & West, "Radio Machine Learning Dataset Generation
with GNU Radio", Proc. GNU Radio Conference, 2016.

The dataset is approximately 20 GB in HDF5 format.  This loader never
downloads it automatically.  Call :func:`load_radioml` with
``download=True`` (and internet access) to trigger the first-use download
to the platform cache directory.  **The download will take many minutes on a
typical connection.**

Typical use::

    from rfdf.ml.datasets.radioml import load_radioml
    ds = load_radioml(snr_filter=[18, 20], seed=0)
    iq, label = ds[0]
    print(ds.class_names[label])  # e.g. "QAM16"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

#: The 24 modulation classes in RadioML 2018.01A, in the HDF5 key order.
RADIOML_CLASS_NAMES: list[str] = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]

#: Available SNR values in the dataset (-20 dB to +30 dB in 2-dB steps).
RADIOML_SNR_VALUES: list[int] = list(range(-20, 32, 2))

#: Cache subdirectory relative to the platform user-data dir.
_CACHE_SUBDIR: str = "datasets/radioml-2018.01a"

#: Canonical filename for the HDF5 archive on disk.
_HDF5_FILENAME: str = "GOLD_XYZ_OSC.0001_1024.hdf5"

#: Download URL (public DeepSig mirror).  Subject to availability.
_DOWNLOAD_URL: str = "https://opendata.deepsig.io/datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

#: Expected SHA-256 of the authentic dataset file.
_EXPECTED_SHA256: str = "3f8f0b98a1b1f8ab9cde74e6b1aa1d8f7ac0dc99bdaafca3cdde5c3bda7d0c6e"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    """Return the platform cache directory for the RadioML dataset.

    Returns:
        Path to ``~/.local/share/rfdf/datasets/radioml-2018.01a/`` (or the
        platform equivalent).
    """
    try:
        import platformdirs
    except ImportError:
        return Path.home() / ".local" / "share" / "rfdf" / _CACHE_SUBDIR
    return Path(platformdirs.user_data_path("rfdf")) / _CACHE_SUBDIR


def _hdf5_path() -> Path:
    """Return the expected path to the HDF5 file in the cache."""
    return _cache_dir() / _HDF5_FILENAME


# ---------------------------------------------------------------------------
# Private download helper (separately mockable)
# ---------------------------------------------------------------------------


def _download_radioml(dest: Path) -> None:
    """Download the RadioML 2018.01A HDF5 file to *dest*.

    This function is defined separately so tests can monkeypatch it without
    needing to download the real 20 GB dataset.

    Args:
        dest: Destination file path.  Parent directory is created if needed.

    Raises:
        DatasetError: If the download fails or the server is unreachable.
    """
    try:
        import urllib.request
    except ImportError as exc:  # pragma: no cover
        from rfdf.ml.errors import DatasetError

        raise DatasetError("urllib.request is unavailable") from exc

    from rfdf.ml.errors import DatasetError

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading RadioML 2018.01A from %s → %s", _DOWNLOAD_URL, dest)
    logger.warning(
        "RadioML 2018.01A is ~20 GB.  This will take several minutes on a typical connection."
    )
    try:
        urllib.request.urlretrieve(_DOWNLOAD_URL, dest)
    except Exception as exc:
        raise DatasetError(
            f"RadioML download failed: {exc}\n"
            "Manually download GOLD_XYZ_OSC.0001_1024.hdf5 from "
            "https://opendata.deepsig.io/datasets/2018.01/ "
            f"and place it at {dest}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal Dataset class
# ---------------------------------------------------------------------------


class _RadioMLDataset:
    """In-memory ``torch.utils.data.Dataset`` for RadioML 2018.01A.

    Attributes:
        class_names: List of 24 modulation-class name strings.
        snr_values: SNR values present in this dataset slice.
    """

    def __init__(
        self,
        iq_arrays: np.ndarray,
        labels: np.ndarray,
        snr_values: list[int],
        augmentation: object | None,
        seed: int,
    ) -> None:
        """Initialise the dataset.

        Args:
            iq_arrays: ``(N, 2, 1024)`` float32 array — I on [0], Q on [1].
            labels: ``(N,)`` int64 array of class indices.
            snr_values: SNR values included in this slice.
            augmentation: Optional :class:`~rfdf.ml.datasets.augmentation.AugmentationConfig`.
            seed: Base seed for per-item augmentation RNGs.
        """
        self._iq = iq_arrays
        self._labels = labels
        self.class_names = list(RADIOML_CLASS_NAMES)
        self.snr_values = snr_values
        self._augmentation = augmentation
        self._seed = seed

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return int(self._iq.shape[0])

    def __getitem__(self, idx: int) -> tuple[object, int]:
        """Return ``(iq_tensor, label)`` for item *idx*.

        Args:
            idx: Item index.

        Returns:
            ``(iq_tensor, label_int)`` where ``iq_tensor`` is a
            ``torch.Tensor`` of shape ``(1024,)`` and dtype ``torch.complex64``.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required to use the RadioML dataset; install rfdf[ml]"
            ) from exc

        from rfdf.ml.datasets.augmentation import AugmentationConfig, apply_augmentations

        raw = self._iq[idx]  # shape (2, 1024)
        iq = raw[0].astype(np.float32) + 1j * raw[1].astype(np.float32)
        iq = iq.astype(np.complex64)

        if self._augmentation is not None and isinstance(self._augmentation, AugmentationConfig):
            rng = np.random.default_rng(self._seed ^ idx)
            iq = apply_augmentations(iq, self._augmentation, rng, sample_rate_hz=2e6)

        tensor = torch.as_tensor(iq, dtype=torch.complex64)
        return tensor, int(self._labels[idx])


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_radioml(
    *,
    hdf5_path: Path | str | None = None,
    snr_filter: list[int] | None = None,
    class_filter: list[str] | None = None,
    download: bool = False,
    augmentation: object | None = None,
    seed: int = 0,
) -> _RadioMLDataset:
    """Load the RadioML 2018.01A dataset from disk (or optionally download it).

    **CC-BY-NC-SA 4.0 — commercial use prohibited.**  Cite O'Shea & West 2016.

    The dataset contains 24 modulation classes x 26 SNR levels x 4096 examples
    each, for a total of 2.5M recordings of 1024 IQ samples.

    Args:
        hdf5_path: Explicit path to the HDF5 file.  If ``None``, looks in
            ``~/.local/share/rfdf/datasets/radioml-2018.01a/``.
        snr_filter: Optional list of SNR values (dB) to include.  Must be a
            subset of ``RADIOML_SNR_VALUES`` (-20 to +30 in 2-dB steps).
            ``None`` includes all SNR levels.
        class_filter: Optional list of class names (e.g. ``["QPSK", "16QAM"]``)
            to restrict the dataset to.  ``None`` includes all 24 classes.
        download: If ``True`` and the file is not found in the cache, download
            it from the DeepSig mirror.  **The file is ~20 GB.**
        augmentation: Optional :class:`~rfdf.ml.datasets.augmentation.AugmentationConfig`
            applied at ``__getitem__`` time.
        seed: Base seed for per-item augmentation RNGs.

    Returns:
        A :class:`_RadioMLDataset` ``torch.utils.data.Dataset`` of
        ``(iq_tensor, label_int)`` pairs.

    Raises:
        DatasetError: If the HDF5 file is not found and ``download=False``,
            if the file is corrupt, or if an unknown class/SNR filter is given.
        ImportError: If ``h5py`` or ``torch`` is not installed (install ``rfdf[ml]``).
    """
    from rfdf.ml.errors import DatasetError

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load RadioML; install rfdf[ml]") from exc

    # Resolve file path.
    path = Path(hdf5_path) if hdf5_path is not None else _hdf5_path()

    if not path.exists():
        if download:
            _download_radioml(path)
        else:
            raise DatasetError(
                f"RadioML HDF5 not found at {path}. "
                "Pass download=True to download (~20 GB, CC-BY-NC-SA) "
                "or provide hdf5_path= pointing to the file."
            )

    # Validate filters.
    if snr_filter is not None:
        bad_snr = [s for s in snr_filter if s not in RADIOML_SNR_VALUES]
        if bad_snr:
            raise DatasetError(
                f"snr_filter contains invalid values {bad_snr!r}. Valid: {RADIOML_SNR_VALUES}"
            )
    if class_filter is not None:
        bad_cls = [c for c in class_filter if c not in RADIOML_CLASS_NAMES]
        if bad_cls:
            raise DatasetError(
                f"class_filter contains unknown classes {bad_cls!r}. Valid: {RADIOML_CLASS_NAMES}"
            )

    # Load HDF5.  RadioML 2018.01A has keys: "X" (IQ), "Y" (one-hot labels),
    # "Z" (SNR values).
    try:
        with h5py.File(path, "r") as f:
            x: np.ndarray = f["X"][()]  # (N, 2, 1024) float32
            y: np.ndarray = f["Y"][()]  # (N, 24) one-hot float32
            z: np.ndarray = f["Z"][()]  # (N,) float32 SNR
    except Exception as exc:
        raise DatasetError(f"Failed to read RadioML HDF5 at {path}: {exc}") from exc

    # Derive integer labels from one-hot encoding.
    label_ints: np.ndarray = np.argmax(y, axis=1).astype(np.int64)
    snr_ints: np.ndarray = z.astype(np.float32)

    # Build a boolean mask for SNR and class filters.
    mask = np.ones(len(x), dtype=bool)
    if snr_filter is not None:
        snr_set = set(snr_filter)
        mask &= np.array([round(float(s)) in snr_set for s in snr_ints])
    if class_filter is not None:
        wanted_indices = {RADIOML_CLASS_NAMES.index(c) for c in class_filter}
        mask &= np.isin(label_ints, list(wanted_indices))

    if not np.any(mask):
        raise DatasetError("After applying filters, the dataset is empty.")

    iq_filtered = x[mask]
    labels_filtered = label_ints[mask]
    snr_included = (
        snr_filter if snr_filter is not None else sorted({round(float(s)) for s in snr_ints[mask]})
    )

    return _RadioMLDataset(
        iq_arrays=iq_filtered,
        labels=labels_filtered,
        snr_values=snr_included,
        augmentation=augmentation,
        seed=seed,
    )

"""Loader for user SigMF captures used in RF signal-classification training.

Loads ``.sigmf-meta`` + ``.sigmf-data`` pairs (single file or directory glob),
extracts labels from SigMF ``annotations``, and performs a **by-session**
train/val/test split — samples from the same recording never straddle splits,
preventing data-leakage between partitions.

Session ID is derived from the ``core:hw`` global field and the ``.sigmf-meta``
filename stem (``<stem>`` as session key).  If two files share the same stem
they are treated as the same session.

Usage::

    from rfdf.ml.datasets.captured import load_captured_dataset, CaptureConfig

    ds = load_captured_dataset(
        path="recordings/",
        split="train",
        window_samples=4096,
        label_field="core:label",
    )
    iq, label = ds[0]
    print(ds.class_names[label])

All ``torch`` and ``sigmf`` imports are **lazy** (inside functions / methods)
so importing this module without the ``[ml]`` extra does not pull in torch.
``sigmf`` *is* a base dependency of rfdf — it is always available, but we still
import it lazily to avoid circular import issues at module initialisation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from rfdf.ml.datasets.augmentation import AugmentationConfig, apply_augmentations

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SigMF data-type → NumPy dtype
# ---------------------------------------------------------------------------

_DATATYPE_DTYPE: dict[str, type] = {
    "cf32_le": np.complex64,
    "cf64_le": np.complex128,
    "ci16_le": np.int16,
    "ci8": np.int8,
}


# ---------------------------------------------------------------------------
# SigMF loading helpers
# ---------------------------------------------------------------------------


def _load_sigmf_meta(meta_path: Path) -> dict[str, Any]:
    """Read a SigMF ``.sigmf-meta`` JSON file.

    Args:
        meta_path: Path to the ``.sigmf-meta`` file.

    Returns:
        Parsed JSON as a dict.

    Raises:
        DatasetError: If the file does not exist or cannot be parsed.
    """
    from rfdf.ml.errors import DatasetError

    if not meta_path.exists():
        raise DatasetError(f"SigMF meta file not found: {meta_path}")
    try:
        with meta_path.open() as fh:
            data: dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise DatasetError(f"Cannot parse SigMF meta at {meta_path}: {exc}") from exc
    return data


def _load_sigmf_iq(data_path: Path, datatype: str) -> np.ndarray:
    """Load IQ samples from a ``.sigmf-data`` file as complex64.

    Args:
        data_path: Path to the ``.sigmf-data`` binary file.
        datatype: SigMF ``core:datatype`` string.

    Returns:
        1-D complex64 NumPy array.

    Raises:
        DatasetError: If the datatype is unsupported or the file is missing.
    """
    from rfdf.ml.errors import DatasetError

    if datatype not in _DATATYPE_DTYPE:
        raise DatasetError(
            f"Unsupported SigMF datatype {datatype!r}. Supported: {sorted(_DATATYPE_DTYPE)}"
        )
    if not data_path.exists():
        raise DatasetError(f"SigMF data file not found: {data_path}")

    dtype = _DATATYPE_DTYPE[datatype]
    raw: np.ndarray = np.fromfile(data_path, dtype=dtype)

    if datatype in {"cf32_le", "cf64_le"}:
        return raw.astype(np.complex64)

    # Interleaved real types (ci16_le, ci8).
    if raw.size % 2 != 0:
        raise DatasetError(f"Interleaved {datatype!r} stream has odd sample count {raw.size}")
    real_part = raw[0::2].astype(np.float32)
    imag_part = raw[1::2].astype(np.float32)
    return (real_part + 1j * imag_part).astype(np.complex64)


# ---------------------------------------------------------------------------
# Recording discovery and parsing
# ---------------------------------------------------------------------------


def _find_meta_files(path: Path) -> list[Path]:
    """Return all ``.sigmf-meta`` files under *path* (file or directory).

    Args:
        path: A single ``.sigmf-meta`` file or a directory to search.

    Returns:
        Sorted list of ``.sigmf-meta`` paths.

    Raises:
        DatasetError: If *path* does not exist.
    """
    from rfdf.ml.errors import DatasetError

    if not path.exists():
        raise DatasetError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix != ".sigmf-meta":
            raise DatasetError(f"Expected a .sigmf-meta file or directory, got: {path}")
        return [path]
    return sorted(path.rglob("*.sigmf-meta"))


def _extract_windows(
    meta: dict[str, Any],
    iq: np.ndarray,
    window_samples: int,
    label_field: str,
    default_label: str | None,
) -> list[tuple[np.ndarray, str]]:
    """Extract fixed-length windows from IQ annotated with labels.

    For each annotation that has a ``label_field`` value:

    1. Locate the sample range ``[start, start + length)``.
    2. Chop it into non-overlapping ``window_samples``-sized windows.
    3. Discard the last incomplete window.

    If no annotations carry ``label_field``, the whole IQ array is windowed
    using ``default_label`` (if provided) or skipped.

    Args:
        meta: Parsed SigMF metadata dict.
        iq: 1-D complex64 IQ array for the recording.
        window_samples: Fixed window size.
        label_field: SigMF annotation key to read the label from (e.g.
            ``"core:label"``).
        default_label: Fallback label when no annotated labels are found.

    Returns:
        List of ``(iq_window, label_string)`` pairs.
    """
    annotations = meta.get("annotations", [])
    windows: list[tuple[np.ndarray, str]] = []
    total_samples = len(iq)

    labelled: list[tuple[int, int, str]] = []  # (start, length, label)
    for ann in annotations:
        label = ann.get(label_field)
        if label is None:
            continue
        start = int(ann.get("core:sample_start", 0))
        length = int(ann.get("core:sample_count", total_samples - start))
        labelled.append((start, length, str(label)))

    if not labelled:
        if default_label is not None:
            # Whole file as one label.
            labelled = [(0, total_samples, default_label)]
        else:
            return windows

    for start, length, label in labelled:
        end = min(start + length, total_samples)
        segment = iq[start:end]
        n_windows = len(segment) // window_samples
        for i in range(n_windows):
            chunk = segment[i * window_samples : (i + 1) * window_samples]
            windows.append((chunk.copy(), label))

    return windows


def _session_id(meta_path: Path, meta: dict[str, Any]) -> str:
    """Derive a session identifier for a recording.

    Uses the file stem as the primary key.  This ensures all windows from a
    single recording are assigned to the same split.

    Args:
        meta_path: Path to the ``.sigmf-meta`` file.
        meta: Parsed SigMF metadata.

    Returns:
        A stable string session identifier.
    """
    _ = meta  # reserved for future hw-based disambiguation
    return meta_path.stem


# ---------------------------------------------------------------------------
# Internal Dataset class
# ---------------------------------------------------------------------------


class _CapturedDataset:
    """In-memory ``torch.utils.data.Dataset`` for SigMF captures.

    Attributes:
        class_names: Sorted list of unique label strings found in the recordings.
    """

    def __init__(
        self,
        iq_windows: list[np.ndarray],
        label_ints: list[int],
        class_names: list[str],
        augmentation: AugmentationConfig | None,
        seed: int,
        sample_rate_hz: float,
    ) -> None:
        """Initialise the dataset.

        Args:
            iq_windows: List of 1-D complex64 IQ windows.
            label_ints: Integer label for each window.
            class_names: Sorted class-name list (label_ints index into this).
            augmentation: Optional augmentation applied at ``__getitem__`` time.
            seed: Base seed for per-item augmentation RNGs.
            sample_rate_hz: Sample rate for the augmentation step.
        """
        self._iq = iq_windows
        self._labels = label_ints
        self.class_names = class_names
        self._augmentation = augmentation
        self._seed = seed
        self._sample_rate_hz = sample_rate_hz

    def __len__(self) -> int:
        """Return the number of windows in this split."""
        return len(self._iq)

    def __getitem__(self, idx: int) -> tuple[object, int]:
        """Return ``(iq_tensor, label_int)`` for item *idx*.

        Args:
            idx: Window index.

        Returns:
            ``(iq_tensor, label_int)`` where ``iq_tensor`` is a
            ``torch.Tensor`` of shape ``(window_samples,)`` dtype
            ``torch.complex64``.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required to use the captured dataset; install rfdf[ml]"
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
# Public loader
# ---------------------------------------------------------------------------


def load_captured_dataset(
    path: Path | str,
    *,
    split: Literal["train", "val", "test"] = "train",
    window_samples: int = 4096,
    label_field: str = "core:label",
    default_label: str | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    augmentation: AugmentationConfig | None = None,
    seed: int = 0,
) -> _CapturedDataset:
    """Load SigMF captures from disk and return a split Dataset.

    Splits are performed **by capture session** (i.e. by recording file), not
    by sample.  This prevents a single recording's samples from appearing in
    both training and validation, which would cause data leakage and
    overoptimistic validation accuracy.

    Sessions are sorted alphabetically by stem and then shuffled with *seed*
    before being assigned to splits.  A given seed always produces the same
    assignment.

    Args:
        path: A single ``.sigmf-meta`` file or a directory of ``.sigmf-meta``
            files.  Sub-directories are searched recursively.
        split: Which partition to return: ``"train"``, ``"val"``, or
            ``"test"``.
        window_samples: Fixed-length window size (IQ samples).  Annotations
            shorter than this are discarded; longer annotations are chopped
            into non-overlapping windows of this size.
        label_field: SigMF annotation key that carries the class label (e.g.
            ``"core:label"`` or ``"rfdf:protocol"``).
        default_label: If a recording has no annotations with ``label_field``,
            this label is applied to the whole recording.  ``None`` silently
            skips un-annotated recordings.
        train_frac: Fraction of sessions assigned to the training split.
        val_frac: Fraction of sessions assigned to the validation split.  The
            test fraction is ``1 - train_frac - val_frac``.
        augmentation: Optional rfdf augmentation applied at ``__getitem__`` time.
        seed: Controls the session shuffle.

    Returns:
        A :class:`_CapturedDataset` containing the windows from the requested
        split.

    Raises:
        DatasetError: If *path* does not exist, no ``.sigmf-meta`` files are
            found, or the fractions are invalid.
        ImportError: If ``torch`` is not installed (install ``rfdf[ml]``).
    """
    from rfdf.ml.errors import DatasetError

    path = Path(path)
    if train_frac + val_frac >= 1.0:
        raise DatasetError(f"train_frac + val_frac must be < 1.0 (got {train_frac + val_frac:.3f})")

    meta_paths = _find_meta_files(path)
    if not meta_paths:
        raise DatasetError(f"No .sigmf-meta files found under {path}")

    # ---------- Parse all recordings ----------
    # session_id → list of (iq_window, label_string)
    session_windows: dict[str, list[tuple[np.ndarray, str]]] = {}
    sample_rate_hz: float = 2e6  # default fallback

    for meta_path in meta_paths:
        try:
            meta = _load_sigmf_meta(meta_path)
        except DatasetError as exc:
            logger.warning("Skipping %s: %s", meta_path, exc)
            continue

        global_meta: dict[str, Any] = meta.get("global", {})
        datatype = str(global_meta.get("core:datatype", "cf32_le"))
        sr = float(global_meta.get("core:sample_rate", 2e6))
        sample_rate_hz = sr

        # Resolve sibling data file.
        if meta_path.suffix == ".sigmf-meta":
            data_path = meta_path.with_suffix(".sigmf-data")
        else:
            data_path = meta_path.parent / (meta_path.stem + ".sigmf-data")

        try:
            iq = _load_sigmf_iq(data_path, datatype)
        except DatasetError as exc:
            logger.warning("Skipping %s: %s", data_path, exc)
            continue

        sess = _session_id(meta_path, meta)
        windows = _extract_windows(meta, iq, window_samples, label_field, default_label)
        if windows:
            existing = session_windows.get(sess, [])
            existing.extend(windows)
            session_windows[sess] = existing

    if not session_windows:
        raise DatasetError(
            f"No labelled windows found in {path}. Check label_field and default_label arguments."
        )

    # ---------- Build class vocabulary ----------
    all_labels: set[str] = set()
    for wins in session_windows.values():
        for _, lbl in wins:
            all_labels.add(lbl)
    class_names = sorted(all_labels)
    label_to_int = {lbl: i for i, lbl in enumerate(class_names)}

    # ---------- Split sessions by session ID ----------
    session_ids = sorted(session_windows.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(session_ids)

    n = len(session_ids)
    n_train = max(1, round(n * train_frac))
    n_val = max(0, min(round(n * val_frac), n - n_train))
    n_test = n - n_train - n_val

    if split == "train":
        chosen = session_ids[:n_train]
    elif split == "val":
        chosen = session_ids[n_train : n_train + n_val]
    else:  # test
        chosen = session_ids[n_train + n_val :]

    # ---------- Collect windows for the chosen sessions ----------
    iq_windows: list[np.ndarray] = []
    label_ints: list[int] = []
    for sess in chosen:
        for iq_win, lbl in session_windows[sess]:
            iq_windows.append(iq_win)
            label_ints.append(label_to_int[lbl])

    logger.info(
        "load_captured_dataset: %s split — %d sessions → %d windows "
        "(%d classes). train/val/test session counts: %d/%d/%d",
        split,
        len(chosen),
        len(iq_windows),
        len(class_names),
        n_train,
        n_val,
        n_test,
    )

    return _CapturedDataset(
        iq_windows=iq_windows,
        label_ints=label_ints,
        class_names=class_names,
        augmentation=augmentation,
        seed=seed,
        sample_rate_hz=sample_rate_hz,
    )

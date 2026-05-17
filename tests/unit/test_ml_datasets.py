"""Unit tests for rfdf.ml.datasets: synthetic, RadioML, and captured loaders.

All tests in this file require torch (and torchsig for the synthetic tests).
The ``pytest.importorskip("torch")`` call at the top makes the whole module
skip cleanly in the base ``test-base`` / ``coverage`` CI jobs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Synthetic dataset tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_modulation_dataset():
    """Return a 3-class, 4-item modulation dataset (tiny for speed)."""
    torchsig = pytest.importorskip("torchsig")  # noqa: F841
    from rfdf.ml.datasets.synthetic import make_modulation_dataset

    return make_modulation_dataset(
        num_signals=3,
        num_samples_per_signal=256,
        impairments="clean",
        seed=0,
    )


@pytest.fixture(scope="module")
def tiny_protocol_dataset():
    """Return a 2-protocol, 2-item-per-protocol protocol dataset."""
    pytest.importorskip("torchsig")
    from rfdf.ml.datasets.synthetic import make_protocol_dataset

    return make_protocol_dataset(
        protocols=["lora", "noise"],
        num_samples_per_protocol=2,
        num_iq_samples=256,
        impairments="clean",
        seed=0,
    )


class TestModulationDataset:
    """Tests for make_modulation_dataset."""

    def test_length(self, tiny_modulation_dataset) -> None:
        """Dataset length equals num_signals."""
        assert len(tiny_modulation_dataset) == 3

    def test_item_is_tensor_and_int(self, tiny_modulation_dataset) -> None:
        """Each item is a (complex64 tensor, int) pair."""
        iq, label = tiny_modulation_dataset[0]
        assert isinstance(iq, torch.Tensor)
        assert iq.dtype == torch.complex64
        assert isinstance(label, int)

    def test_iq_shape(self, tiny_modulation_dataset) -> None:
        """IQ tensor has the expected 1-D shape."""
        iq, _ = tiny_modulation_dataset[0]
        assert iq.ndim == 1
        assert iq.shape[0] == 256

    def test_label_in_range(self, tiny_modulation_dataset) -> None:
        """All labels are in [0, num_signals)."""
        n_classes = len(tiny_modulation_dataset.class_names)
        for i in range(len(tiny_modulation_dataset)):
            _, label = tiny_modulation_dataset[i]
            assert 0 <= label < n_classes, f"label {label} out of range"

    def test_class_names_non_empty(self, tiny_modulation_dataset) -> None:
        """class_names is a non-empty list of strings."""
        names = tiny_modulation_dataset.class_names
        assert isinstance(names, list)
        assert len(names) == 3
        assert all(isinstance(n, str) for n in names)

    def test_determinism(self) -> None:
        """Same seed produces the same first item."""
        pytest.importorskip("torchsig")
        from rfdf.ml.datasets.synthetic import make_modulation_dataset

        ds1 = make_modulation_dataset(num_signals=3, num_samples_per_signal=128, seed=7)
        ds2 = make_modulation_dataset(num_signals=3, num_samples_per_signal=128, seed=7)
        iq1, lbl1 = ds1[0]
        iq2, lbl2 = ds2[0]
        assert lbl1 == lbl2
        assert torch.allclose(iq1, iq2), "Different seeds should produce same output for same seed"

    def test_different_seeds_different_output(self) -> None:
        """Different seeds produce different IQ content."""
        pytest.importorskip("torchsig")
        from rfdf.ml.datasets.synthetic import make_modulation_dataset

        ds1 = make_modulation_dataset(num_signals=2, num_samples_per_signal=128, seed=0)
        ds2 = make_modulation_dataset(num_signals=2, num_samples_per_signal=128, seed=99)
        _iq1, _ = ds1[0]
        _iq2, _ = ds2[0]
        # The class_names should be the same, but IQ content can differ.
        assert ds1.class_names == ds2.class_names

    def test_augmentation_changes_output(self) -> None:
        """Augmentation on retrieval changes the tensor content."""
        pytest.importorskip("torchsig")
        from rfdf.ml.datasets.augmentation import AugmentationConfig
        from rfdf.ml.datasets.synthetic import make_modulation_dataset

        ds_no_aug = make_modulation_dataset(num_signals=2, num_samples_per_signal=256, seed=1)
        ds_aug = make_modulation_dataset(
            num_signals=2,
            num_samples_per_signal=256,
            augmentation=AugmentationConfig(add_awgn=(5.0, 5.0)),
            seed=1,
        )
        iq_plain, _ = ds_no_aug[0]
        iq_augmented, _ = ds_aug[0]
        assert not torch.allclose(iq_plain, iq_augmented), "Augmentation had no effect"

    def test_too_many_signals_raises(self) -> None:
        """Requesting more signals than the catalogue raises DatasetError."""
        pytest.importorskip("torchsig")
        from rfdf.ml.datasets.synthetic import make_modulation_dataset
        from rfdf.ml.errors import DatasetError

        with pytest.raises(DatasetError):
            make_modulation_dataset(num_signals=99999)


class TestProtocolDataset:
    """Tests for make_protocol_dataset."""

    def test_length(self, tiny_protocol_dataset) -> None:
        """Dataset length equals num_protocols x num_samples_per_protocol."""
        assert len(tiny_protocol_dataset) == 4  # 2 protocols x 2 samples

    def test_class_names(self, tiny_protocol_dataset) -> None:
        """class_names matches the requested protocols."""
        assert tiny_protocol_dataset.class_names == ["lora", "noise"]

    def test_item_types(self, tiny_protocol_dataset) -> None:
        """Each item is a (complex64 tensor, int) pair."""
        iq, label = tiny_protocol_dataset[0]
        assert isinstance(iq, torch.Tensor)
        assert iq.dtype == torch.complex64
        assert isinstance(label, int)

    def test_unknown_protocol_raises(self) -> None:
        """Unknown protocol name raises DatasetError."""
        pytest.importorskip("torchsig")
        from rfdf.ml.datasets.synthetic import make_protocol_dataset
        from rfdf.ml.errors import DatasetError

        with pytest.raises(DatasetError, match="Unknown protocols"):
            make_protocol_dataset(protocols=["lora", "notaprotocol"])


# ---------------------------------------------------------------------------
# RadioML fixture: tiny synthetic HDF5 in RadioML on-disk shape
# ---------------------------------------------------------------------------


@pytest.fixture()
def radioml_hdf5(tmp_path: Path) -> Path:
    """Write a tiny (6-sample) HDF5 file in RadioML 2018.01A format.

    Shape: (6, 2, 1024) X, (6, 24) Y one-hot, (6,) Z SNR.
    Classes 0, 1, 2 at SNR 10, 20, -10 respectively (repeated twice).
    """
    pytest.importorskip("h5py")
    import h5py  # type: ignore[import-untyped]

    n = 6
    x = np.random.default_rng(0).standard_normal((n, 2, 1024)).astype(np.float32)
    y = np.zeros((n, 24), dtype=np.float32)
    z_snrs = [10, 20, -10, 10, 20, -10]
    class_ids = [0, 1, 2, 0, 1, 2]
    for i, cls in enumerate(class_ids):
        y[i, cls] = 1.0
    z = np.array(z_snrs, dtype=np.float32)

    h5_path = tmp_path / "radioml_fixture.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=x)
        f.create_dataset("Y", data=y)
        f.create_dataset("Z", data=z)
    return h5_path


class TestRadioMLDataset:
    """Tests for the RadioML 2018.01A loader (against a tiny synthetic HDF5)."""

    def test_load_full(self, radioml_hdf5: Path) -> None:
        """Loading without filters returns all 6 items."""
        pytest.importorskip("h5py")
        from rfdf.ml.datasets.radioml import load_radioml

        ds = load_radioml(hdf5_path=radioml_hdf5)
        assert len(ds) == 6

    def test_item_tensor_shape(self, radioml_hdf5: Path) -> None:
        """Each item's IQ tensor has shape (1024,) and dtype complex64."""
        pytest.importorskip("h5py")
        from rfdf.ml.datasets.radioml import load_radioml

        ds = load_radioml(hdf5_path=radioml_hdf5)
        iq, label = ds[0]
        assert isinstance(iq, torch.Tensor)
        assert iq.dtype == torch.complex64
        assert iq.shape == (1024,)
        assert isinstance(label, int)

    def test_snr_filter(self, radioml_hdf5: Path) -> None:
        """snr_filter=[-10] returns only items at SNR -10."""
        pytest.importorskip("h5py")
        from rfdf.ml.datasets.radioml import load_radioml

        ds = load_radioml(hdf5_path=radioml_hdf5, snr_filter=[-10])
        assert len(ds) == 2  # two items at SNR -10

    def test_class_filter(self, radioml_hdf5: Path) -> None:
        """class_filter restricts to the requested classes."""
        pytest.importorskip("h5py")
        from rfdf.ml.datasets.radioml import (
            RADIOML_CLASS_NAMES,
            load_radioml,
        )

        cls0 = RADIOML_CLASS_NAMES[0]
        ds = load_radioml(hdf5_path=radioml_hdf5, class_filter=[cls0])
        assert len(ds) == 2  # two OOK items

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing HDF5 without download=True raises DatasetError."""
        from rfdf.ml.datasets.radioml import load_radioml
        from rfdf.ml.errors import DatasetError

        with pytest.raises(DatasetError, match="not found"):
            load_radioml(hdf5_path=tmp_path / "nonexistent.hdf5")

    def test_class_names_length(self, radioml_hdf5: Path) -> None:
        """The dataset exposes all 24 class names."""
        pytest.importorskip("h5py")
        from rfdf.ml.datasets.radioml import load_radioml

        ds = load_radioml(hdf5_path=radioml_hdf5)
        assert len(ds.class_names) == 24

    def test_augmentation_changes_output(self, radioml_hdf5: Path) -> None:
        """Augmentation at getitem time changes the tensor content."""
        pytest.importorskip("h5py")
        from rfdf.ml.datasets.augmentation import AugmentationConfig
        from rfdf.ml.datasets.radioml import load_radioml

        ds_no_aug = load_radioml(hdf5_path=radioml_hdf5)
        ds_aug = load_radioml(
            hdf5_path=radioml_hdf5,
            augmentation=AugmentationConfig(add_awgn=(0.0, 0.0)),
        )
        iq_plain, _ = ds_no_aug[0]
        iq_aug, _ = ds_aug[0]
        assert not torch.allclose(iq_plain, iq_aug)

    def test_download_false_no_internet(self, tmp_path: Path) -> None:
        """download=False raises DatasetError (no net access in CI)."""
        from rfdf.ml.datasets.radioml import load_radioml
        from rfdf.ml.errors import DatasetError

        with pytest.raises(DatasetError, match="download=True"):
            load_radioml(hdf5_path=tmp_path / "absent.hdf5", download=False)


# ---------------------------------------------------------------------------
# Captured dataset: SigMF fixture helpers
# ---------------------------------------------------------------------------


def _write_sigmf_pair(
    directory: Path,
    stem: str,
    iq: np.ndarray,
    label: str,
    sample_rate: float = 2e6,
) -> Path:
    """Write a minimal SigMF meta + data pair to *directory*.

    Args:
        directory: Destination directory.
        stem: Base filename (no extension).
        iq: 1-D complex64 IQ array.
        label: Label string stored in the first annotation's ``core:label``.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the written ``.sigmf-meta`` file.
    """
    # Write interleaved float32 I/Q.
    data_path = directory / f"{stem}.sigmf-data"
    interleaved = np.empty(2 * len(iq), dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    interleaved.tofile(data_path)

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:hw": "test-fixture",
            "core:version": "1.0.0",
        },
        "captures": [{"core:sample_start": 0, "core:frequency": 868e6}],
        "annotations": [
            {
                "core:sample_start": 0,
                "core:sample_count": len(iq),
                "core:label": label,
            }
        ],
    }
    meta_path = directory / f"{stem}.sigmf-meta"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


class TestCapturedDataset:
    """Tests for load_captured_dataset against small SigMF fixtures."""

    @pytest.fixture()
    def sigmf_dir(self, tmp_path: Path) -> Path:
        """Create 4 SigMF recordings across 2 classes: 'wifi' and 'lte'."""
        rng = np.random.default_rng(0)
        n = 8192  # enough for ≥2 windows of 4096
        for i in range(2):
            iq = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            iq = iq.astype(np.complex64)
            _write_sigmf_pair(tmp_path, f"wifi_{i:02d}", iq, "wifi")
        for i in range(2):
            iq = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            iq = iq.astype(np.complex64)
            _write_sigmf_pair(tmp_path, f"lte_{i:02d}", iq, "lte")
        return tmp_path

    def test_train_dataset_non_empty(self, sigmf_dir: Path) -> None:
        """Training split returns a non-empty dataset."""
        from rfdf.ml.datasets.captured import load_captured_dataset

        ds = load_captured_dataset(sigmf_dir, split="train", window_samples=4096, seed=0)
        assert len(ds) > 0

    def test_item_type(self, sigmf_dir: Path) -> None:
        """Each item is a (complex64 tensor, int) pair."""
        from rfdf.ml.datasets.captured import load_captured_dataset

        ds = load_captured_dataset(sigmf_dir, split="train", window_samples=4096, seed=0)
        iq, label = ds[0]
        assert isinstance(iq, torch.Tensor)
        assert iq.dtype == torch.complex64
        assert isinstance(label, int)

    def test_iq_shape(self, sigmf_dir: Path) -> None:
        """IQ tensor has shape (window_samples,)."""
        from rfdf.ml.datasets.captured import load_captured_dataset

        ds = load_captured_dataset(sigmf_dir, split="train", window_samples=4096, seed=0)
        iq, _ = ds[0]
        assert iq.shape == (4096,)

    def test_class_names(self, sigmf_dir: Path) -> None:
        """class_names contains both label strings."""
        from rfdf.ml.datasets.captured import load_captured_dataset

        ds = load_captured_dataset(sigmf_dir, split="train", window_samples=4096, seed=0)
        assert "wifi" in ds.class_names
        assert "lte" in ds.class_names

    def test_by_session_split_no_leakage(self, tmp_path: Path) -> None:
        """Every recording's windows land in exactly one split (no by-sample leakage).

        Nine recordings each get a unique label and several windows. A by-session
        split keeps all of a recording's windows together, so each label appears in
        exactly one of {train, val, test}; a by-sample split would scatter a
        recording's windows and leak its label across splits.
        """
        from rfdf.ml.datasets.captured import load_captured_dataset

        rng = np.random.default_rng(0)
        n_sessions = 9
        window_samples = 512
        iq_len = window_samples * 4  # several windows per recording
        for k in range(n_sessions):
            iq = (rng.standard_normal(iq_len) + 1j * rng.standard_normal(iq_len)).astype(
                np.complex64
            )
            _write_sigmf_pair(tmp_path, f"sess_{k:02d}", iq, f"label_{k}")

        splits = {
            name: load_captured_dataset(
                tmp_path, split=name, window_samples=window_samples, seed=42
            )
            for name in ("train", "val", "test")
        }
        # The fixture is large enough that every split is actually exercised.
        for name, ds in splits.items():
            assert len(ds) > 0, f"{name} split is empty"

        # No leakage: each recording's label appears in exactly one split.
        label_to_splits: dict[str, set[str]] = {}
        for name, ds in splits.items():
            for label_int in ds._labels:
                label_to_splits.setdefault(ds.class_names[label_int], set()).add(name)
        assert len(label_to_splits) == n_sessions, "every recording's label should appear"
        for label, where in sorted(label_to_splits.items()):
            assert len(where) == 1, f"{label} leaked across splits {sorted(where)}"

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """Non-existent path raises DatasetError."""
        from rfdf.ml.datasets.captured import load_captured_dataset
        from rfdf.ml.errors import DatasetError

        with pytest.raises(DatasetError, match="does not exist"):
            load_captured_dataset(tmp_path / "ghost", split="train", window_samples=512)

    def test_single_file_input(self, tmp_path: Path) -> None:
        """A single .sigmf-meta file path is accepted."""
        from rfdf.ml.datasets.captured import load_captured_dataset

        rng = np.random.default_rng(7)
        iq = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(np.complex64)
        meta_path = _write_sigmf_pair(tmp_path, "single", iq, "am")
        ds = load_captured_dataset(meta_path, split="train", window_samples=4096, seed=0)
        assert len(ds) > 0
        assert "am" in ds.class_names

    def test_invalid_fractions_raise(self, sigmf_dir: Path) -> None:
        """train_frac + val_frac >= 1.0 raises DatasetError."""
        from rfdf.ml.datasets.captured import load_captured_dataset
        from rfdf.ml.errors import DatasetError

        with pytest.raises(DatasetError, match="train_frac"):
            load_captured_dataset(
                sigmf_dir,
                split="train",
                window_samples=4096,
                train_frac=0.8,
                val_frac=0.3,
            )

    def test_augmentation_applied(self, sigmf_dir: Path) -> None:
        """Augmentation changes the tensor returned at getitem time."""
        from rfdf.ml.datasets.augmentation import AugmentationConfig
        from rfdf.ml.datasets.captured import load_captured_dataset

        ds_plain = load_captured_dataset(sigmf_dir, split="train", window_samples=4096, seed=0)
        ds_aug = load_captured_dataset(
            sigmf_dir,
            split="train",
            window_samples=4096,
            seed=0,
            augmentation=AugmentationConfig(add_awgn=(0.0, 0.0)),
        )
        iq_plain, _ = ds_plain[0]
        iq_aug, _ = ds_aug[0]
        assert not torch.allclose(iq_plain, iq_aug)

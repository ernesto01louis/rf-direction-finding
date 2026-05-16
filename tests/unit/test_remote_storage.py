"""Unit tests for the RemoteStorage protocol and implementations.

LocalFilesystemStorage is tested fully (pure stdlib, no mocks needed).
RunpodVolumeStorage is tested with the runpod SDK mocked.
ModalVolumeStorage is tested with the modal SDK mocked.
VastAiStorage is tested with the vastai SDK mocked.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from rfdf.backends.compute._remote_storage import (
    LocalFilesystemStorage,
    ModalVolumeStorage,
    RemoteStorage,
    RunpodVolumeStorage,
    VastAiStorage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# RemoteStorage Protocol — isinstance checks
# ---------------------------------------------------------------------------


def test_local_is_remote_storage(tmp_path: Path) -> None:
    store = LocalFilesystemStorage(tmp_path)
    assert isinstance(store, RemoteStorage)


def test_runpod_volume_is_remote_storage() -> None:
    store = RunpodVolumeStorage(volume_id="vol-abc")
    assert isinstance(store, RemoteStorage)


def test_modal_volume_is_remote_storage() -> None:
    store = ModalVolumeStorage(volume_name="rfdf-test")
    assert isinstance(store, RemoteStorage)


def test_vastai_storage_is_remote_storage() -> None:
    store = VastAiStorage()
    assert isinstance(store, RemoteStorage)


# ---------------------------------------------------------------------------
# LocalFilesystemStorage
# ---------------------------------------------------------------------------


class TestLocalFilesystemStorage:
    def test_name(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path)
        assert store.name == "local"

    def test_root_created_on_init(self, tmp_path: Path) -> None:
        root = tmp_path / "deep" / "store"
        store = LocalFilesystemStorage(root)
        assert store.root.is_dir()

    def test_put_get_round_trip(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        src = tmp_path / "data.bin"
        src.write_bytes(b"\x00\x01\x02\x03")

        returned_key = _run(store.put(src, "blobs/data.bin"))
        assert returned_key == "blobs/data.bin"

        dest = tmp_path / "restored.bin"
        _run(store.get("blobs/data.bin", dest))
        assert dest.read_bytes() == b"\x00\x01\x02\x03"

    def test_put_creates_parent_dirs(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        src = tmp_path / "file.txt"
        src.write_text("hello")
        _run(store.put(src, "a/b/c/file.txt"))
        assert (store.root / "a" / "b" / "c" / "file.txt").is_file()

    def test_put_raises_if_source_missing(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            _run(store.put(tmp_path / "nonexistent.bin", "x.bin"))

    def test_get_raises_if_key_missing(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        with pytest.raises(KeyError):
            _run(store.get("no-such-key", tmp_path / "out.bin"))

    def test_exists_true(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        src = tmp_path / "f.txt"
        src.write_text("x")
        _run(store.put(src, "f.txt"))
        assert _run(store.exists("f.txt")) is True

    def test_exists_false(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        assert _run(store.exists("definitely-not-there")) is False

    def test_list_returns_all_keys(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        for name in ("alpha.txt", "beta.txt", "sub/gamma.txt"):
            src = tmp_path / "src.txt"
            src.write_text(name)
            _run(store.put(src, name))
        keys = _run(store.list())
        assert sorted(keys) == ["alpha.txt", "beta.txt", "sub/gamma.txt"]

    def test_list_prefix_filter(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        for name in ("models/a.pt", "models/b.pt", "data/x.h5"):
            src = tmp_path / "f"
            src.write_text(name)
            _run(store.put(src, name))
        keys = _run(store.list(prefix="models/"))
        assert sorted(keys) == ["models/a.pt", "models/b.pt"]

    def test_list_empty_store(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        assert _run(store.list()) == []

    def test_list_empty_prefix_returns_all(self, tmp_path: Path) -> None:
        store = LocalFilesystemStorage(tmp_path / "store")
        src = tmp_path / "x.txt"
        src.write_text("y")
        _run(store.put(src, "x.txt"))
        keys = _run(store.list(""))
        assert "x.txt" in keys


# ---------------------------------------------------------------------------
# RunpodVolumeStorage — mocked SDK
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_runpod_in_modules(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a minimal fake runpod module into sys.modules."""
    fake = types.ModuleType("runpod")
    fake.api_key = "test-key"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod", fake)
    return fake


class TestRunpodVolumeStorage:
    def test_name(self) -> None:
        store = RunpodVolumeStorage(volume_id="vol-1")
        assert store.name == "runpod-volume"

    def test_put_adds_key(
        self,
        fake_runpod_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = RunpodVolumeStorage(volume_id="vol-1")
        src = tmp_path / "data.bin"
        src.write_bytes(b"hello")
        returned = _run(store.put(src, "blobs/data.bin"))
        assert returned == "blobs/data.bin"
        assert _run(store.exists("blobs/data.bin")) is True

    def test_put_raises_if_source_missing(
        self,
        fake_runpod_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = RunpodVolumeStorage(volume_id="vol-1")
        with pytest.raises(FileNotFoundError):
            _run(store.put(tmp_path / "missing.bin", "x.bin"))

    def test_exists_false_before_put(
        self,
        fake_runpod_in_modules: types.ModuleType,
    ) -> None:
        store = RunpodVolumeStorage(volume_id="vol-1")
        assert _run(store.exists("not-uploaded")) is False

    def test_list_prefix_filter(
        self,
        fake_runpod_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = RunpodVolumeStorage(volume_id="vol-1")
        for name in ("models/a.pt", "models/b.pt", "data/x.h5"):
            src = tmp_path / "f"
            src.write_text(name)
            _run(store.put(src, name))
        keys = _run(store.list("models/"))
        assert sorted(keys) == ["models/a.pt", "models/b.pt"]

    def test_list_empty_prefix(
        self,
        fake_runpod_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = RunpodVolumeStorage(volume_id="vol-1")
        src = tmp_path / "x.txt"
        src.write_text("y")
        _run(store.put(src, "x.txt"))
        keys = _run(store.list())
        assert "x.txt" in keys

    def test_get_raises_for_unknown_key(
        self,
        fake_runpod_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = RunpodVolumeStorage(volume_id="vol-1")
        with pytest.raises(KeyError):
            _run(store.get("nonexistent-key", tmp_path / "out.bin"))

    def test_get_raises_not_implemented_for_known_key(
        self,
        fake_runpod_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        """get() on a known key documents the SSH-transfer stub."""
        store = RunpodVolumeStorage(volume_id="vol-1")
        src = tmp_path / "f.bin"
        src.write_bytes(b"data")
        _run(store.put(src, "f.bin"))
        with pytest.raises(NotImplementedError):
            _run(store.get("f.bin", tmp_path / "restored.bin"))

    def test_sdk_missing_raises_import_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """RunpodVolumeStorage methods raise ImportError when SDK absent."""
        monkeypatch.delitem(sys.modules, "runpod", raising=False)

        # Reload remote storage to clear any cached import.
        import importlib

        import rfdf.backends.compute._remote_storage as mod

        importlib.reload(mod)
        store = mod.RunpodVolumeStorage(volume_id="vol-1")
        src = tmp_path / "f.bin"
        src.write_bytes(b"data")

        # Patch _require_runpod to raise ImportError.
        def _fail() -> Any:
            raise ImportError(
                "RunpodVolumeStorage requires the runpod SDK; "
                "install it with:  pip install rfdf[compute-runpod]"
            )

        monkeypatch.setattr(mod, "_require_runpod", _fail)

        with pytest.raises(ImportError, match="rfdf\\[compute-runpod\\]"):
            _run(store.put(src, "k"))


# ---------------------------------------------------------------------------
# ModalVolumeStorage — mocked SDK
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_modal_in_modules(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a minimal fake modal module into sys.modules."""
    import types as _types

    fake = _types.ModuleType("modal")
    fake_cfg = _types.ModuleType("modal.config")

    class _Cfg:
        def get(self, k: str) -> str:
            return "tok"

    fake_cfg.config = _Cfg()  # type: ignore[attr-defined]
    fake.config = fake_cfg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setitem(sys.modules, "modal.config", fake_cfg)  # type: ignore[arg-type]
    return fake


class TestModalVolumeStorage:
    def test_name(self) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        assert store.name == "modal-volume"

    def test_put_adds_key(
        self,
        fake_modal_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        src = tmp_path / "data.bin"
        src.write_bytes(b"hello")
        returned = _run(store.put(src, "blobs/data.bin"))
        assert returned == "blobs/data.bin"
        assert _run(store.exists("blobs/data.bin")) is True

    def test_put_raises_if_source_missing(
        self,
        fake_modal_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        with pytest.raises(FileNotFoundError):
            _run(store.put(tmp_path / "missing.bin", "x.bin"))

    def test_exists_false_before_put(
        self,
        fake_modal_in_modules: types.ModuleType,
    ) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        assert _run(store.exists("not-uploaded")) is False

    def test_list_prefix_filter(
        self,
        fake_modal_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        for name in ("models/a.pt", "models/b.pt", "data/x.h5"):
            src = tmp_path / "f"
            src.write_text(name)
            _run(store.put(src, name))
        keys = _run(store.list("models/"))
        assert sorted(keys) == ["models/a.pt", "models/b.pt"]

    def test_list_empty_prefix(
        self,
        fake_modal_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        src = tmp_path / "x.txt"
        src.write_text("y")
        _run(store.put(src, "x.txt"))
        keys = _run(store.list())
        assert "x.txt" in keys

    def test_get_raises_for_unknown_key(
        self,
        fake_modal_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        with pytest.raises(KeyError):
            _run(store.get("nonexistent-key", tmp_path / "out.bin"))

    def test_get_raises_not_implemented_for_known_key(
        self,
        fake_modal_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        """get() on a known key documents the Volume.read_file stub."""
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        src = tmp_path / "f.bin"
        src.write_bytes(b"data")
        _run(store.put(src, "f.bin"))
        with pytest.raises(NotImplementedError):
            _run(store.get("f.bin", tmp_path / "restored.bin"))

    def test_sdk_missing_raises_import_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """ModalVolumeStorage methods raise ImportError when SDK absent."""
        monkeypatch.delitem(sys.modules, "modal", raising=False)
        monkeypatch.delitem(sys.modules, "modal.config", raising=False)

        import importlib

        import rfdf.backends.compute._remote_storage as mod

        importlib.reload(mod)
        store = mod.ModalVolumeStorage(volume_name="rfdf-vol")
        src = tmp_path / "f.bin"
        src.write_bytes(b"data")

        def _fail() -> Any:
            raise ImportError(
                "ModalVolumeStorage requires the modal SDK; "
                "install it with:  pip install rfdf[compute-modal]"
            )

        monkeypatch.setattr(mod, "_require_modal", _fail)

        with pytest.raises(ImportError, match="rfdf\\[compute-modal\\]"):
            _run(store.put(src, "k"))

    def test_default_mount_path(self) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol")
        assert store.mount_path == "/modal-volume"

    def test_custom_mount_path(self) -> None:
        store = ModalVolumeStorage(volume_name="rfdf-vol", mount_path="/data")
        assert store.mount_path == "/data"


# ---------------------------------------------------------------------------
# VastAiStorage — mocked SDK
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_vastai_in_modules(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a minimal fake vastai module into sys.modules."""
    fake = types.ModuleType("vastai")
    monkeypatch.setitem(sys.modules, "vastai", fake)
    return fake


class TestVastAiStorage:
    def test_name(self) -> None:
        store = VastAiStorage()
        assert store.name == "vastai-instance-disk"

    def test_default_attributes(self) -> None:
        store = VastAiStorage()
        assert store.instance_disk_gb == 10.0
        assert store.mount_path == "/workspace"

    def test_custom_attributes(self) -> None:
        store = VastAiStorage(instance_disk_gb=50.0, mount_path="/data")
        assert store.instance_disk_gb == 50.0
        assert store.mount_path == "/data"

    def test_put_adds_key(
        self,
        fake_vastai_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = VastAiStorage()
        src = tmp_path / "data.bin"
        src.write_bytes(b"hello")
        returned = _run(store.put(src, "blobs/data.bin"))
        assert returned == "blobs/data.bin"
        assert _run(store.exists("blobs/data.bin")) is True

    def test_put_raises_if_source_missing(
        self,
        fake_vastai_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = VastAiStorage()
        with pytest.raises(FileNotFoundError):
            _run(store.put(tmp_path / "missing.bin", "x.bin"))

    def test_exists_false_before_put(
        self,
        fake_vastai_in_modules: types.ModuleType,
    ) -> None:
        store = VastAiStorage()
        assert _run(store.exists("not-uploaded")) is False

    def test_list_prefix_filter(
        self,
        fake_vastai_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = VastAiStorage()
        for name in ("models/a.pt", "models/b.pt", "data/x.h5"):
            src = tmp_path / "f"
            src.write_text(name)
            _run(store.put(src, name))
        keys = _run(store.list("models/"))
        assert sorted(keys) == ["models/a.pt", "models/b.pt"]

    def test_list_empty_prefix(
        self,
        fake_vastai_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = VastAiStorage()
        src = tmp_path / "x.txt"
        src.write_text("y")
        _run(store.put(src, "x.txt"))
        keys = _run(store.list())
        assert "x.txt" in keys

    def test_get_raises_for_unknown_key(
        self,
        fake_vastai_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        store = VastAiStorage()
        with pytest.raises(KeyError):
            _run(store.get("nonexistent-key", tmp_path / "out.bin"))

    def test_get_raises_not_implemented_for_known_key(
        self,
        fake_vastai_in_modules: types.ModuleType,
        tmp_path: Path,
    ) -> None:
        """get() on a known key documents the SSH-transfer stub."""
        store = VastAiStorage()
        src = tmp_path / "f.bin"
        src.write_bytes(b"data")
        _run(store.put(src, "f.bin"))
        with pytest.raises(NotImplementedError):
            _run(store.get("f.bin", tmp_path / "restored.bin"))

    def test_sdk_missing_raises_import_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """VastAiStorage methods raise ImportError when SDK absent."""
        monkeypatch.delitem(sys.modules, "vastai", raising=False)

        import importlib

        import rfdf.backends.compute._remote_storage as mod

        importlib.reload(mod)
        store = mod.VastAiStorage()
        src = tmp_path / "f.bin"
        src.write_bytes(b"data")

        def _fail() -> Any:
            raise ImportError(
                "VastAiStorage requires the vastai SDK; "
                "install it with:  pip install rfdf[compute-vastai]"
            )

        monkeypatch.setattr(mod, "_require_vastai", _fail)

        with pytest.raises(ImportError, match="rfdf\\[compute-vastai\\]"):
            _run(store.put(src, "k"))

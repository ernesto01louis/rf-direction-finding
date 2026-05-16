"""``RemoteStorage`` Protocol and built-in implementations.

A ``RemoteStorage`` provides a key-value store for datasets and artifacts that
persist between compute runs so that large files are not re-uploaded with every
job submission.  Two implementations ship in-tree:

* :class:`LocalFilesystemStorage` — backed by a plain directory on disk; zero
  dependencies, fully testable without any provider account.
* :class:`RunpodVolumeStorage` — backed by a RunPod Network Volume; lazy
  ``runpod`` import so the module is importable without the SDK.

Both satisfy the :class:`RemoteStorage` ``@runtime_checkable`` Protocol.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RemoteStorage(Protocol):
    """Async key-value store for large datasets and job artifacts.

    Keys are opaque strings (e.g. ``"datasets/train.h5"``).  Implementations
    are responsible for any namespace / prefix conventions they need.  Callers
    should treat keys as path-like strings without assuming a filesystem layout.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier (e.g. ``"local"`` or ``"runpod-volume"``)."""
        ...

    async def put(self, local_path: Path, remote_key: str) -> str:
        """Upload *local_path* and associate it with *remote_key*.

        Args:
            local_path: Source file on the local filesystem.
            remote_key: Destination key in the remote store.

        Returns:
            The canonical remote key under which the file was stored (may differ
            from *remote_key* if the backend normalises it).

        Raises:
            FileNotFoundError: If *local_path* does not exist.
        """
        ...

    async def get(self, remote_key: str, local_path: Path) -> None:
        """Download the object at *remote_key* to *local_path*.

        Args:
            remote_key: Key of the object to retrieve.
            local_path: Destination file path (parent directory must exist).

        Raises:
            KeyError: If *remote_key* does not exist in the store.
        """
        ...

    async def exists(self, remote_key: str) -> bool:
        """Return ``True`` if *remote_key* exists in the store.

        Args:
            remote_key: Key to probe.
        """
        ...

    async def list(self, prefix: str = "") -> list[str]:
        """Return all keys that start with *prefix* (empty → all keys).

        Args:
            prefix: Optional key prefix filter.

        Returns:
            Sorted list of matching keys.
        """
        ...


# ---------------------------------------------------------------------------
# LocalFilesystemStorage
# ---------------------------------------------------------------------------


class LocalFilesystemStorage:
    """Implements :class:`RemoteStorage` using a directory on the local disk.

    The directory is created on first use.  Keys map to relative paths inside
    that directory — path separators in a key create sub-directories.  The
    implementation is pure stdlib with no network calls, making it ideal as a
    development fallback and for testing without a provider account.

    Attributes:
        root: The root directory backing the store.
    """

    def __init__(self, root: Path) -> None:
        """Initialise the store, creating *root* if it does not exist.

        Args:
            root: Directory under which all objects are stored.
        """
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Always ``"local"``."""
        return "local"

    async def put(self, local_path: Path, remote_key: str) -> str:
        """Copy *local_path* into the store under *remote_key*.

        Args:
            local_path: Source file on the local filesystem.
            remote_key: Destination key (relative path within the store root).

        Returns:
            The *remote_key* unchanged (no normalisation applied).

        Raises:
            FileNotFoundError: If *local_path* does not exist.
        """
        if not local_path.exists():
            raise FileNotFoundError(f"LocalFilesystemStorage.put: {local_path} not found")
        dest = self.root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, str(local_path), str(dest))
        return remote_key

    async def get(self, remote_key: str, local_path: Path) -> None:
        """Copy the stored object at *remote_key* to *local_path*.

        Args:
            remote_key: Key of the object to retrieve.
            local_path: Destination path; parent directory must exist.

        Raises:
            KeyError: If *remote_key* does not exist in the store.
        """
        src = self.root / remote_key
        if not src.exists():
            raise KeyError(f"LocalFilesystemStorage: key {remote_key!r} not found")
        await asyncio.to_thread(shutil.copy2, str(src), str(local_path))

    async def exists(self, remote_key: str) -> bool:
        """Return ``True`` if *remote_key* exists in the store.

        Args:
            remote_key: Key to probe.
        """
        return (self.root / remote_key).exists()

    async def list(self, prefix: str = "") -> list[str]:
        """Return sorted keys whose paths start with *prefix*.

        Args:
            prefix: Optional filter; empty string returns all keys.

        Returns:
            Sorted list of relative key strings.
        """
        keys: list[str] = []
        for path in self.root.rglob("*"):
            if path.is_file():
                key = str(path.relative_to(self.root))
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)


# ---------------------------------------------------------------------------
# RunpodVolumeStorage
# ---------------------------------------------------------------------------


def _require_runpod() -> object:
    """Import and return the ``runpod`` module, raising a clear error if absent.

    Returns:
        The ``runpod`` module object.

    Raises:
        ImportError: With install instructions if ``runpod`` is not installed.
    """
    try:
        import runpod  # lazy import — intentional

        return runpod
    except ImportError as exc:
        raise ImportError(
            "RunpodVolumeStorage requires the runpod SDK; "
            "install it with:  pip install rfdf[compute-runpod]"
        ) from exc


class RunpodVolumeStorage:
    """Implements :class:`RemoteStorage` via a RunPod Network Volume (SSH/SCP).

    RunPod Network Volumes are persistent NFS-backed storage attached to pods at
    a configurable mount path.  This implementation transfers files to/from the
    volume by connecting to a pod that has the volume attached.

    .. note::
        This is a **partial implementation**.  Real transfer to a RunPod Network
        Volume needs an SSH endpoint on a pod with the volume attached, which is
        unavailable without a live RunPod account: :meth:`put` validates the
        source and records the key in-process (so :meth:`exists` / :meth:`list`
        work within a session); :meth:`get` raises :class:`NotImplementedError`.
        Use :class:`LocalFilesystemStorage` for real storage without a RunPod
        account, or extend these methods with SCP/rsync against a live pod.

    Attributes:
        volume_id: RunPod Network Volume ID.
        mount_path: Path where the volume is mounted inside the pod.
    """

    def __init__(self, volume_id: str, mount_path: str = "/runpod-volume") -> None:
        """Initialise the storage backend.

        Args:
            volume_id: The RunPod Network Volume ID (visible in the RunPod
                console under "Storage").
            mount_path: Mount path used when attaching the volume to a pod.
                Defaults to ``"/runpod-volume"``.
        """
        self.volume_id = volume_id
        self.mount_path = mount_path
        # Key→virtual path mapping persisted in-memory for this session.
        # In production the actual file layout on the volume is the source of
        # truth; this cache makes exists() / list() cheap for the common case
        # where the same process uploaded the keys.
        self._uploaded_keys: set[str] = set()

    @property
    def name(self) -> str:
        """Always ``"runpod-volume"``."""
        return "runpod-volume"

    async def put(self, local_path: Path, remote_key: str) -> str:
        """Register *remote_key* for *local_path* on the RunPod Network Volume.

        Partial implementation (see the class note): the real file transfer to a
        Network Volume needs an SSH endpoint on a volume-attached pod, which is
        unavailable without a live RunPod account.  ``put`` validates the source
        file and records the key in-process so :meth:`exists` / :meth:`list`
        work within a session; operators extend it with an SCP/rsync transfer
        against a volume-attached pod.

        Args:
            local_path: Source file on the local filesystem.
            remote_key: Destination key inside the volume (relative path).

        Returns:
            The *remote_key* unchanged.

        Raises:
            FileNotFoundError: If *local_path* does not exist.
        """
        _require_runpod()
        if not local_path.exists():
            raise FileNotFoundError(f"RunpodVolumeStorage.put: {local_path} not found")
        self._uploaded_keys.add(remote_key)
        return remote_key

    async def get(self, remote_key: str, local_path: Path) -> None:
        """Download the object at *remote_key* from the RunPod Network Volume.

        Args:
            remote_key: Key of the object to retrieve.
            local_path: Destination path; parent directory must exist.

        Raises:
            KeyError: If *remote_key* is not known to this storage instance.
            RuntimeError: If the download fails.
        """
        _require_runpod()
        if remote_key not in self._uploaded_keys:
            raise KeyError(f"RunpodVolumeStorage: key {remote_key!r} not found")
        # In a full implementation this would SCP from the volume pod.
        raise NotImplementedError(
            "RunpodVolumeStorage.get: file download from RunPod Network Volume "
            "requires an active SSH endpoint; attach a pod with the volume and "
            "implement SCP transfer."
        )

    async def exists(self, remote_key: str) -> bool:
        """Return ``True`` if *remote_key* was uploaded by this storage instance.

        Args:
            remote_key: Key to probe.
        """
        _require_runpod()
        return remote_key in self._uploaded_keys

    async def list(self, prefix: str = "") -> list[str]:
        """Return sorted known keys that start with *prefix*.

        Args:
            prefix: Optional filter; empty string returns all known keys.

        Returns:
            Sorted list of matching key strings.
        """
        _require_runpod()
        return sorted(k for k in self._uploaded_keys if k.startswith(prefix))


__all__ = [
    "LocalFilesystemStorage",
    "RemoteStorage",
    "RunpodVolumeStorage",
]

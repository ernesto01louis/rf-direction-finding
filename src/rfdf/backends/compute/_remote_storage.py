"""``RemoteStorage`` Protocol and built-in implementations.

A ``RemoteStorage`` provides a key-value store for datasets and artifacts that
persist between compute runs so that large files are not re-uploaded with every
job submission.  Five implementations ship in-tree:

* :class:`LocalFilesystemStorage` — backed by a plain directory on disk; zero
  dependencies, fully testable without any provider account.
* :class:`RunpodVolumeStorage` — backed by a RunPod Network Volume; lazy
  ``runpod`` import so the module is importable without the SDK.
* :class:`ModalVolumeStorage` — backed by a Modal Volume; lazy ``modal`` import
  so the module is importable without the SDK.
* :class:`VastAiStorage` — backed by Vast.ai per-instance disk; lazy
  ``vastai`` import so the module is importable without the SDK.
* :class:`SkyPilotBucketStorage` — backed by a SkyPilot-managed cloud bucket
  (S3, GCS, or R2 depending on the enabled cloud); lazy ``sky`` import so the
  module is importable without the SDK.

All satisfy the :class:`RemoteStorage` ``@runtime_checkable`` Protocol.
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


# ---------------------------------------------------------------------------
# ModalVolumeStorage
# ---------------------------------------------------------------------------


def _require_modal() -> object:
    """Import and return the ``modal`` module, raising a clear error if absent.

    Returns:
        The ``modal`` module object.

    Raises:
        ImportError: With install instructions if ``modal`` is not installed.
    """
    try:
        import modal  # lazy import — intentional

        return modal
    except ImportError as exc:
        raise ImportError(
            "ModalVolumeStorage requires the modal SDK; "
            "install it with:  pip install rfdf[compute-modal]"
        ) from exc


class ModalVolumeStorage:
    """Implements :class:`RemoteStorage` via a Modal Volume.

    Modal Volumes are persistent network-backed storage that can be mounted
    into Modal Sandboxes or Functions at a configurable path.  This
    implementation records uploaded keys in-process so that :meth:`exists` /
    :meth:`list` work within a session.

    .. note::
        This is a **partial implementation**.  Real file transfer to a Modal
        Volume requires the ``modal`` SDK and active credentials (set up via
        ``modal token new`` or ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``
        env vars).  :meth:`put` validates the source file and records the key
        in-process; the actual upload to Modal's storage layer requires a live
        Modal account and is documented as a stub.  :meth:`get` raises
        :class:`NotImplementedError` because downloading from a Modal Volume
        outside a Modal container requires ``Volume.read_file()``, which needs
        active credentials.  Use :class:`LocalFilesystemStorage` for real
        storage without a Modal account.

    Attributes:
        volume_name: Modal Volume name (created via ``modal volume create``
            or :func:`modal.Volume.from_name`).
        mount_path: Path where the volume is mounted inside Modal containers.
    """

    def __init__(
        self, volume_name: str = "rfdf-default", mount_path: str = "/modal-volume"
    ) -> None:
        """Initialise the storage backend.

        Args:
            volume_name: The Modal Volume name.  The volume must already exist
                (create with ``modal volume create <name>``) or be configured
                with ``create_if_missing=True`` when the SDK acquires it.
            mount_path: Mount path used when attaching the volume to a sandbox
                or function.  Defaults to ``"/modal-volume"``.
        """
        self.volume_name = volume_name
        self.mount_path = mount_path
        # Key → virtual path mapping persisted in-memory for this session.
        # In production the actual file layout on the Modal Volume is the
        # source of truth; this cache makes exists() / list() cheap for the
        # common case where the same process uploaded the keys.
        self._uploaded_keys: set[str] = set()

    @property
    def name(self) -> str:
        """Always ``"modal-volume"``."""
        return "modal-volume"

    async def put(self, local_path: Path, remote_key: str) -> str:
        """Register *remote_key* for *local_path* on the Modal Volume.

        Partial implementation (see the class note): the real file upload to a
        Modal Volume requires active credentials and an async context via
        ``Volume.batch_upload()``.  ``put`` validates the source file and
        records the key in-process so :meth:`exists` / :meth:`list` work
        within a session.  Operators extend it with a
        ``vol.batch_upload()``-based transfer against a live Modal account.

        Args:
            local_path: Source file on the local filesystem.
            remote_key: Destination key inside the volume (relative path).

        Returns:
            The *remote_key* unchanged.

        Raises:
            FileNotFoundError: If *local_path* does not exist.
            ImportError: If the ``modal`` SDK is not installed.
        """
        _require_modal()
        if not local_path.exists():
            raise FileNotFoundError(f"ModalVolumeStorage.put: {local_path} not found")
        self._uploaded_keys.add(remote_key)
        return remote_key

    async def get(self, remote_key: str, local_path: Path) -> None:
        """Download the object at *remote_key* from the Modal Volume.

        Args:
            remote_key: Key of the object to retrieve.
            local_path: Destination path; parent directory must exist.

        Raises:
            KeyError: If *remote_key* is not known to this storage instance.
            NotImplementedError: Always — downloading from a Modal Volume
                outside a Modal container requires ``Volume.read_file()`` with
                active credentials; operators implement this by acquiring the
                volume via ``modal.Volume.from_name(volume_name)`` and calling
                ``async for chunk in vol.read_file(remote_key): ...`` to write
                chunks to *local_path*.
            ImportError: If the ``modal`` SDK is not installed.
        """
        _require_modal()
        if remote_key not in self._uploaded_keys:
            raise KeyError(f"ModalVolumeStorage: key {remote_key!r} not found")
        # In a full implementation this would use modal.Volume.read_file():
        #
        #   vol = modal.Volume.from_name(self.volume_name)
        #   with open(local_path, "wb") as fh:
        #       async for chunk in vol.read_file(remote_key):
        #           fh.write(chunk)
        #
        # This requires active Modal credentials and an async context.
        raise NotImplementedError(
            "ModalVolumeStorage.get: file download from a Modal Volume "
            "requires active Modal credentials; use modal.Volume.from_name() "
            "and Volume.read_file() to implement the transfer."
        )

    async def exists(self, remote_key: str) -> bool:
        """Return ``True`` if *remote_key* was uploaded by this storage instance.

        Args:
            remote_key: Key to probe.

        Raises:
            ImportError: If the ``modal`` SDK is not installed.
        """
        _require_modal()
        return remote_key in self._uploaded_keys

    async def list(self, prefix: str = "") -> list[str]:
        """Return sorted known keys that start with *prefix*.

        Args:
            prefix: Optional filter; empty string returns all known keys.

        Returns:
            Sorted list of matching key strings.

        Raises:
            ImportError: If the ``modal`` SDK is not installed.
        """
        _require_modal()
        return sorted(k for k in self._uploaded_keys if k.startswith(prefix))


# ---------------------------------------------------------------------------
# VastAiStorage
# ---------------------------------------------------------------------------


def _require_vastai() -> object:
    """Import and return the ``vastai`` module, raising a clear error if absent.

    Returns:
        The ``vastai`` module object.

    Raises:
        ImportError: With install instructions if ``vastai`` is not installed.
    """
    try:
        import vastai  # lazy import — intentional

        return vastai
    except ImportError as exc:
        raise ImportError(
            "VastAiStorage requires the vastai SDK; "
            "install it with:  pip install rfdf[compute-vastai]"
        ) from exc


class VastAiStorage:
    """Implements :class:`RemoteStorage` via Vast.ai per-instance disk.

    Vast.ai instances have local disk attached at creation time (the ``disk``
    parameter, in GiB).  Unlike RunPod Network Volumes or Modal Volumes, this
    disk is **not shared** across instances — it is ephemeral to a single
    instance booking.  If a job writes artifacts to instance-local disk and
    the instance is destroyed before :meth:`fetch_artifacts` can retrieve them,
    the data is lost.

    .. note::
        This is a **partial implementation**.  Real file transfer to/from a
        Vast.ai instance requires SSH access via the instance's public IP and
        SSH port (the Vast.ai ``vastai copy`` command or ``rsync``/``scp``
        against the instance endpoint), which is unavailable without a live
        Vast.ai account.  :meth:`put` validates the source file and records the
        key in-process so :meth:`exists` / :meth:`list` work within a session;
        :meth:`get` raises :class:`NotImplementedError`.  Use
        :class:`LocalFilesystemStorage` for real storage without a Vast.ai
        account, or extend these methods with SCP/rsync against a live instance
        SSH endpoint.

    Attributes:
        instance_disk_gb: Local disk size in GiB allocated to each instance.
        mount_path: Conventional path under which files are stored on the
            instance disk.
    """

    def __init__(
        self,
        instance_disk_gb: float = 10.0,
        mount_path: str = "/workspace",
    ) -> None:
        """Initialise the storage backend.

        Args:
            instance_disk_gb: Local disk size in GiB to allocate to each
                instance (passed as the ``disk`` parameter to
                ``create_instance``).  Defaults to 10 GiB.
            mount_path: Conventional working path on the instance disk.
                Defaults to ``"/workspace"``.
        """
        self.instance_disk_gb = instance_disk_gb
        self.mount_path = mount_path
        # Key → virtual path mapping persisted in-memory for this session.
        # In production the actual file layout on the instance disk is the
        # source of truth; this cache makes exists() / list() cheap for the
        # common case where the same process uploaded the keys.
        self._uploaded_keys: set[str] = set()

    @property
    def name(self) -> str:
        """Always ``"vastai-instance-disk"``."""
        return "vastai-instance-disk"

    async def put(self, local_path: Path, remote_key: str) -> str:
        """Register *remote_key* for *local_path* on the Vast.ai instance disk.

        Partial implementation (see the class note): real file transfer to a
        Vast.ai instance disk needs SSH access to the instance, which is
        unavailable without a live Vast.ai account.  ``put`` validates the
        source file and records the key in-process so :meth:`exists` /
        :meth:`list` work within a session; operators extend it with an SCP
        or ``vastai copy`` transfer against a running instance.

        Args:
            local_path: Source file on the local filesystem.
            remote_key: Destination key on the instance disk (relative path).

        Returns:
            The *remote_key* unchanged.

        Raises:
            FileNotFoundError: If *local_path* does not exist.
            ImportError: If the ``vastai`` SDK is not installed.
        """
        _require_vastai()
        if not local_path.exists():
            raise FileNotFoundError(f"VastAiStorage.put: {local_path} not found")
        self._uploaded_keys.add(remote_key)
        return remote_key

    async def get(self, remote_key: str, local_path: Path) -> None:
        """Download the object at *remote_key* from the Vast.ai instance disk.

        Args:
            remote_key: Key of the object to retrieve.
            local_path: Destination path; parent directory must exist.

        Raises:
            KeyError: If *remote_key* is not known to this storage instance.
            NotImplementedError: Always — downloading from a Vast.ai instance
                disk requires SSH access to the running instance.  Operators
                implement this via ``vastai copy <instance_id>:/path /local/path``
                or ``scp -P <port> root@<ip>:<path> <local_path>``.
            ImportError: If the ``vastai`` SDK is not installed.
        """
        _require_vastai()
        if remote_key not in self._uploaded_keys:
            raise KeyError(f"VastAiStorage: key {remote_key!r} not found")
        # In a full implementation this would use `vastai copy` or SCP:
        #
        #   vastai copy <instance_id>:<mount_path>/<remote_key> <local_path>
        #   # or via SSH:
        #   scp -P <ssh_port> root@<ip>:<mount_path>/<remote_key> <local_path>
        #
        # This requires a running instance and active Vast.ai credentials.
        raise NotImplementedError(
            "VastAiStorage.get: file download from a Vast.ai instance disk "
            "requires SSH access to the instance; use `vastai copy "
            "<instance_id>:/path /local/path` or SCP/rsync against the "
            "instance's SSH endpoint."
        )

    async def exists(self, remote_key: str) -> bool:
        """Return ``True`` if *remote_key* was uploaded by this storage instance.

        Args:
            remote_key: Key to probe.

        Raises:
            ImportError: If the ``vastai`` SDK is not installed.
        """
        _require_vastai()
        return remote_key in self._uploaded_keys

    async def list(self, prefix: str = "") -> list[str]:
        """Return sorted known keys that start with *prefix*.

        Args:
            prefix: Optional filter; empty string returns all known keys.

        Returns:
            Sorted list of matching key strings.

        Raises:
            ImportError: If the ``vastai`` SDK is not installed.
        """
        _require_vastai()
        return sorted(k for k in self._uploaded_keys if k.startswith(prefix))


# ---------------------------------------------------------------------------
# SkyPilotBucketStorage
# ---------------------------------------------------------------------------


def _require_sky() -> object:
    """Import and return the ``sky`` module, raising a clear error if absent.

    Returns:
        The ``sky`` module object.

    Raises:
        ImportError: With install instructions if ``sky`` is not installed.
    """
    try:
        import sky  # lazy import — intentional

        return sky
    except ImportError as exc:
        raise ImportError(
            "SkyPilotBucketStorage requires the skypilot SDK; "
            "install it with:  pip install rfdf[compute-skypilot]"
        ) from exc


class SkyPilotBucketStorage:
    """Implements :class:`RemoteStorage` via a SkyPilot-managed cloud bucket.

    SkyPilot manages storage buckets across the cloud providers it supports
    (AWS S3, GCP GCS, Cloudflare R2, etc.).  The specific bucket backend is
    determined by which cloud is active in the operator's ``sky check``
    environment.  A ``sky.Storage`` object is created with ``persistent=True``
    so the bucket survives across task runs.

    .. note::
        This is a **partial implementation**.  Real file transfer to/from a
        SkyPilot-managed bucket requires active cloud credentials (configured
        via ``sky check``) and a live bucket.  :meth:`put` validates the source
        file and records the key in-process so :meth:`exists` / :meth:`list`
        work within a session; the actual upload to the cloud bucket would use
        ``sky.Storage.upload_local_dir`` or the cloud SDK directly.  :meth:`get`
        raises :class:`NotImplementedError` because downloading from a cloud
        bucket outside a SkyPilot cluster requires cloud-provider-specific
        transfer logic.  Use :class:`LocalFilesystemStorage` for real storage
        without cloud credentials, or extend these methods with
        ``boto3``/``google-cloud-storage`` transfers against the active bucket.

    Attributes:
        bucket_name: SkyPilot-managed bucket name (must be globally unique
            across the target cloud provider's namespace).
        mount_path: Conventional path where the bucket is mounted inside
            SkyPilot cluster nodes.
    """

    def __init__(
        self,
        bucket_name: str = "rfdf-skypilot",
        mount_path: str = "/sky-bucket",
    ) -> None:
        """Initialise the storage backend.

        Args:
            bucket_name: The bucket name passed to ``sky.Storage``.  Must
                follow cloud provider naming rules (lowercase, hyphens, no
                underscores for S3; 63-char max).  Defaults to
                ``"rfdf-skypilot"``.
            mount_path: Path where the bucket is mounted inside SkyPilot
                cluster nodes.  Defaults to ``"/sky-bucket"``.
        """
        self.bucket_name = bucket_name
        self.mount_path = mount_path
        # Key → virtual path mapping persisted in-memory for this session.
        # In production the actual file layout in the cloud bucket is the
        # source of truth; this cache makes exists() / list() cheap for the
        # common case where the same process uploaded the keys.
        self._uploaded_keys: set[str] = set()

    @property
    def name(self) -> str:
        """Always ``"skypilot-bucket"``."""
        return "skypilot-bucket"

    async def put(self, local_path: Path, remote_key: str) -> str:
        """Register *remote_key* for *local_path* in the SkyPilot bucket.

        Partial implementation (see the class note): real file transfer to a
        SkyPilot-managed cloud bucket requires active cloud credentials and a
        live bucket, unavailable without ``sky check``-configured providers.
        ``put`` validates the source file and records the key in-process so
        :meth:`exists` / :meth:`list` work within a session; operators extend
        it with a ``boto3.put_object`` / ``google.cloud.storage.Blob.upload``
        call or by writing to the bucket mount path on the cluster.

        Args:
            local_path: Source file on the local filesystem.
            remote_key: Destination key in the bucket (relative path).

        Returns:
            The *remote_key* unchanged.

        Raises:
            FileNotFoundError: If *local_path* does not exist.
            ImportError: If the ``sky`` SDK is not installed.
        """
        _require_sky()
        if not local_path.exists():
            raise FileNotFoundError(f"SkyPilotBucketStorage.put: {local_path} not found")
        self._uploaded_keys.add(remote_key)
        return remote_key

    async def get(self, remote_key: str, local_path: Path) -> None:
        """Download the object at *remote_key* from the SkyPilot bucket.

        Args:
            remote_key: Key of the object to retrieve.
            local_path: Destination path; parent directory must exist.

        Raises:
            KeyError: If *remote_key* is not known to this storage instance.
            NotImplementedError: Always — downloading from a SkyPilot-managed
                cloud bucket outside a cluster requires the cloud provider's
                native SDK (``boto3`` for S3, ``google-cloud-storage`` for GCS,
                etc.).  Operators implement this by identifying the bucket's
                provider from ``sky.Storage`` and calling the appropriate SDK.
                Alternatively, mount the bucket inside a cluster node and read
                from the mount path.
            ImportError: If the ``sky`` SDK is not installed.
        """
        _require_sky()
        if remote_key not in self._uploaded_keys:
            raise KeyError(f"SkyPilotBucketStorage: key {remote_key!r} not found")
        # In a full implementation this would use the cloud provider's SDK:
        #
        #   # For S3 (AWS):
        #   import boto3
        #   s3 = boto3.client("s3")
        #   s3.download_file(self.bucket_name, remote_key, str(local_path))
        #
        #   # For GCS (GCP):
        #   from google.cloud import storage
        #   client = storage.Client()
        #   bucket = client.bucket(self.bucket_name)
        #   bucket.blob(remote_key).download_to_filename(str(local_path))
        #
        # The active cloud is determined by which provider SkyPilot selects;
        # detecting it requires inspecting the sky.Storage or sky.check output.
        raise NotImplementedError(
            "SkyPilotBucketStorage.get: file download from a SkyPilot-managed "
            "cloud bucket requires the cloud provider's native SDK (boto3 for "
            "S3, google-cloud-storage for GCS).  Identify the active provider "
            "from sky.check output and implement the corresponding download."
        )

    async def exists(self, remote_key: str) -> bool:
        """Return ``True`` if *remote_key* was uploaded by this storage instance.

        Args:
            remote_key: Key to probe.

        Raises:
            ImportError: If the ``sky`` SDK is not installed.
        """
        _require_sky()
        return remote_key in self._uploaded_keys

    async def list(self, prefix: str = "") -> list[str]:
        """Return sorted known keys that start with *prefix*.

        Args:
            prefix: Optional filter; empty string returns all known keys.

        Returns:
            Sorted list of matching key strings.

        Raises:
            ImportError: If the ``sky`` SDK is not installed.
        """
        _require_sky()
        return sorted(k for k in self._uploaded_keys if k.startswith(prefix))


__all__ = [
    "LocalFilesystemStorage",
    "ModalVolumeStorage",
    "RemoteStorage",
    "RunpodVolumeStorage",
    "SkyPilotBucketStorage",
    "VastAiStorage",
]

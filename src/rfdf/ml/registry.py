"""Local filesystem model registry for trained RF signal classifiers.

The registry lives under ``~/.local/share/rfdf/models/`` by default (via
:func:`platformdirs.user_data_dir`), or the path set in the
``RFDF_MODEL_REGISTRY`` environment variable.  This makes it trivial to
point tests at a temporary directory without touching real state.

Each registered model occupies ``<registry_root>/<model_id>/`` containing:

* ``manifest.json`` — :class:`ModelManifest` (append-only evaluation history).
* ``model.pt`` — PyTorch checkpoint (``torch.save`` dict).
* ``classes.json`` — list of class-name strings.
* ``model.onnx`` (optional) — ONNX export.
* ``model.hef`` (optional) — Hailo HEF binary.
* ``confusion.png`` (optional) — confusion-matrix image.

**Lazy-import rule:** importing this module must not load ``torch``.  The
``load_model`` function imports torch inside the function body.  All other
public functions are pure filesystem / JSON operations.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rfdf.ml._manifest import current_git_sha
from rfdf.ml.errors import RegistryError

# ---------------------------------------------------------------------------
# Registry root
# ---------------------------------------------------------------------------


def _registry_root() -> Path:
    """Return the active registry root directory.

    Checks ``RFDF_MODEL_REGISTRY`` first, falls back to the platform-specific
    user data directory.

    Returns:
        Absolute path to the registry root (not yet created).
    """
    env_override = os.environ.get("RFDF_MODEL_REGISTRY", "")
    if env_override:
        return Path(env_override)
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir("rfdf")) / "models"
    except ImportError:
        # platformdirs is a base dependency — this branch is a safety net only.
        return Path.home() / ".local" / "share" / "rfdf" / "models"


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


class _DatasetInfo(BaseModel):
    """Dataset provenance inside :class:`ModelManifest`."""

    model_config = ConfigDict(frozen=True)

    name: str = ""
    version: str = ""
    hash: str = ""


class _TrainingInfo(BaseModel):
    """Training provenance inside :class:`ModelManifest`."""

    model_config = ConfigDict(frozen=True)

    config: dict[str, Any] = Field(default_factory=dict)
    compute_backend: str = ""
    gpu_model: str | None = None
    duration_h: float = 0.0
    cost_eur: float = 0.0


class _EvaluationInfo(BaseModel):
    """Evaluation metrics inside :class:`ModelManifest`."""

    model_config = ConfigDict(frozen=True)

    accuracy_top1: float = 0.0
    accuracy_top5: float = 0.0
    confusion_matrix_path: str | None = None
    per_class_metrics: dict[str, Any] = Field(default_factory=dict)


class _ProvenanceInfo(BaseModel):
    """Code + operator provenance inside :class:`ModelManifest`."""

    model_config = ConfigDict(frozen=True)

    git_sha: str = ""
    trained_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    trainer: str = ""


class _ExportsInfo(BaseModel):
    """Paths to exported artefacts inside :class:`ModelManifest`.

    Each field is the *filename* (relative to the model directory), not an
    absolute path, so manifests remain portable after ``export_model`` /
    ``import_model`` round-trips.
    """

    model_config = ConfigDict(frozen=True)

    onnx: str | None = None
    hef: str | None = None
    tflite: str | None = None
    coreml: str | None = None


class ModelManifest(BaseModel):
    """Immutable record for a registered model.

    The ``evaluation_history`` list is append-only: re-evaluating or
    re-registering a model appends a new :class:`_EvaluationInfo` snapshot
    rather than overwriting the existing one.  The ``evaluation`` field
    always reflects the *latest* evaluation; prior records are preserved in
    ``evaluation_history`` for auditing.

    Attributes:
        model_id: Unique identifier for this model in the registry.
        architecture: Model architecture name (see :const:`rfdf.ml.models.ARCHITECTURES`).
        task: High-level task descriptor (e.g. ``"modulation_classification"``).
        dataset: Dataset provenance.
        training: Training provenance and resource usage.
        evaluation: Latest evaluation metrics.
        provenance: Code and operator provenance.
        exports: Paths (filenames) of exported artefacts.
        evaluation_history: Append-only list of all prior evaluations.
        registered_at: ISO-8601 UTC timestamp when the model was first registered.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_id: str
    architecture: str
    task: str = "modulation_classification"
    dataset: _DatasetInfo = Field(default_factory=_DatasetInfo)
    training: _TrainingInfo = Field(default_factory=_TrainingInfo)
    evaluation: _EvaluationInfo = Field(default_factory=_EvaluationInfo)
    provenance: _ProvenanceInfo = Field(default_factory=_ProvenanceInfo)
    exports: _ExportsInfo = Field(default_factory=_ExportsInfo)
    evaluation_history: list[_EvaluationInfo] = Field(default_factory=list)
    registered_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_dir(model_id: str) -> Path:
    """Return the directory for *model_id* inside the registry root."""
    return _registry_root() / model_id


def _manifest_path(model_id: str) -> Path:
    """Return the manifest.json path for *model_id*."""
    return _model_dir(model_id) / "manifest.json"


def _load_manifest_dict(model_id: str) -> dict[str, Any]:
    """Read and return the raw manifest dict for *model_id*.

    Raises:
        RegistryError: If the model directory or manifest file is absent.
    """
    mpath = _manifest_path(model_id)
    if not mpath.exists():
        raise RegistryError(
            f"Model {model_id!r} not found in registry "
            f"(expected manifest at {mpath}).  "
            "Run 'rfdf ml registry list' to see registered models."
        )
    with open(mpath, encoding="utf-8") as fh:
        return dict(json.load(fh))


def _save_manifest(manifest: ModelManifest) -> None:
    """Serialise *manifest* to disk (overwrites the existing file).

    The parent directory must already exist.
    """
    mdir = _model_dir(manifest.model_id)
    mdir.mkdir(parents=True, exist_ok=True)
    with open(mdir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest.model_dump(), fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_model(
    model_id: str,
    output_dir: Path,
    *,
    architecture: str,
    task: str = "modulation_classification",
    dataset: _DatasetInfo | None = None,
    training: _TrainingInfo | None = None,
    evaluation: _EvaluationInfo | None = None,
    provenance: _ProvenanceInfo | None = None,
) -> ModelManifest:
    """Ingest a training ``output_dir`` into the registry as *model_id*.

    Copies ``model.pt`` (or ``models/best.pt``), ``classes.json``, and any
    optional artefacts from *output_dir* into the registry directory for
    *model_id*.  If a prior manifest already exists the new evaluation is
    appended to ``evaluation_history`` (append-only; prior records preserved).

    Args:
        model_id: Unique name for the model (e.g. ``"resnet1d-modulation-v1"``).
        output_dir: The ``output_dir`` that :func:`~rfdf.ml.training.train`
            wrote to (contains ``models/best.pt``, ``manifest.json``,
            ``classes.json``, etc.).
        architecture: Architecture name (e.g. ``"resnet1d"``).
        task: High-level task descriptor (default ``"modulation_classification"``).
        dataset: Optional dataset provenance.  If omitted, an empty record is used.
        training: Optional training provenance.  If omitted, values from the
            ``output_dir/manifest.json`` training config are used where available.
        evaluation: Optional evaluation metrics.  If omitted, values from
            ``output_dir/manifest.json`` are used where available.
        provenance: Optional code/operator provenance.  If omitted, the current
            git SHA is captured.

    Returns:
        The newly written :class:`ModelManifest`.

    Raises:
        RegistryError: If neither ``output_dir/model.pt`` nor
            ``output_dir/models/best.pt`` can be found.
    """
    output_dir = Path(output_dir)
    mdir = _model_dir(model_id)
    mdir.mkdir(parents=True, exist_ok=True)

    # ---- Locate and copy the checkpoint -----------------------------------
    candidate_pt: Path | None = None
    for p in [output_dir / "model.pt", output_dir / "models" / "best.pt"]:
        if p.exists():
            candidate_pt = p
            break
    if candidate_pt is None:
        raise RegistryError(
            f"Cannot register {model_id!r}: no checkpoint found in {output_dir}.  "
            "Expected 'model.pt' or 'models/best.pt'."
        )
    shutil.copy2(candidate_pt, mdir / "model.pt")

    # ---- Copy optional artefacts -----------------------------------------
    for filename in ("classes.json", "confusion.png", "model.onnx", "model.hef"):
        src = output_dir / filename
        if src.exists():
            shutil.copy2(src, mdir / filename)

    # ---- Resolve provenance from existing manifest.json (if present) ------
    src_manifest_path = output_dir / "manifest.json"
    src_manifest_data: dict[str, Any] = {}
    if src_manifest_path.exists():
        with open(src_manifest_path, encoding="utf-8") as fh:
            src_manifest_data = dict(json.load(fh))

    resolved_training = training or _TrainingInfo(
        config=dict(src_manifest_data.get("training_config", {})),
        gpu_model=str(src_manifest_data.get("gpu_model") or "") or None,
    )
    src_eval: dict[str, Any] = dict(src_manifest_data.get("evaluation", {}))
    resolved_evaluation = evaluation or _EvaluationInfo(
        accuracy_top1=float(src_eval.get("best_val_accuracy", 0.0)),
    )
    resolved_provenance = provenance or _ProvenanceInfo(
        git_sha=str(src_manifest_data.get("git_sha", "") or current_git_sha()),
        trained_at=str(src_manifest_data.get("completed_at", datetime.now(tz=UTC).isoformat())),
    )

    # ---- Detect exports already present in the model directory ------------
    exports = _ExportsInfo(
        onnx="model.onnx" if (mdir / "model.onnx").exists() else None,
        hef="model.hef" if (mdir / "model.hef").exists() else None,
    )

    # ---- Append-only evaluation history -----------------------------------
    existing_history: list[_EvaluationInfo] = []
    if _manifest_path(model_id).exists():
        try:
            raw = _load_manifest_dict(model_id)
            existing_history = [_EvaluationInfo(**e) for e in raw.get("evaluation_history", [])]
            # Append the *previous* primary evaluation to history before
            # replacing it with the new one.
            prev_eval_raw = raw.get("evaluation", {})
            if prev_eval_raw:
                existing_history.append(_EvaluationInfo(**prev_eval_raw))
        except (RegistryError, Exception):
            pass

    manifest = ModelManifest(
        model_id=model_id,
        architecture=architecture,
        task=task,
        dataset=dataset or _DatasetInfo(),
        training=resolved_training,
        evaluation=resolved_evaluation,
        provenance=resolved_provenance,
        exports=exports,
        evaluation_history=existing_history,
    )
    _save_manifest(manifest)
    return manifest


def get_manifest(model_id: str) -> ModelManifest:
    """Return the :class:`ModelManifest` for *model_id*.

    Args:
        model_id: Registry model identifier.

    Returns:
        The parsed :class:`ModelManifest`.

    Raises:
        RegistryError: If *model_id* is not registered.
    """
    raw = _load_manifest_dict(model_id)
    return ModelManifest(**raw)


def load_model(model_id: str) -> Any:
    """Load and return the PyTorch model for *model_id*.

    Torch is imported lazily inside this function so that importing
    :mod:`rfdf.ml.registry` remains torch-free.

    Args:
        model_id: Registry model identifier.

    Returns:
        The loaded PyTorch checkpoint dict (as returned by ``torch.load``).
        Use ``checkpoint["model_state_dict"]`` to restore model weights.

    Raises:
        RegistryError: If *model_id* is not registered or ``model.pt`` is absent.
        ImportError: If torch is not installed (the ``[ml]`` extra is required).
    """
    import torch  # lazy import — only here

    mdir = _model_dir(model_id)
    pt_path = mdir / "model.pt"
    if not pt_path.exists():
        raise RegistryError(
            f"Model {model_id!r}: 'model.pt' not found in registry directory {mdir}.  "
            "The model may have been manually deleted.  Re-register to restore it."
        )
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def list_models() -> list[str]:
    """Return the sorted list of registered model IDs.

    Returns:
        Sorted list of model identifier strings.  Empty list if the registry
        root does not exist yet.
    """
    root = _registry_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def delete_model(model_id: str) -> None:
    """Remove *model_id* and all its artefacts from the registry.

    Args:
        model_id: Registry model identifier.

    Raises:
        RegistryError: If *model_id* is not registered.
    """
    mdir = _model_dir(model_id)
    if not mdir.exists() or not _manifest_path(model_id).exists():
        raise RegistryError(f"Model {model_id!r} not found in registry — cannot delete.")
    shutil.rmtree(mdir)


def export_model(model_id: str, dest: Path) -> Path:
    """Bundle the model directory into a ``.tar.gz`` archive at *dest*.

    The archive name is ``<model_id>.tar.gz``.  If *dest* is a directory,
    the archive is written inside it; otherwise *dest* is used as the
    exact file path.

    Args:
        model_id: Registry model identifier.
        dest: Destination path or directory.

    Returns:
        Path to the created archive.

    Raises:
        RegistryError: If *model_id* is not registered.
    """
    mdir = _model_dir(model_id)
    if not mdir.exists():
        raise RegistryError(f"Model {model_id!r} not found — cannot export.")

    dest = Path(dest)
    # Treat dest as a directory if it has no archive-like suffix.
    archive_path = (
        dest / f"{model_id}.tar.gz"
        if not dest.suffix or dest.suffix not in (".gz", ".tgz", ".tar")
        else dest
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(mdir, arcname=model_id)
    return archive_path


def import_model(archive: Path) -> str:
    """Unpack a model archive (created by :func:`export_model`) into the registry.

    If a model with the same ID already exists it is overwritten.

    Args:
        archive: Path to a ``.tar.gz`` archive produced by :func:`export_model`.

    Returns:
        The model ID extracted from the archive.

    Raises:
        RegistryError: If the archive is not a valid model bundle (no
            ``manifest.json`` inside).
    """
    archive = Path(archive)
    if not archive.exists():
        raise RegistryError(f"Archive not found: {archive}")

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(archive, "r:gz") as tar:
            # Safety: only extract within tmpdir.
            tar.extractall(path=tmpdir)

        tmp_path = Path(tmpdir)
        # The archive contains exactly one top-level directory whose name is
        # the model_id.
        top_level = [p for p in tmp_path.iterdir() if p.is_dir()]
        if not top_level:
            raise RegistryError(f"Archive {archive} contains no model directory.")
        model_dir_extracted = top_level[0]
        model_id = model_dir_extracted.name

        if not (model_dir_extracted / "manifest.json").exists():
            raise RegistryError(
                f"Archive {archive} does not contain a manifest.json — "
                "it may not be a valid rfdf model bundle."
            )

        dest = _registry_root() / model_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(model_dir_extracted, dest)

    return model_id


def update_exports(model_id: str, **kwargs: str | None) -> ModelManifest:
    """Update the exports record for *model_id* after a new artefact is created.

    This is called by :mod:`rfdf.ml.export` functions after a successful export
    to keep the manifest in sync.

    Args:
        model_id: Registry model identifier.
        **kwargs: Export format keys and their filenames (e.g. ``onnx="model.onnx"``).
            Pass ``None`` to clear an entry.

    Returns:
        The updated :class:`ModelManifest`.

    Raises:
        RegistryError: If *model_id* is not registered.
    """
    raw = _load_manifest_dict(model_id)
    exports_raw: dict[str, Any] = dict(raw.get("exports", {}))
    exports_raw.update({k: v for k, v in kwargs.items()})
    raw["exports"] = exports_raw
    manifest = ModelManifest(**raw)
    _save_manifest(manifest)
    return manifest


__all__: list[str] = [
    "ModelManifest",
    "delete_model",
    "export_model",
    "get_manifest",
    "import_model",
    "list_models",
    "load_model",
    "register_model",
    "update_exports",
]

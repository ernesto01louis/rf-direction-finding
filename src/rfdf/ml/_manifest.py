"""Training run manifest — a frozen record of a completed training run.

:class:`TrainingManifest` captures enough metadata to reproduce the run and
audit its provenance.  It is written as ``manifest.json`` at the end of every
training run by :func:`write_manifest`.

This module is **pure / torch-free** at module level: stdlib + pydantic only.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------


class TrainingManifest(BaseModel):
    """Provenance record for a completed training run.

    Attributes:
        training_config: Serialised :class:`~rfdf.ml.recipes.TrainingSpec` as a
            plain dict.  Written verbatim so the JSON is self-describing.
        dataset_version: Free-form dataset version/fingerprint string
            (e.g. a hash or a version tag).
        git_sha: Git commit SHA of the codebase at training time.  ``"unknown"``
            when the repository is unavailable.
        starting_seeds: Per-component seeds used to initialise random state
            (e.g. ``{"torch": 0, "numpy": 0, "python": 0}``).
        host: Hostname of the machine that ran the training.
        gpu_model: GPU model string (e.g. ``"NVIDIA A100 80GB"``).
            ``None`` for CPU-only runs.
        completed_at: ISO-8601 timestamp (UTC) when the run finished.
        evaluation: Free-form evaluation metrics dict
            (e.g. ``{"best_val_accuracy": 0.923, "final_step": 5000}``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    training_config: dict[str, Any]
    dataset_version: str
    git_sha: str
    starting_seeds: dict[str, int] = Field(default_factory=dict)
    host: str = ""
    gpu_model: str | None = None
    completed_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    evaluation: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def current_git_sha() -> str:
    """Return the HEAD commit SHA of the current repository.

    Args:
        (none)

    Returns:
        A 40-character hex SHA string, or ``"unknown"`` if ``git`` is not
        available or the working directory is not a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def write_manifest(manifest: TrainingManifest, path: Path) -> None:
    """Serialise *manifest* to JSON at *path*.

    The parent directory is created if it does not exist.

    Args:
        manifest: The manifest to serialise.
        path: Destination file path (e.g. ``output_dir / "manifest.json"``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest.model_dump(), fh, indent=2)
        fh.write("\n")


__all__: list[str] = [
    "TrainingManifest",
    "current_git_sha",
    "write_manifest",
]

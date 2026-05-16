"""Training recipe system for RF signal-classification models.

A :class:`TrainingRecipe` is a frozen TOML-backed specification that captures
everything needed to reproduce a training run: the dataset, model architecture,
optimisation hyper-parameters, and compute-backend requirements.

Recipes are **pure / torch-free** at module level.  Functions that need torch
(e.g. :func:`TrainingRecipe.to_compute_job`) write a ``train.py`` runpy shim so
the job can be submitted to any HAL :class:`~rfdf.hal.compute.ComputeBackend`
without importing torch here.

Usage::

    from pathlib import Path
    from rfdf.ml.recipes import load_recipe

    recipe = load_recipe(Path("recipes/sig53-resnet1d-baseline.toml"))
    job = recipe.to_compute_job(working_dir=Path("/tmp/run-001"))
    # submit job to any ComputeBackend…
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rfdf.hal.compute import ComputeJob
from rfdf.ml.datasets.augmentation import AugmentationConfig

# ---------------------------------------------------------------------------
# Sub-model specs
# ---------------------------------------------------------------------------


class DatasetSpec(BaseModel):
    """Dataset specification for a training recipe.

    Attributes:
        kind: Dataset source — one of ``"modulation"``, ``"protocol"``,
            ``"radioml"``, or ``"captured"``.
        num_signals: For ``"modulation"`` kind — number of distinct classes.
        num_samples_per_signal: For synthetic kinds — IQ recordings per class.
        num_iq_samples: IQ samples per recording (synthetic kinds).
        impairments: TorchSig impairment level for synthetic datasets.
        augmentation: Optional rfdf augmentation applied at ``__getitem__``.
        dataset_path: Filesystem path to the HDF5 or SigMF directory (for
            ``"radioml"`` and ``"captured"`` kinds).
        snr_filter: SNR values to include (``"radioml"`` only).
        class_filter: Class names to include (``"radioml"`` only).
        seed: RNG seed for reproducible generation.
    """

    model_config = ConfigDict(frozen=True)

    kind: str  # "modulation" | "protocol" | "radioml" | "captured"
    # --- synthetic-dataset fields ---
    num_signals: int = 53
    num_samples_per_signal: int = 256
    num_iq_samples: int = 4096
    impairments: str = "cabled"
    augmentation: AugmentationConfig | None = None
    # --- radioml / captured ---
    dataset_path: str | None = None
    snr_filter: list[int] | None = None
    class_filter: list[str] | None = None
    # --- common ---
    seed: int = 0


class ModelSpec(BaseModel):
    """Model architecture specification.

    Attributes:
        architecture: One of ``"resnet1d"``, ``"resnet2d"``, ``"transformer"``,
            or ``"efficientnet"``.
        num_classes: Number of output classes (must match the dataset).
        input_shape: Expected input tensor shape excluding the batch dimension.
            For ``"resnet1d"`` use ``(2, num_samples)``; for 2-D architectures
            use ``(channels, freq_bins, time_bins)``.
        extra_kwargs: Additional keyword arguments forwarded to
            :func:`~rfdf.ml.models.build_model` (e.g. ``depth="resnet34"``).
    """

    model_config = ConfigDict(frozen=True)

    architecture: str
    num_classes: int
    input_shape: tuple[int, ...]
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class TrainingSpec(BaseModel):
    """Optimisation hyper-parameters for a training run.

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Mini-batch size per GPU (or CPU).
        learning_rate: Peak learning rate for the optimizer.
        optimizer: Optimizer name.  Only ``"adamw"`` is currently supported.
        scheduler: LR schedule.  Only ``"cosine"`` is currently supported.
        warmup_steps: Number of linear-warmup steps before the cosine decay.
        grad_accum_steps: Gradient-accumulation steps (effective batch = batch
            * grad_accum_steps).
        amp: Enable automatic mixed-precision training (CUDA only).
        ddp: Wrap the model in ``DistributedDataParallel`` when
            ``torch.distributed.is_initialized()``.
        checkpoint_every_steps: Save a checkpoint every N optimizer steps.
        keep_top_k: Keep at most this many checkpoints ranked by val accuracy.
        seed: Random seed for weight initialisation and DataLoader workers.
        wandb: Enable Weights & Biases logging (lazy import; must install wandb).
        mlflow: Enable MLflow logging (lazy import; must install mlflow).
    """

    model_config = ConfigDict(frozen=True)

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    grad_accum_steps: int = 1
    amp: bool = False
    ddp: bool = False
    checkpoint_every_steps: int = 500
    keep_top_k: int = 3
    seed: int = 0
    wandb: bool = False
    mlflow: bool = False


class ComputeSpec(BaseModel):
    """Compute-backend requirements for a training job.

    Attributes:
        backend: Backend name passed to the HAL (e.g. ``"local"``, ``"runpod"``).
        gpu_count: Number of GPUs requested (0 = CPU).
        gpu_min_vram_gb: Minimum per-GPU VRAM in GB.
        gpu_model: Optional GPU model preference (e.g. ``"A6000"``).
        timeout_h: Wall-clock timeout in **hours**.  Converted to seconds when
            producing a :class:`~rfdf.hal.compute.ComputeJob`.
        priority: Scheduling priority hint (``"low" | "normal" | "high"``).
        container_image: OCI image reference.  When set,
            :meth:`TrainingRecipe.to_compute_job` populates
            ``ComputeJob.container_image`` and leaves ``pip_requirements``
            empty.  Mutually exclusive with ``pip_requirements`` on the
            resulting :class:`ComputeJob`.
        pip_requirements: PEP 440 requirement strings installed before the job
            runs.  Used when ``container_image`` is ``None``.
    """

    model_config = ConfigDict(frozen=True)

    backend: str = "local"
    gpu_count: int = 0
    gpu_min_vram_gb: float = 0.0
    gpu_model: str | None = None
    timeout_h: float = 6.0
    priority: str = "normal"
    container_image: str | None = None
    pip_requirements: list[str] = Field(default_factory=lambda: ["rfdf[ml,ml-onnx]"])


# ---------------------------------------------------------------------------
# Top-level recipe
# ---------------------------------------------------------------------------


class TrainingRecipe(BaseModel):
    """A complete, self-contained training specification.

    A ``TrainingRecipe`` bundles the dataset, model, optimisation
    hyper-parameters, and compute requirements into a single frozen artefact
    that can be serialised (TOML, JSON), version-controlled, and reproduced.

    The :meth:`to_compute_job` method compiles the recipe into a
    :class:`~rfdf.hal.compute.ComputeJob` ready for submission to any HAL
    backend.

    Attributes:
        name: Human-readable recipe identifier (used in logs and artefact paths).
        dataset: Dataset specification.
        model: Model architecture specification.
        training: Optimisation hyper-parameters.
        compute: Compute-backend requirements.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    dataset: DatasetSpec
    model: ModelSpec
    training: TrainingSpec
    compute: ComputeSpec

    def to_compute_job(self, working_dir: Path) -> ComputeJob:
        """Compile the recipe into a HAL :class:`~rfdf.hal.compute.ComputeJob`.

        Writes a 3-line ``train.py`` runpy shim into *working_dir* so the job
        entry-script is a plain file path (the HAL contract requires a path,
        not a ``-m module`` string).  The shim delegates to
        :mod:`rfdf.ml.train_entrypoint`.

        The recipe is serialised as JSON and injected via the
        ``RFDF_RECIPE`` environment variable, so the entrypoint can reconstruct
        it without additional file I/O.

        Args:
            working_dir: Directory uploaded to the compute backend.  Created if
                it does not exist.

        Returns:
            A validated :class:`~rfdf.hal.compute.ComputeJob`.  Either
            ``container_image`` or ``pip_requirements`` is populated (never
            both), satisfying the XOR validator.
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        # Write the runpy shim so entry_script points to a real file.
        shim_path = working_dir / "train.py"
        shim_path.write_text(
            "import runpy\nrunpy.run_module('rfdf.ml.train_entrypoint', run_name='__main__')\n"
        )

        # Serialise the recipe to JSON for the entrypoint.
        recipe_json = self.model_dump_json()

        artifact_globs = [
            "models/*.pt",
            "models/*.onnx",
            "manifest.json",
            "classes.json",
            "confusion.png",
        ]

        timeout_s = self.compute.timeout_h * 3600.0

        # XOR: use container_image when set, else pip_requirements.
        container_image: str | None = self.compute.container_image
        pip_requirements: list[str] = (
            [] if container_image is not None else list(self.compute.pip_requirements)
        )

        return ComputeJob(
            entry_script="train.py",
            container_image=container_image,
            pip_requirements=pip_requirements,
            working_dir=working_dir,
            env={"RFDF_RECIPE": recipe_json},
            gpu_count=self.compute.gpu_count,
            gpu_min_vram_gb=self.compute.gpu_min_vram_gb,
            gpu_model=self.compute.gpu_model,
            timeout_s=timeout_s,
            priority=self.compute.priority,
            artifact_globs=artifact_globs,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_recipe(path: Path) -> TrainingRecipe:
    """Parse a TOML recipe file into a :class:`TrainingRecipe`.

    The TOML file must contain ``[dataset]``, ``[model]``, ``[training]``, and
    ``[compute]`` sections plus a top-level ``name`` key.  Unknown keys are
    ignored by Pydantic so that forward-compatible recipe files work on older
    versions of the library.

    Args:
        path: Path to the TOML recipe file.

    Returns:
        A validated, frozen :class:`TrainingRecipe`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        pydantic.ValidationError: If the parsed data does not match the schema.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    # Pydantic expects nested sub-models; the TOML sections map directly.
    return TrainingRecipe.model_validate(data)


__all__: list[str] = [
    "ComputeSpec",
    "DatasetSpec",
    "ModelSpec",
    "TrainingRecipe",
    "TrainingSpec",
    "load_recipe",
]

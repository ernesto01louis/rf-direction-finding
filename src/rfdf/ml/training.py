"""Backend-agnostic training loop for RF signal-classification models.

:func:`train` is the single entry point.  It accepts a
:class:`~rfdf.ml.recipes.TrainingRecipe`, pre-built datasets, and an output
directory, and returns a :class:`TrainingResult` describing the best checkpoint.

Key features:

* **Deterministic seeding** — ``random``, ``numpy``, ``torch``, CUDA, and
  DataLoader workers are all seeded from ``training.seed``.
* **Optimizer** — AdamW with weight-decay 0.01.
* **LR schedule** — linear warmup → cosine annealing.
* **Mixed precision** — ``torch.autocast`` + ``GradScaler`` when ``amp=True``
  and CUDA is available.
* **Gradient accumulation** — effective batch = batch x grad_accum_steps.
* **Checkpointing** — saves every ``checkpoint_every_steps`` steps; retains
  only the top-K by val accuracy (``keep_top_k``).
* **DDP** — wraps the model in ``DistributedDataParallel`` and uses
  ``DistributedSampler`` when ``ddp=True`` and ``dist.is_initialized()``.
* **IQ→tensor adapter** — converts complex64 IQ to the shape the model
  expects: 2-tuple ``input_shape`` → ``(2, N)`` raw IQ; 3-tuple → ``(2, F, T)``
  STFT spectrogram.
* **WandB / MLflow** — lazy-imported and used only when their flags are set.

This module imports ``torch`` at module top level and is only ever loaded when a
training run actually happens.  Nothing in ``rfdf.ml.__init__`` imports it.
"""

from __future__ import annotations

import contextlib
import heapq
import logging
import math
import random
import shutil
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from rfdf.ml._manifest import TrainingManifest, current_git_sha, write_manifest
from rfdf.ml.errors import TrainingError
from rfdf.ml.models import build_model
from rfdf.ml.recipes import TrainingRecipe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class TrainingResult:
    """Outcome of a completed :func:`train` call.

    Attributes:
        best_checkpoint: Path to the best model checkpoint (``.pt``).
        best_val_accuracy: Validation accuracy at the best checkpoint.
        manifest_path: Path to ``manifest.json`` written at run end.
        final_step: Total optimizer steps completed.
        wall_clock_s: Wall-clock time in seconds for the full run.
    """

    def __init__(
        self,
        *,
        best_checkpoint: Path,
        best_val_accuracy: float,
        manifest_path: Path,
        final_step: int,
        wall_clock_s: float,
    ) -> None:
        self.best_checkpoint = best_checkpoint
        self.best_val_accuracy = best_val_accuracy
        self.manifest_path = manifest_path
        self.final_step = final_step
        self.wall_clock_s = wall_clock_s

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"TrainingResult(best_val_accuracy={self.best_val_accuracy:.4f}, "
            f"final_step={self.final_step}, "
            f"wall_clock_s={self.wall_clock_s:.1f})"
        )


# ---------------------------------------------------------------------------
# IQ → model-input adapter
# ---------------------------------------------------------------------------


def _iq_to_tensor(iq_complex: Tensor, input_shape: tuple[int, ...]) -> Tensor:
    """Convert a batch of complex64 IQ tensors to the shape the model expects.

    Args:
        iq_complex: ``(batch, N)`` complex64 IQ tensor.
        input_shape: Model's expected input shape, *excluding* batch dimension.
            * 2-tuple ``(2, N)`` → stack real and imaginary as two real channels.
            * 3-tuple ``(2, F, T)`` → compute a short-time Fourier transform
              spectrogram; real and imaginary parts of the STFT output form the
              two channels.

    Returns:
        Real-valued tensor of shape ``(batch, *input_shape)`` on the same device
        as *iq_complex*.

    Raises:
        TrainingError: If *input_shape* has an unsupported number of dimensions.
    """
    batch = iq_complex.shape[0]
    n_samples = iq_complex.shape[1]

    if len(input_shape) == 2:
        # 1-D model: stack I and Q as two channels → (batch, 2, N)
        i_ch = iq_complex.real  # (batch, N)
        q_ch = iq_complex.imag  # (batch, N)
        out: Tensor = torch.stack([i_ch, q_ch], dim=1)  # (batch, 2, N)
        # Resize along the time axis if the model expects a different length.
        target_n = input_shape[1]
        if out.shape[2] != target_n:
            out = torch.nn.functional.interpolate(
                out, size=target_n, mode="linear", align_corners=False
            )
        return out

    if len(input_shape) == 3:
        # 2-D model: compute STFT and form (batch, 2, freq_bins, time_frames).
        _, freq_bins, time_bins = input_shape
        # Choose an STFT window that targets the requested freq/time resolution.
        n_fft = (freq_bins - 1) * 2  # DFT size from desired freq bins
        hop = max(1, n_samples // (time_bins + 1))
        window = torch.hann_window(n_fft, device=iq_complex.device)

        stft_list: list[Tensor] = []
        for b in range(batch):
            # torch.stft requires a real or complex 1-D input.
            sig = iq_complex[b]  # (N,) complex64
            stft_out: Tensor = torch.stft(
                sig,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                return_complex=True,
                center=False,
            )  # (freq_bins, T) complex
            # Stack real and imaginary as two channels.
            spec: Tensor = torch.stack([stft_out.real, stft_out.imag], dim=0)  # (2, freq_bins, T)
            stft_list.append(spec)

        # Stack batch and interpolate to the exact target shape.
        batched: Tensor = torch.stack(stft_list, dim=0)  # (batch, 2, freq_bins, T)
        if batched.shape[2] != freq_bins or batched.shape[3] != time_bins:
            batched = torch.nn.functional.interpolate(
                batched,
                size=(freq_bins, time_bins),
                mode="bilinear",
                align_corners=False,
            )
        return batched

    raise TrainingError(
        f"Unsupported input_shape dimensionality {len(input_shape)} "
        f"(got {input_shape!r}).  Expected a 2-tuple (1-D models) or "
        "3-tuple (2-D models)."
    )


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    """Seed all random-number generators for reproducibility.

    Args:
        seed: Integer seed applied to Python ``random``, NumPy, and PyTorch
            (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker seed function for :class:`~torch.utils.data.DataLoader`.

    Args:
        worker_id: Worker index provided by PyTorch's DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_acc: float,
    output_dir: Path,
) -> Path:
    """Save a model checkpoint and return its path.

    Args:
        model: The model (unwrapped from DDP if necessary).
        optimizer: The current optimizer.
        step: Current optimizer step number.
        val_acc: Validation accuracy at this checkpoint.
        output_dir: Root output directory (``models/`` subdir is used).

    Returns:
        Path to the saved ``.pt`` file.
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"step{step:08d}_acc{val_acc:.4f}.pt"
    actual_model: nn.Module = model.module if hasattr(model, "module") else model  # type: ignore[assignment]
    torch.save(
        {
            "step": step,
            "val_acc": val_acc,
            "model_state_dict": actual_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def _prune_checkpoints(
    heap: list[tuple[float, Path]],
    keep_top_k: int,
) -> None:
    """Remove lowest-accuracy checkpoints beyond ``keep_top_k``.

    The heap stores ``(val_acc, path)``; ``heapq`` is a min-heap, so
    ``heappop`` removes the *lowest*-accuracy (worst) checkpoint first,
    leaving the top-``keep_top_k`` by validation accuracy on disk.

    Args:
        heap: Min-heap of ``(val_acc, path)`` tuples (modified in-place).
        keep_top_k: Maximum number of checkpoints to retain.
    """
    while len(heap) > keep_top_k:
        _, old_path = heapq.heappop(heap)
        if old_path.exists():
            old_path.unlink()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _evaluate(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    input_shape: tuple[int, ...],
) -> float:
    """Run one full pass over *loader* and return top-1 accuracy.

    Args:
        model: The model in eval mode.
        loader: Validation DataLoader.
        device: Torch device.
        input_shape: Model input shape (used by :func:`_iq_to_tensor`).

    Returns:
        Top-1 accuracy in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_iq, batch_labels in loader:
            batch_iq = batch_iq.to(device, dtype=torch.complex64)
            batch_labels = batch_labels.to(device)
            x = _iq_to_tensor(batch_iq, input_shape)
            logits: Tensor = model(x)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch_labels).sum().item())
            total += batch_labels.size(0)
    model.train()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear-warmup + cosine-decay LR lambda scheduler.

    Args:
        optimizer: The optimizer to schedule.
        total_steps: Total number of optimizer steps.
        warmup_steps: Number of linear-warmup steps.

    Returns:
        A :class:`~torch.optim.lr_scheduler.LambdaLR` scheduler.
    """

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Optional tracker helpers
# ---------------------------------------------------------------------------


def _init_wandb(recipe: TrainingRecipe) -> Any:
    """Initialise a Weights & Biases run.

    Args:
        recipe: The training recipe (used as the W&B config).

    Returns:
        The W&B run object, or ``None`` if wandb is not installed.
    """
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("wandb not installed — W&B logging disabled.")
        return None
    run = wandb.init(project="rfdf", name=recipe.name, config=recipe.model_dump())
    return run


def _init_mlflow(recipe: TrainingRecipe) -> None:
    """Start an MLflow run and log the recipe as parameters.

    Args:
        recipe: The training recipe.
    """
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("mlflow not installed — MLflow logging disabled.")
        return
    mlflow.start_run(run_name=recipe.name)
    mlflow.log_params(recipe.model_dump())


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
    *,
    recipe: TrainingRecipe,
    train_dataset: Dataset,  # type: ignore[type-arg]
    val_dataset: Dataset,  # type: ignore[type-arg]
    output_dir: Path,
    device: str | None = None,
    resume_from: Path | None = None,
    progress_cb: Callable[..., None] | None = None,
) -> TrainingResult:
    """Run the backend-agnostic training loop.

    Applies deterministic seeding, builds the model and optimizer, trains for
    the number of epochs specified in ``recipe.training``, evaluates on
    *val_dataset* after each checkpoint interval, keeps the top-K checkpoints
    by validation accuracy, and writes ``manifest.json`` on completion.

    Args:
        recipe: Frozen :class:`~rfdf.ml.recipes.TrainingRecipe` describing the
            model, dataset, and hyper-parameters.
        train_dataset: Training split (yields ``(complex64 IQ tensor, int label)``).
        val_dataset: Validation split (same schema).
        output_dir: Root directory for checkpoints, manifest, and ONNX export.
            Created if it does not exist.
        device: Torch device string (e.g. ``"cuda:0"``).  Auto-detected when
            ``None``: CUDA if available, else CPU.
        resume_from: Optional checkpoint path to resume from.  Loads
            ``model_state_dict`` and ``optimizer_state_dict`` and continues
            from the saved step.
        progress_cb: Optional callback invoked at each step with keyword
            arguments ``step``, ``loss``, ``val_acc`` (the last may be
            ``None`` if no eval has run yet).

    Returns:
        A :class:`TrainingResult` describing the best checkpoint.

    Raises:
        TrainingError: If the training run fails with a hard error (NaN loss,
            etc.).
    """
    t_start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device -----------------------------------------------------------
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    # ---- Seeding ----------------------------------------------------------
    ts = recipe.training
    _seed_everything(ts.seed)

    # ---- Model ------------------------------------------------------------
    ms = recipe.model
    model_raw: nn.Module = build_model(
        ms.architecture,
        num_classes=ms.num_classes,
        input_shape=ms.input_shape,
        **ms.extra_kwargs,
    )
    model_raw = model_raw.to(resolved_device)

    # ---- DDP wrap ---------------------------------------------------------
    use_ddp = ts.ddp and dist.is_available() and dist.is_initialized()
    model: nn.Module = nn.parallel.DistributedDataParallel(model_raw) if use_ddp else model_raw

    # ---- Optimizer --------------------------------------------------------
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ts.learning_rate,
        weight_decay=0.01,
    )

    # ---- AMP --------------------------------------------------------------
    use_amp = ts.amp and resolved_device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ---- Resume -----------------------------------------------------------
    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=resolved_device)
        actual_model: nn.Module = model.module if hasattr(model, "module") else model  # type: ignore[assignment]
        actual_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        logger.info("Resumed from %s at step %d", resume_from, start_step)

    # ---- DataLoaders ------------------------------------------------------
    train_sampler: DistributedSampler[Any] | None = (
        DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    )
    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=ts.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,  # keep deterministic; bump for production
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        pin_memory=(resolved_device.type == "cuda"),
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset,
        batch_size=ts.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(resolved_device.type == "cuda"),
    )

    # ---- LR scheduler -----------------------------------------------------
    steps_per_epoch = max(1, len(train_loader))
    total_steps = ts.epochs * steps_per_epoch
    scheduler = _make_scheduler(optimizer, total_steps, ts.warmup_steps)
    # Advance scheduler to the resume point.
    for _ in range(start_step):
        scheduler.step()

    # ---- Optional trackers ------------------------------------------------
    wandb_run: Any = None
    if ts.wandb:
        wandb_run = _init_wandb(recipe)
    if ts.mlflow:
        _init_mlflow(recipe)

    # ---- Checkpoint heap (min-heap on val_acc; heappop pops the worst) ----
    ckpt_heap: list[tuple[float, Path]] = []
    # -1.0 sentinel: the first checkpoint (val_acc >= 0) always becomes the
    # "best", so models/best.pt is always written after >= 1 epoch.
    best_val_acc = -1.0
    best_ckpt: Path = output_dir / "models" / "best.pt"

    loss_fn = nn.CrossEntropyLoss()
    global_step = start_step
    model.train()

    # ---- Training loop ----------------------------------------------------
    for epoch in range(ts.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step, (batch_iq, batch_labels) in enumerate(train_loader):
            batch_iq = batch_iq.to(resolved_device, dtype=torch.complex64)
            batch_labels = batch_labels.to(resolved_device)

            x = _iq_to_tensor(batch_iq, ms.input_shape)

            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits: Tensor = model(x)
                loss: Tensor = loss_fn(logits, batch_labels)
                loss = loss / ts.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            is_last_micro = (micro_step + 1) % ts.grad_accum_steps == 0
            is_last_batch = micro_step == len(train_loader) - 1

            if is_last_micro or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_loss = 0.0

                # ---- Checkpoint ------------------------------------------------
                should_ckpt = (
                    ts.checkpoint_every_steps > 0 and global_step % ts.checkpoint_every_steps == 0
                )
                if should_ckpt:
                    val_acc = _evaluate(model, val_loader, resolved_device, ms.input_shape)
                    ckpt_path = _save_checkpoint(model, optimizer, global_step, val_acc, output_dir)
                    heapq.heappush(ckpt_heap, (val_acc, ckpt_path))
                    _prune_checkpoints(ckpt_heap, ts.keep_top_k)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        # Copy to a stable name the pruning heap never touches.
                        shutil.copy2(ckpt_path, best_ckpt)

                    logger.info(
                        "step=%d epoch=%d/%d val_acc=%.4f",
                        global_step,
                        epoch + 1,
                        ts.epochs,
                        val_acc,
                    )

                    if progress_cb is not None:
                        progress_cb(step=global_step, loss=loss.item(), val_acc=val_acc)

                    if wandb_run is not None:
                        with contextlib.suppress(Exception):
                            wandb_run.log({"val_acc": val_acc, "step": global_step})
                    if ts.mlflow:
                        try:
                            import mlflow

                            mlflow.log_metric("val_acc", val_acc, step=global_step)
                        except Exception:
                            pass

        # End-of-epoch checkpoint (always).
        epoch_val_acc = _evaluate(model, val_loader, resolved_device, ms.input_shape)
        epoch_ckpt = _save_checkpoint(model, optimizer, global_step, epoch_val_acc, output_dir)
        heapq.heappush(ckpt_heap, (epoch_val_acc, epoch_ckpt))
        _prune_checkpoints(ckpt_heap, ts.keep_top_k)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Copy to a stable name the pruning heap never touches.
            shutil.copy2(epoch_ckpt, best_ckpt)

        logger.info(
            "epoch=%d/%d val_acc=%.4f (end-of-epoch)",
            epoch + 1,
            ts.epochs,
            epoch_val_acc,
        )

    # ---- Manifest ---------------------------------------------------------
    gpu_model_str: str | None = None
    if resolved_device.type == "cuda":
        try:
            gpu_model_str = torch.cuda.get_device_name(resolved_device)
        except Exception:
            gpu_model_str = "unknown-gpu"

    manifest = TrainingManifest(
        training_config=recipe.training.model_dump(),
        dataset_version=recipe.dataset.kind,
        git_sha=current_git_sha(),
        starting_seeds={"torch": ts.seed, "numpy": ts.seed, "python": ts.seed},
        host=socket.gethostname(),
        gpu_model=gpu_model_str,
        evaluation={
            "best_val_accuracy": best_val_acc,
            "final_step": global_step,
        },
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    # ---- Optional: write classes.json -------------------------------------
    classes_path = output_dir / "classes.json"
    if not classes_path.exists() and hasattr(train_dataset, "class_names"):
        with open(classes_path, "w") as fh:
            import json as _json

            _json.dump(train_dataset.class_names, fh, indent=2)
            fh.write("\n")

    # ---- Close optional trackers ------------------------------------------
    if wandb_run is not None:
        with contextlib.suppress(Exception):
            wandb_run.finish()
    if ts.mlflow:
        with contextlib.suppress(Exception):
            import mlflow

            mlflow.end_run()

    wall_clock_s = time.monotonic() - t_start

    return TrainingResult(
        best_checkpoint=best_ckpt,
        best_val_accuracy=best_val_acc,
        manifest_path=manifest_path,
        final_step=global_step,
        wall_clock_s=wall_clock_s,
    )


__all__: list[str] = [
    "TrainingResult",
    "train",
]

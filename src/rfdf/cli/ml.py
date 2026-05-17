"""``rfdf ml`` CLI sub-commands.

Provides two command groups:

* ``rfdf ml train``            — load a recipe, estimate cost, confirm, submit.
* ``rfdf ml registry <sub>``  — list / show / export / import / delete models.
* ``rfdf ml export``           — export a registry model to ONNX / HEF / TFLite / CoreML.

**Critical lazy-import rule:** this module is imported by ``cli/main.py`` at
startup, so it MUST be torch-free at module level.  Every torch / training /
export import goes INSIDE the function body (mirror ``cli/doa.py``).

**Cost-confirmation guardrail:** ``rfdf ml train`` NEVER auto-submits a cloud
job without explicit operator confirmation (``--yes`` or interactive prompt)
UNLESS ``config.compute.require_cost_confirmation`` is ``False``.  This is an
audit guardrail to prevent accidental cloud spend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

ml_app = typer.Typer(
    name="ml",
    help="Train, export, and manage RF signal-classification models.",
    no_args_is_help=True,
    add_completion=False,
)

registry_app = typer.Typer(
    name="registry",
    help="Manage the local model registry (list, show, export, import, delete).",
    no_args_is_help=True,
    add_completion=False,
)
ml_app.add_typer(registry_app, name="registry")

_console = Console()


# ---------------------------------------------------------------------------
# ml train
# ---------------------------------------------------------------------------


@ml_app.command("train")
def train(
    recipe_path: Annotated[
        Path,
        typer.Option(
            "--recipe",
            "-r",
            help="Path to the TOML training recipe.",
            exists=False,  # we check ourselves for a better error message
        ),
    ],
    compute_backend: Annotated[
        str | None,
        typer.Option(
            "--compute",
            "-c",
            help="Compute backend override (e.g. 'local', 'runpod').  "
            "Defaults to the backend in the recipe's [compute] section.",
        ),
    ] = None,
    epochs_override: Annotated[
        int | None,
        typer.Option(
            "--epochs",
            "-e",
            help="Override the recipe's epoch count.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the cost-confirmation prompt and submit immediately.",
        ),
    ] = False,
) -> None:
    """Load a recipe, estimate cost, confirm, and submit a training job.

    The cost estimate is always printed.  Submission is gated behind an
    explicit confirmation unless ``--yes`` is passed or
    ``compute.require_cost_confirmation = false`` in config.

    For ``--compute local`` the training subprocess runs on this machine.
    For cloud backends the job is dispatched to the provider.
    """
    # --- Lazy imports (torch-free at module level) --------------------------
    import asyncio
    import tempfile

    # Recipes are pure pydantic — no torch.
    from rfdf.ml.recipes import load_recipe

    # --- Load recipe -------------------------------------------------------
    if not recipe_path.exists():
        _console.print(f"[red]Recipe not found:[/red] {recipe_path}")
        raise typer.Exit(code=1)

    try:
        recipe = load_recipe(recipe_path)
    except Exception as exc:
        _console.print(f"[red]Failed to load recipe:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # --- Apply CLI overrides -----------------------------------------------
    if epochs_override is not None or compute_backend is not None:
        # TrainingRecipe is frozen; rebuild with pydantic copy().
        recipe_data = recipe.model_dump()
        if epochs_override is not None:
            recipe_data["training"]["epochs"] = epochs_override
        if compute_backend is not None:
            recipe_data["compute"]["backend"] = compute_backend
        from rfdf.ml.recipes import TrainingRecipe

        recipe = TrainingRecipe.model_validate(recipe_data)

    effective_backend = recipe.compute.backend

    # --- Show recipe summary -----------------------------------------------
    _console.print(f"Recipe:  [cyan]{recipe.name}[/cyan]")
    _console.print(f"Backend: [cyan]{effective_backend}[/cyan]")
    _console.print(
        f"Epochs:  {recipe.training.epochs}  |  "
        f"Batch:   {recipe.training.batch_size}  |  "
        f"GPUs:    {recipe.compute.gpu_count}"
    )

    # --- Load compute backend ----------------------------------------------
    from rfdf.hal.discovery import discover_backends

    catalog = discover_backends("rfdf.backends.compute")
    if effective_backend not in catalog:
        available = sorted(catalog)
        _console.print(
            f"[red]Compute backend {effective_backend!r} not found.[/red] "
            f"Available: {available or 'none'}"
        )
        raise typer.Exit(code=1)

    try:
        backend = catalog[effective_backend]()
    except ImportError as exc:
        _console.print(
            f"[yellow]Backend '{effective_backend}' requires an optional SDK "
            f"that is not installed: {exc.name or exc}[/yellow]\n"
            "Install the relevant extra (e.g. pip install 'rfdf\\[ml]') and re-run."
        )
        raise typer.Exit(code=1) from exc

    # --- Build the ComputeJob for estimation --------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        work = Path(tmp_dir)
        job = recipe.to_compute_job(work)

        # --- Cost estimate -------------------------------------------------
        async def _get_estimate() -> object:
            return await backend.cost_estimate(job)

        try:
            est = asyncio.run(_get_estimate())
        except Exception as exc:
            _console.print(f"[yellow]Cost estimate unavailable:[/yellow] {exc}")
            est = None

        if est is not None:
            table = Table(title="Cost estimate")
            table.add_column("field", style="cyan")
            table.add_column("value", justify="right")
            table.add_row("low (USD)", f"${est.low_usd:.4f}")  # type: ignore[attr-defined]
            table.add_row("estimated (USD)", f"${est.estimated_usd:.4f}")  # type: ignore[attr-defined]
            table.add_row("high (USD)", f"${est.high_usd:.4f}")  # type: ignore[attr-defined]
            table.add_row("rationale", est.rationale or "(none)")  # type: ignore[attr-defined]
            _console.print(table)

        # --- Cost-confirmation guardrail -----------------------------------
        from rfdf.config import load_config

        cfg = load_config()
        need_confirm = cfg.compute.require_cost_confirmation and not yes

        if need_confirm:
            confirmed = typer.confirm(
                "Submit the training job?",
                default=False,
            )
            if not confirmed:
                _console.print("[yellow]Submission cancelled.[/yellow]")
                raise typer.Exit(code=0)

        # --- Submit --------------------------------------------------------
        _console.print("Submitting job…")

        async def _submit() -> object:
            return await backend.submit(job)

        try:
            handle = asyncio.run(_submit())
        except Exception as exc:
            _console.print(f"[red]Submission failed:[/red] {type(exc).__name__}: {exc}")
            raise typer.Exit(code=1) from exc

        job_id = getattr(handle, "job_id", str(handle))
        _console.print(
            f"[green]Job submitted.[/green]  Job ID: [cyan]{job_id}[/cyan]\n"
            "Use [cyan]rfdf compute jobs[/cyan] (this session) or the provider "
            "console for status."
        )


# ---------------------------------------------------------------------------
# ml registry list
# ---------------------------------------------------------------------------


@registry_app.command("list")
def registry_list() -> None:
    """List all models registered in the local model registry."""
    # Lazy: registry is pure filesystem / JSON — no torch.
    from rfdf.ml.registry import list_models

    models = list_models()
    if not models:
        _console.print("[yellow]Registry is empty.[/yellow]")
        _console.print(
            "Train a model with [cyan]rfdf ml train[/cyan] and register it "
            "with [cyan]rfdf ml registry import[/cyan]."
        )
        return

    table = Table(title="Registered models")
    table.add_column("model_id", style="cyan")
    for model_id in models:
        table.add_row(model_id)
    _console.print(table)


# ---------------------------------------------------------------------------
# ml registry show
# ---------------------------------------------------------------------------


@registry_app.command("show")
def registry_show(
    model_id: Annotated[str, typer.Argument(help="Model ID to inspect.")],
) -> None:
    """Print the manifest for a registered model."""
    from rfdf.ml.errors import RegistryError
    from rfdf.ml.registry import get_manifest

    try:
        manifest = get_manifest(model_id)
    except RegistryError as exc:
        _console.print(f"[red]Registry error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    import json

    _console.print_json(json.dumps(manifest.model_dump(mode="json"), indent=2, default=str))


# ---------------------------------------------------------------------------
# ml registry export
# ---------------------------------------------------------------------------


@registry_app.command("export")
def registry_export(
    model_id: Annotated[str, typer.Argument(help="Model ID to export.")],
    dest: Annotated[
        Path,
        typer.Option("--to", help="Destination directory or archive path."),
    ] = Path("."),
) -> None:
    """Bundle a registered model into a .tar.gz archive."""
    from rfdf.ml.errors import RegistryError
    from rfdf.ml.registry import export_model

    try:
        archive = export_model(model_id, dest)
    except RegistryError as exc:
        _console.print(f"[red]Registry error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(f"[green]Exported[/green] '{model_id}' → {archive}")


# ---------------------------------------------------------------------------
# ml registry import
# ---------------------------------------------------------------------------


@registry_app.command("import")
def registry_import(
    archive: Annotated[Path, typer.Argument(help="Path to a .tar.gz model archive.")],
) -> None:
    """Import a model archive into the local registry."""
    from rfdf.ml.errors import RegistryError
    from rfdf.ml.registry import import_model

    try:
        model_id = import_model(archive)
    except RegistryError as exc:
        _console.print(f"[red]Registry error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(f"[green]Imported[/green] model '{model_id}' into registry.")


# ---------------------------------------------------------------------------
# ml registry delete
# ---------------------------------------------------------------------------


@registry_app.command("delete")
def registry_delete(
    model_id: Annotated[str, typer.Argument(help="Model ID to delete.")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """Delete a model and all its artefacts from the registry."""
    from rfdf.ml.errors import RegistryError
    from rfdf.ml.registry import delete_model

    if not yes:
        confirmed = typer.confirm(
            f"Delete model '{model_id}' and ALL its artefacts?",
            default=False,
        )
        if not confirmed:
            _console.print("[yellow]Delete cancelled.[/yellow]")
            raise typer.Exit(code=0)

    try:
        delete_model(model_id)
    except RegistryError as exc:
        _console.print(f"[red]Registry error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(f"[green]Deleted[/green] model '{model_id}'.")


# ---------------------------------------------------------------------------
# ml export
# ---------------------------------------------------------------------------


@ml_app.command("export")
def export_model_cmd(
    model_id: Annotated[str, typer.Argument(help="Registry model ID to export.")],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Export format: onnx, hef, tflite, coreml.",
        ),
    ] = "onnx",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path."),
    ] = Path("."),
) -> None:
    """Export a registered model to ONNX, HEF, TFLite, or CoreML format.

    Requires the ``[ml]`` extra (and ``[ml-onnx]``, ``[ml-tflite]``, or
    ``[ml-coreml]`` for the respective format).  The Hailo HEF format
    additionally requires the Hailo Dataflow Compiler on PATH.
    """
    # Validate the export format before importing torch — argument validation
    # must not depend on the [ml] extra being installed.
    fmt_lower = fmt.lower()
    valid_formats = {"onnx", "hef", "tflite", "coreml"}
    if fmt_lower not in valid_formats:
        _console.print(
            f"[red]Unknown format '{fmt}'.[/red] Valid formats: {', '.join(sorted(valid_formats))}"
        )
        raise typer.Exit(code=1)

    # Lazy imports — torch is required for export.
    try:
        import torch
    except ImportError as exc:
        _console.print(
            "[red]Export requires torch.[/red] Install the \\[ml] extra:\n"
            "  pip install 'rfdf\\[ml]'"
        )
        raise typer.Exit(code=1) from exc

    from rfdf.ml.errors import ExportError, RegistryError
    from rfdf.ml.registry import get_manifest, load_model

    # --- Load model --------------------------------------------------------
    try:
        manifest = get_manifest(model_id)
        checkpoint = load_model(model_id)
    except RegistryError as exc:
        _console.print(f"[red]Registry error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Rebuild the model from checkpoint.
    try:
        from rfdf.ml.models import build_model

        # num_classes is not stored in the manifest evaluation record;
        # use 53 (the Sig53 default) as a safe stand-in — build_model only
        # uses it to size the final linear layer, which is then overwritten
        # by load_state_dict below.
        model = build_model(
            manifest.architecture,
            num_classes=53,
            input_shape=(2, 1024),  # default; overridden by load_state_dict
        )
        # Restore weights — the checkpoint is a dict with model_state_dict.
        state = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as exc:
        _console.print(f"[red]Failed to reconstruct model from checkpoint:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Default input shape for representative inference.
    sample = torch.zeros(1, 2, 1024)

    # --- Route to the right exporter ----------------------------------------
    try:
        if fmt_lower == "onnx":
            from rfdf.ml.export import export_onnx

            out_path = output if output.suffix == ".onnx" else output / f"{model_id}.onnx"
            result = export_onnx(model, sample, out_path)
        elif fmt_lower == "hef":
            # HEF requires an ONNX intermediate; produce it in a temp dir.
            import tempfile

            from rfdf.ml.export import export_hailo, export_onnx

            with tempfile.TemporaryDirectory() as tmp:
                onnx_tmp = Path(tmp) / "tmp.onnx"
                export_onnx(model, sample, onnx_tmp)
                out_path = output if output.suffix == ".hef" else output / f"{model_id}.hef"
                result = export_hailo(onnx_tmp, out_path)
        elif fmt_lower == "tflite":
            from rfdf.ml.export import export_tflite

            out_path = output if output.suffix == ".tflite" else output / f"{model_id}.tflite"
            result = export_tflite(model, sample, out_path)
        else:  # coreml
            from rfdf.ml.export import export_coreml

            out_path = (
                output
                if output.suffix in (".mlpackage", ".mlmodel")
                else output / f"{model_id}.mlpackage"
            )
            result = export_coreml(model, sample, out_path)
    except ExportError as exc:
        _console.print(f"[red]Export failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        _console.print(f"[red]Unexpected export error:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(f"[green]Exported[/green] '{model_id}' ({fmt_lower}) → {result}")

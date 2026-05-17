"""``rfdf compute`` CLI sub-commands.

Provides four sub-command groups for managing compute backends:

* ``rfdf compute list``   — list every discovered compute backend.
* ``rfdf compute test``   — submit a trivial no-op job to a named backend.
* ``rfdf compute estimate`` — print a pre-submit cost estimate.
* ``rfdf compute jobs``   — report per-session job state (in-process only).
* ``rfdf compute logs``   — stream / print logs for a job ID.
* ``rfdf compute cancel`` — cancel a job by ID.

**Lazy-import rule:** this module must be torch-free at module level.  All
heavy imports (cloud SDK, torch, training) go inside function bodies.
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

compute_app = typer.Typer(
    name="compute",
    help="Inspect and exercise compute backends for ML training dispatch.",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()

# ---------------------------------------------------------------------------
# In-process job registry (per-session only; a fresh CLI invocation starts
# empty — cloud backends have their own provider consoles for persistence).
# ---------------------------------------------------------------------------
_SESSION_JOBS: dict[str, object] = {}  # job_id -> JobHandle


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@compute_app.command("list")
def list_backends() -> None:
    """List every discovered compute backend with name, GPU support, and availability."""
    # Lazy import — heavy discovery only when actually invoked.
    from rfdf.hal.discovery import discover_backends

    catalog = discover_backends("rfdf.backends.compute")

    table = Table(title="Compute backends")
    table.add_column("name", style="cyan")
    table.add_column("supports_gpu", justify="center")
    table.add_column("status")

    for name, factory in sorted(catalog.items()):
        try:
            backend = factory()
            supports_gpu = "yes" if backend.supports_gpu else "no"
            # Do a cheap availability check: for local we can attempt
            # instantiation (already done); for cloud we check import health.
            status = "available"
        except ImportError as exc:
            supports_gpu = "unknown"
            status = f"needs install: {exc.name or 'unknown'}"
        except Exception as exc:
            supports_gpu = "unknown"
            status = f"error: {type(exc).__name__}"

        table.add_row(name, supports_gpu, status)

    if not catalog:
        _console.print("[yellow]No compute backends discovered.[/yellow]")
        _console.print("Install rfdf with the appropriate extras, e.g. [cyan]rfdf\\[ml][/cyan].")
        return

    _console.print(table)


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


@compute_app.command("test")
def test_backend(
    backend_name: Annotated[
        str, typer.Option("--backend", "-b", help="Backend name to test (e.g. 'local').")
    ] = "local",
) -> None:
    """Submit a trivial no-op job to the named backend and report the outcome.

    For ``local`` this runs a ``print('ok')`` script end-to-end.
    For cloud backends it reports a clear status without crashing when
    credentials or the optional SDK are absent.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from rfdf.hal.compute import ComputeJob
    from rfdf.hal.discovery import discover_backends

    catalog = discover_backends("rfdf.backends.compute")
    if backend_name not in catalog:
        available = sorted(catalog)
        _console.print(
            f"[red]Unknown backend {backend_name!r}.[/red] Available: {available or 'none'}"
        )
        raise typer.Exit(code=1)

    try:
        backend = catalog[backend_name]()
    except ImportError as exc:
        _console.print(
            f"[yellow]Backend '{backend_name}' requires an optional SDK that is not "
            f"installed: {exc.name or exc}[/yellow]\n"
            "Install the relevant extra (e.g. pip install 'rfdf\\[ml]') and re-run."
        )
        raise typer.Exit(code=1) from exc

    _console.print(f"Testing backend: [cyan]{backend_name}[/cyan]")
    t_start = time.monotonic()

    async def _run() -> str:
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "noop.py").write_text("print('rfdf compute test ok')\n")
            job = ComputeJob(entry_script="noop.py", working_dir=work)
            try:
                handle = await backend.submit(job)
            except Exception as exc:
                # Cloud backends may need credentials that are not present in CI /
                # dev environments — report clearly rather than crashing.
                return f"submit failed ({type(exc).__name__}): {exc}"

            for _ in range(300):  # up to 30 s
                status = await backend.status(handle)
                if str(status) in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.1)
            return str(status)

    try:
        result_status = asyncio.run(_run())
    except Exception as exc:
        _console.print(f"[red]Test error:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    elapsed_ms = (time.monotonic() - t_start) * 1000.0
    if result_status == "completed":
        _console.print(
            f"[green]PASS[/green] — backend '{backend_name}' completed the no-op job "
            f"in {elapsed_ms:.0f} ms."
        )
    else:
        _console.print(
            f"[yellow]RESULT:[/yellow] backend '{backend_name}' returned status "
            f"'{result_status}' after {elapsed_ms:.0f} ms.\n"
            "For cloud backends, missing credentials return a non-completed status — "
            "this is expected in environments without provider access."
        )
        if "submit failed" in result_status or result_status not in {
            "completed",
            "running",
            "pending",
        }:
            raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


@compute_app.command("estimate")
def estimate(
    backend_name: Annotated[
        str, typer.Option("--backend", "-b", help="Backend name (e.g. 'local', 'runpod').")
    ] = "local",
    gpu_model: Annotated[
        str | None, typer.Option("--gpu-model", help="GPU model preference (e.g. 'A6000').")
    ] = None,
    gpu_count: Annotated[int, typer.Option("--gpu-count", help="Number of GPUs to request.")] = 1,
    timeout_h: Annotated[
        float, typer.Option("--timeout-h", help="Wall-clock timeout in hours.")
    ] = 6.0,
) -> None:
    """Print a pre-submit cost estimate for a representative training job."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from rfdf.hal.compute import ComputeJob
    from rfdf.hal.discovery import discover_backends

    catalog = discover_backends("rfdf.backends.compute")
    if backend_name not in catalog:
        available = sorted(catalog)
        _console.print(
            f"[red]Unknown backend {backend_name!r}.[/red] Available: {available or 'none'}"
        )
        raise typer.Exit(code=1)

    try:
        backend = catalog[backend_name]()
    except ImportError as exc:
        _console.print(
            f"[yellow]Backend '{backend_name}' requires an optional SDK: {exc.name or exc}[/yellow]"
        )
        raise typer.Exit(code=1) from exc

    async def _estimate() -> object:
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "train.py").write_text("# representative training job\n")
            job = ComputeJob(
                entry_script="train.py",
                working_dir=work,
                gpu_count=gpu_count,
                gpu_model=gpu_model,
                gpu_min_vram_gb=16.0,
                timeout_s=timeout_h * 3600.0,
                pip_requirements=["rfdf[ml]"],
            )
            return await backend.cost_estimate(job)

    try:
        estimate_result = asyncio.run(_estimate())
    except Exception as exc:
        _console.print(f"[red]Estimate error:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Cost estimate — {backend_name}")
    table.add_column("field", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("backend", str(estimate_result.backend))  # type: ignore[attr-defined]
    table.add_row("low (USD)", f"${estimate_result.low_usd:.4f}")  # type: ignore[attr-defined]
    table.add_row("estimated (USD)", f"${estimate_result.estimated_usd:.4f}")  # type: ignore[attr-defined]
    table.add_row("high (USD)", f"${estimate_result.high_usd:.4f}")  # type: ignore[attr-defined]
    table.add_row("rationale", estimate_result.rationale or "(none)")  # type: ignore[attr-defined]
    _console.print(table)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


@compute_app.command("jobs")
def jobs() -> None:
    """List jobs submitted in the current CLI session.

    Job state is in-process only: a fresh CLI invocation starts with an empty
    job list.  For cloud backends, use the provider's console for persistence
    (e.g. RunPod console, Modal dashboard).
    """
    if not _SESSION_JOBS:
        _console.print(
            "[yellow]No jobs in this session.[/yellow]\n"
            "Note: job state is per-session only — a new CLI invocation always "
            "starts empty.  For cloud jobs, visit the provider console for "
            "persistent job listings."
        )
        return

    table = Table(title="Session jobs")
    table.add_column("job_id", style="cyan")
    table.add_column("backend")
    for job_id, handle in _SESSION_JOBS.items():
        backend_name = getattr(handle, "backend", "unknown")
        table.add_row(job_id, str(backend_name))
    _console.print(table)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


@compute_app.command("logs")
def logs(
    job_id: Annotated[str, typer.Argument(help="Job ID returned at submission.")],
) -> None:
    """Print logs for a job submitted in the current CLI session.

    Job handles are per-session only; pass ``job_id`` as returned by the
    backend at submission.  For cloud backends, use the provider console if
    the job was not submitted in this session.
    """
    handle = _SESSION_JOBS.get(job_id)
    if handle is None:
        _console.print(
            f"[red]Job '{job_id}' not found in this session.[/red]\n"
            "Job state is per-session only.  For cloud jobs not in this session, "
            "visit the provider console for log access."
        )
        raise typer.Exit(code=1)

    from rfdf.hal.discovery import discover_backends

    backend_name = getattr(handle, "backend", "local")
    catalog = discover_backends("rfdf.backends.compute")
    if backend_name not in catalog:
        _console.print(f"[red]Backend '{backend_name}' not found.[/red]")
        raise typer.Exit(code=1)

    backend = catalog[backend_name]()

    async def _stream() -> None:
        async for line in backend.logs(handle):
            _console.print(line, end="")

    asyncio.run(_stream())


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


@compute_app.command("cancel")
def cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID returned at submission.")],
) -> None:
    """Cancel a job submitted in the current CLI session.

    For cloud backends, if the job was not submitted in this session, use
    the provider console to cancel it directly.
    """
    handle = _SESSION_JOBS.get(job_id)
    if handle is None:
        _console.print(
            f"[red]Job '{job_id}' not found in this session.[/red]\n"
            "Job state is per-session only.  For cloud jobs not in this session, "
            "cancel via the provider console."
        )
        raise typer.Exit(code=1)

    from rfdf.hal.discovery import discover_backends

    backend_name = getattr(handle, "backend", "local")
    catalog = discover_backends("rfdf.backends.compute")
    if backend_name not in catalog:
        _console.print(f"[red]Backend '{backend_name}' not found.[/red]")
        raise typer.Exit(code=1)

    backend = catalog[backend_name]()

    async def _cancel() -> None:
        await backend.cancel(handle)

    asyncio.run(_cancel())
    _console.print(f"[green]Cancel request sent for job '{job_id}'.[/green]")
    _SESSION_JOBS.pop(job_id, None)

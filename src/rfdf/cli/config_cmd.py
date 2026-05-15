"""``rfdf config`` CLI subcommands.

Two subcommands:

* ``rfdf config show`` — print the resolved RfdfConfig with origin annotations
  (``cli`` / ``env`` / ``toml`` / ``default``). Output format selectable via
  ``--format=table`` (default) or ``--format=json``.
* ``rfdf config validate`` — re-load the config and re-validate; exit 0 on
  success, exit 1 on validation failure with a clear error message.
"""

from __future__ import annotations

import json
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from rfdf.config import load_config, resolve_value_sources

config_app = typer.Typer(
    name="config",
    help="Inspect + validate the resolved rfdf configuration.",
    no_args_is_help=True,
    add_completion=False,
)


@config_app.command("show")
def show(
    fmt: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table (rich) or json.",
        case_sensitive=False,
    ),
) -> None:
    """Print resolved config with source annotations."""
    cfg = load_config()
    sources = resolve_value_sources()
    payload: dict[str, dict[str, Any]] = {}
    for section, source in sources.items():
        section_model = getattr(cfg, section)
        payload[section] = {
            "source": source,
            "values": section_model.model_dump(mode="json"),
        }

    fmt_lc = fmt.lower()
    if fmt_lc == "json":
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if fmt_lc != "table":
        raise typer.BadParameter(f"unknown format {fmt!r} (expected: table, json)")
    console = Console()
    for section, entry in payload.items():
        table = Table(title=f"[bold]{section}[/bold]  (source: {entry['source']})")
        table.add_column("key", style="cyan")
        table.add_column("value")
        for key, value in entry["values"].items():
            table.add_row(key, repr(value))
        console.print(table)


@config_app.command("validate")
def validate() -> None:
    """Re-load and re-validate the config; exit non-zero on failure."""
    try:
        load_config()
    except ValidationError as exc:
        typer.echo(f"config invalid:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("config OK")

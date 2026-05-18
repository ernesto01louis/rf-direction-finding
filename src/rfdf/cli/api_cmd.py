"""``rfdf api`` CLI subcommands (Stage 7 R8).

``rfdf api serve`` runs the optional rfdf REST API (the ``[api]`` extra).
This is the entry point the ``rfdf-api`` systemd unit invokes.
"""

from __future__ import annotations

from typing import Annotated

import typer

api_app = typer.Typer(
    name="api",
    help="Optional rfdf REST API (requires the [api] extra).",
    no_args_is_help=True,
)


@api_app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8001,
) -> None:
    """Serve the rfdf REST API + capability server.

    Exits with an install hint when the ``[api]`` extra is absent.
    """
    try:
        import uvicorn
    except ImportError:
        typer.echo("rfdf REST API not available.")
        typer.echo("Install with: pip install rfdf[api]")
        raise typer.Exit(code=1) from None

    from rfdf.api import create_app

    uvicorn.run(create_app(), host=host, port=port)

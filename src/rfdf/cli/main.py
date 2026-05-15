"""rfdf CLI entry point (Stage 1 minimal — only ``--version`` is wired)."""

from __future__ import annotations

import typer

from rfdf import __version__

app = typer.Typer(
    name="rfdf",
    help="Hardware-agnostic RF direction finding, signal classification, and phased-array research.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the rfdf version and exit.",
    ),
) -> None:
    """Entry point for the ``rfdf`` command-line interface.

    Subcommands land in later stages. Run ``rfdf --version`` for the installed version.
    """
    _ = version  # parameter consumed by the callback


if __name__ == "__main__":  # pragma: no cover
    app()

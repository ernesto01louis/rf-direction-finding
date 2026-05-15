"""rfdf CLI entry point.

Stage 2 wires ``rfdf hw {list-backends, selftest}`` and ``rfdf config
{show, validate}`` subcommands. Subsequent stages add the DSP / ML / hardware
subcommands documented in ``src/rfdf/cli/__init__.py``.
"""

from __future__ import annotations

import typer

from rfdf import __version__
from rfdf.cli.config_cmd import config_app
from rfdf.cli.doa import doa_app
from rfdf.cli.hw import hw_app

app = typer.Typer(
    name="rfdf",
    help="Hardware-agnostic RF direction finding, signal classification, and phased-array research.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(hw_app, name="hw")
app.add_typer(config_app, name="config")
app.add_typer(doa_app, name="doa")


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

    Run ``rfdf --version`` for the installed version, or ``rfdf <subcommand>
    --help`` for subcommand-specific help.
    """
    _ = version  # parameter consumed by the callback


if __name__ == "__main__":  # pragma: no cover
    app()

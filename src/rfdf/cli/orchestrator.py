"""``rfdf orchestrator`` CLI subcommands (Stage 7 R8).

Optional orchestrator integration from the command line:

* ``rfdf orchestrator status``   — connection state + declared capabilities
* ``rfdf orchestrator register`` — register this platform as a consumer
* ``rfdf orchestrator hindsight`` — manual Hindsight memory write
* ``rfdf orchestrator vault``     — manual L5 vault note
* ``rfdf orchestrator planner``   — request a flowgraph from the planner

Every subcommand degrades gracefully when the ``[orchestrator]`` extra
is absent: it prints an install hint and exits non-zero rather than
raising an ImportError.

Connection settings come from the environment: ``ORCHESTRATOR_URL``
(default ``http://localhost:8000``) and ``ORCHESTRATOR_TOKEN``.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import typer

orchestrator_app = typer.Typer(
    name="orchestrator",
    help="Optional ai-orchestrator integration (requires the [orchestrator] extra).",
    no_args_is_help=True,
)


def _require_orchestrator() -> None:
    """Exit with an install hint when the optional extra is missing."""
    from rfdf.orchestrator import is_available

    if not is_available():
        typer.echo("Orchestrator integration not available.")
        typer.echo("Install with: pip install rfdf[orchestrator]")
        raise typer.Exit(code=1)


def _make_client() -> Any:
    """Build an OrchestratorClient from the environment."""
    from ai_orchestrator_client import BearerTokenAuth, OrchestratorClient

    base_url = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
    token = os.environ.get("ORCHESTRATOR_TOKEN", "").strip()
    auth = BearerTokenAuth(token) if token else None
    return OrchestratorClient(base_url=base_url, auth=auth)


@orchestrator_app.command("status")
def status() -> None:
    """Show orchestrator connection state and declared capabilities."""
    _require_orchestrator()
    from rfdf.orchestrator import RfdfConsumer

    consumer = RfdfConsumer()
    typer.echo(f"consumer:      {consumer.name}")
    typer.echo(f"capabilities:  {', '.join(consumer.capabilities)}")
    base_url = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
    typer.echo(f"orchestrator:  {base_url}")
    try:
        with _make_client() as client:
            client.health()
        typer.echo("connection:    reachable")
    except Exception as exc:
        typer.echo(f"connection:    unreachable ({exc})")


@orchestrator_app.command("register")
def register(
    base_url: Annotated[
        str, typer.Option(help="This platform's externally reachable base URL.")
    ] = "http://localhost:8001",
    callback_token: Annotated[
        str, typer.Option(help="Token the orchestrator presents on dispatch.")
    ] = "",
) -> None:
    """Register this rfdf instance as an orchestrator consumer."""
    _require_orchestrator()
    from rfdf.orchestrator import RfdfConsumer
    from rfdf.orchestrator.consumer import DATA_PLANE_CAPABILITIES

    consumer = RfdfConsumer()
    with _make_client() as client:
        ack = consumer.register(
            client,
            base_url,
            callback_token=callback_token or None,
            description="rf-direction-finding platform",
            extra_capabilities=DATA_PLANE_CAPABILITIES,
        )
    typer.echo(f"registered: {ack.status}")
    typer.echo(f"capabilities: {', '.join(ack.consumer.capabilities)}")


@orchestrator_app.command("hindsight")
def hindsight(
    content: Annotated[str, typer.Option(help="Memory narrative to retain.")],
) -> None:
    """Write a Hindsight memory entry (debugging helper)."""
    _require_orchestrator()
    from rfdf.orchestrator import Hindsight

    with _make_client() as client:
        Hindsight(client, "rfdf").write(content)
    typer.echo("memory written")


@orchestrator_app.command("vault")
def vault(
    title: Annotated[str, typer.Option(help="Vault note title.")],
    body: Annotated[str, typer.Option(help="Vault note body (markdown).")],
) -> None:
    """Write an L5 vault note (debugging helper)."""
    _require_orchestrator()
    from rfdf.orchestrator import Vault

    with _make_client() as client:
        Vault(client, "rfdf").write_note(title, body, tags=["rfdf"])
    typer.echo("vault note written")


@orchestrator_app.command("planner")
def planner(
    prompt: Annotated[str, typer.Option(help="Flowgraph request for the planner.")],
    name: Annotated[str, typer.Option(help="Flowgraph name.")] = "flowgraph",
) -> None:
    """Request a GNU Radio flowgraph from the orchestrator's planner."""
    _require_orchestrator()
    from rfdf.orchestrator import FlowgraphBridge

    with _make_client() as client:
        bridge = FlowgraphBridge(client)
        flowgraph = bridge.generate(prompt, name=name)
        result = bridge.validate(flowgraph)
    typer.echo(f"flowgraph:  {flowgraph.name} (run {flowgraph.run_id})")
    typer.echo(f"validation: {'ok' if result.ok else 'failed'}")
    if not result.ok:
        typer.echo(result.output)

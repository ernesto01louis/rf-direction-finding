"""Planner-dispatched GNU Radio flowgraphs (Stage 7 R5).

:class:`FlowgraphBridge` lets the orchestrator's planner LLM generate a
GNU Radio flowgraph on demand, validates it locally with ``grcc
--no-execute``, and deploys the validated ``.grc`` to the rfdf-tools
host.

There is no SDK API for an ad-hoc planner prompt, so :meth:`generate`
wraps the orchestrator's existing run pipeline: an ``OrchestrateRequest``
carrying a flowgraph-shaped prompt is dispatched with ``client.run``,
and the generated artifact is read back from the run result.

``grcc`` ships with GNU Radio and is **not** present in the rfdf build
environment — :meth:`validate` and :meth:`deploy` are unit-tested with a
mocked subprocess; the live path is an operator-verified checklist item.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ai_orchestrator_client import OrchestratorClient

# The orchestrator's coding planner. Overridable via FlowgraphBridge args.
_DEFAULT_PLANNER_MODEL = "qwen2.5-coder:32b"

_PROMPT_TEMPLATE = (
    "Generate a GNU Radio Companion flowgraph as GRC XML.\n"
    "Requirement: {request}\n"
    "Return only the .grc XML document — no prose, no code fences."
)


@dataclass
class Flowgraph:
    """A generated GNU Radio flowgraph."""

    name: str
    grc_xml: str
    run_id: str | None = None


@dataclass
class ValidationResult:
    """Outcome of a ``grcc --no-execute`` validation pass."""

    ok: bool
    output: str


@dataclass
class DeployHandle:
    """Where a validated flowgraph was deployed."""

    host: str
    path: str


def _extract_grc(result: dict[str, Any]) -> str:
    """Pull the GRC XML out of an orchestrator run result.

    The generator returns ``{"files": {name: content}}``; prefer a
    ``.grc`` file, fall back to the first file, then to the raw result.
    """
    files = result.get("files") if isinstance(result, dict) else None
    if isinstance(files, dict) and files:
        for name, content in files.items():
            if str(name).endswith((".grc", ".grc.xml")):
                return str(content)
        return str(next(iter(files.values())))
    return str(result)


class FlowgraphBridge:
    """Bridge the orchestrator's planner to GNU Radio flowgraph builds."""

    def __init__(
        self,
        client: OrchestratorClient,
        *,
        planner_model: str = _DEFAULT_PLANNER_MODEL,
        generator_model: str = _DEFAULT_PLANNER_MODEL,
        judge_model: str = _DEFAULT_PLANNER_MODEL,
        deploy_target: str = "local",
    ) -> None:
        self._client = client
        self._planner_model = planner_model
        self._generator_model = generator_model
        self._judge_model = judge_model
        self._deploy_target = deploy_target

    def generate(
        self,
        request: str,
        *,
        name: str = "flowgraph",
        project_name: str = "rfdf-flowgraph",
        timeout: float = 600.0,
    ) -> Flowgraph:
        """Ask the orchestrator's planner to generate a flowgraph.

        Dispatches an ``OrchestrateRequest`` with a flowgraph-shaped
        prompt, waits for completion, and reads the GRC XML back from
        the run result.
        """
        from ai_orchestrator_client import OrchestrateRequest

        req = OrchestrateRequest(
            project_name=project_name,
            prompt=_PROMPT_TEMPLATE.format(request=request),
            planner_model=self._planner_model,
            generator_models=[self._generator_model],
            judge_model=self._judge_model,
            deploy_target=self._deploy_target,
        )
        ack = self._client.run(req)
        self._client.wait_for_completion(ack.run_id, timeout=timeout)
        result = self._client.get_result(ack.run_id)
        return Flowgraph(name=name, grc_xml=_extract_grc(result), run_id=ack.run_id)

    def validate(self, flowgraph: Flowgraph) -> ValidationResult:
        """Validate a flowgraph with ``grcc --no-execute``.

        Writes the GRC XML to a temp file and compiles it without
        running. A missing ``grcc`` (GNU Radio not installed) yields
        ``ok=False`` with a clear message rather than an exception.
        """
        with tempfile.TemporaryDirectory() as tmp:
            grc_path = Path(tmp) / f"{flowgraph.name}.grc"
            grc_path.write_text(flowgraph.grc_xml, encoding="utf-8")
            try:
                proc = subprocess.run(
                    ["grcc", "--no-execute", "-o", tmp, str(grc_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except FileNotFoundError:
                return ValidationResult(
                    ok=False,
                    output="grcc not found — GNU Radio is not installed",
                )
            except subprocess.SubprocessError as exc:
                return ValidationResult(ok=False, output=f"grcc failed: {exc}")
            output = (proc.stdout + proc.stderr).strip()
            return ValidationResult(ok=proc.returncode == 0, output=output)

    def deploy(self, flowgraph: Flowgraph, *, target_host: str = "rfdf-tools") -> DeployHandle:
        """Deploy a validated flowgraph to the rfdf-tools host via scp.

        The flowgraph should pass :meth:`validate` first. Raises
        ``RuntimeError`` if the scp transfer fails.
        """
        remote_path = f"/opt/rfdf-flowgraphs/{flowgraph.name}.grc"
        with tempfile.TemporaryDirectory() as tmp:
            grc_path = Path(tmp) / f"{flowgraph.name}.grc"
            grc_path.write_text(flowgraph.grc_xml, encoding="utf-8")
            proc = subprocess.run(
                [
                    "scp",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    str(grc_path),
                    f"{target_host}:{remote_path}",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"flowgraph deploy to {target_host} failed: {proc.stderr.strip()}")
        return DeployHandle(host=target_host, path=remote_path)

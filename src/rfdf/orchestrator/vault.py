"""Hindsight memory + L5 vault writes for rfdf (Stage 7 R4).

:class:`RfdfRecorder` turns rfdf artifacts — detections, calibrations,
geometry presets, model cards, capture campaigns — into orchestrator
memory entries and wiki-linked L5 vault notes. Over time the vault notes
accumulate into the operator's RF research diary.

Every method is **fail-tolerant**: a write to an unreachable
orchestrator logs nothing fatal, returns a status dict, and never
raises. A standalone platform keeps working when the orchestrator is
down — the writes are bonus context, not a hard dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ai_orchestrator_client import OrchestratorClient

_logger = logging.getLogger(__name__)


class RfdfRecorder:
    """Writes rfdf artifacts to orchestrator memory + the L5 vault.

    Bind once to an :class:`~ai_orchestrator_client.OrchestratorClient`;
    the per-method calls are thin wrappers over the SDK's ``Hindsight``
    and ``Vault`` thin clients.
    """

    def __init__(self, client: OrchestratorClient, *, consumer_id: str = "rfdf") -> None:
        from ai_orchestrator_client import Hindsight, Vault

        self._hindsight = Hindsight(client, consumer_id)
        self._vault = Vault(client, consumer_id)

    # ----- internal fail-tolerant primitives --------------------------

    def _safe_memory(self, content: str) -> dict[str, Any]:
        from ai_orchestrator_client import OrchestratorError

        try:
            self._hindsight.write(content)
            return {"memory": "ok"}
        except (OrchestratorError, OSError) as exc:
            _logger.warning("rfdf memory write failed: %s", exc)
            return {"memory": "failed", "error": str(exc)}

    def _safe_note(self, title: str, body: str, tags: list[str]) -> dict[str, Any]:
        from ai_orchestrator_client import OrchestratorError

        try:
            self._vault.write_note(title, body, tags=tags)
            return {"vault": "ok"}
        except (OrchestratorError, OSError) as exc:
            _logger.warning("rfdf vault write failed: %s", exc)
            return {"vault": "failed", "error": str(exc)}

    # ----- public record_* helpers ------------------------------------

    def record_detection(
        self,
        *,
        bearing_deg: float,
        frequency_hz: float,
        modulation: str = "unknown",
        confidence: float = 1.0,
        bundle_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an RF detection as a memory entry + an rf-detection note."""
        summary = (
            f"Detected emitter at {bearing_deg:.1f} deg, "
            f"{frequency_hz / 1e6:.1f} MHz, classified as {modulation} "
            f"(confidence {confidence:.2f})."
        )
        body = (
            f"- Bearing: {bearing_deg:.2f} deg\n"
            f"- Frequency: {frequency_hz / 1e6:.3f} MHz\n"
            f"- Modulation: {modulation}\n"
            f"- Confidence: {confidence:.3f}\n"
            f"- Evidence bundle: {bundle_id or 'n/a'}\n"
        )
        result = self._safe_memory(summary)
        result.update(
            self._safe_note(
                f"detection-{frequency_hz / 1e6:.0f}MHz-{bearing_deg:.0f}deg",
                body,
                ["rf-detection", modulation],
            )
        )
        return result

    def record_calibration(
        self, *, name: str, procedure: str, geometry_hash: str, body: str = ""
    ) -> dict[str, Any]:
        """Record an array calibration as an rf-calibration vault note."""
        note_body = body or (f"- Procedure: {procedure}\n- Geometry hash: {geometry_hash}\n")
        return self._safe_note(f"calibration-{name}", note_body, ["rf-calibration", procedure])

    def record_geometry_preset(
        self, *, name: str, preset: str, num_elements: int, body: str = ""
    ) -> dict[str, Any]:
        """Record an antenna-geometry preset as an rf-geometry-preset note."""
        note_body = body or (f"- Preset: {preset}\n- Elements: {num_elements}\n")
        return self._safe_note(f"geometry-{name}", note_body, ["rf-geometry-preset", preset])

    def record_model_card(
        self,
        *,
        model_id: str,
        architecture: str,
        metrics: dict[str, Any] | None = None,
        body: str = "",
    ) -> dict[str, Any]:
        """Record a trained model as an rf-model-card vault note."""
        metric_lines = "\n".join(f"- {k}: {v}" for k, v in (metrics or {}).items())
        note_body = body or (f"- Architecture: {architecture}\n{metric_lines}\n")
        return self._safe_note(f"model-{model_id}", note_body, ["rf-model-card", architecture])

    def record_campaign(self, *, campaign_id: str, summary: str, body: str = "") -> dict[str, Any]:
        """Record a capture campaign as an rf-campaign vault note."""
        return self._safe_note(f"campaign-{campaign_id}", body or summary, ["rf-campaign"])

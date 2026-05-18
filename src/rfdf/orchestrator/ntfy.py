"""ntfy alerting for rfdf (Stage 7 R6).

:class:`RfdfAlerts` pushes interesting RF events to the orchestrator's
notification service over three logical channels:

* ``alerts``   — high priority: unusual emitters, calibration drift,
  hardware errors.
* ``ops``      — normal: capture started/stopped, training completed,
  model registered.
* ``research`` — low priority: scientific findings, novel signal
  patterns, long-term trend changes.

The orchestrator's ``/consumers/{id}/notify`` endpoint is a single
notification sink, so the channel is carried as a title prefix
(``[rfdf-alerts]``) plus a tag and selects a default priority.

Every call is **fail-tolerant** — a notification to an unreachable
orchestrator returns a status dict and never raises.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ai_orchestrator_client import OrchestratorClient

_logger = logging.getLogger(__name__)

# channel -> default ntfy priority (1 min .. 5 max).
_CHANNEL_PRIORITY = {"alerts": 4, "ops": 3, "research": 2}


class RfdfAlerts:
    """Sends rfdf event notifications over the orchestrator's ntfy."""

    def __init__(self, client: OrchestratorClient, *, consumer_id: str = "rfdf") -> None:
        from ai_orchestrator_client import Ntfy

        self._ntfy = Ntfy(client, consumer_id)

    def alert(
        self,
        channel: str,
        title: str,
        message: str,
        *,
        priority: int | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send a notification on one of the three rfdf channels.

        ``channel`` must be ``alerts``, ``ops``, or ``research``.
        ``priority`` overrides the channel default. Never raises.
        """
        if channel not in _CHANNEL_PRIORITY:
            raise ValueError(f"unknown channel {channel!r}; choose 'alerts', 'ops', or 'research'")
        from ai_orchestrator_client import OrchestratorError

        effective_priority = priority if priority is not None else _CHANNEL_PRIORITY[channel]
        channel_tags = [f"rfdf-{channel}", *(tags or [])]
        try:
            self._ntfy.alert(
                f"[rfdf-{channel}] {title}",
                message,
                priority=effective_priority,
                tags=channel_tags,
            )
            return {"channel": channel, "status": "sent"}
        except (OrchestratorError, OSError) as exc:
            _logger.warning("rfdf ntfy alert failed: %s", exc)
            return {"channel": channel, "status": "failed", "error": str(exc)}

    def alerts(self, title: str, message: str, **kwargs: Any) -> dict[str, Any]:
        """High-priority alert (unusual emitter, calibration drift, …)."""
        return self.alert("alerts", title, message, **kwargs)

    def ops(self, title: str, message: str, **kwargs: Any) -> dict[str, Any]:
        """Normal-priority operational notification."""
        return self.alert("ops", title, message, **kwargs)

    def research(self, title: str, message: str, **kwargs: Any) -> dict[str, Any]:
        """Low-priority research finding notification."""
        return self.alert("research", title, message, **kwargs)

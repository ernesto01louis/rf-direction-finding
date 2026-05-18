"""Real orchestrator integration — imports ``ai_orchestrator_client``.

This module and its siblings under ``rfdf.orchestrator`` (``consumer``,
``evidence``, ``planner``, …) are the **only** modules under
``src/rfdf/`` that import ``ai_orchestrator_client`` at module scope.
Everything else reaches them lazily through
:func:`rfdf.orchestrator.__getattr__`, so a base install (no
``[orchestrator]`` extra) never imports the optional dependency.

Importing this module when the extra is absent raises ``ImportError`` —
callers go through :mod:`rfdf.orchestrator`, which checks
:func:`~rfdf.orchestrator.availability.is_available` first and raises a
friendly :class:`~rfdf.orchestrator.availability.OrchestratorNotAvailableError`.
"""

from __future__ import annotations

import ai_orchestrator_client as aoc

from .consumer import RfdfConsumer
from .evidence import RfdfEvidenceBundle, build_bundle
from .vault import RfdfRecorder

# Re-export the SDK's consumer-integration surface so callers use a
# single import root: ``from rfdf.orchestrator import Consumer``.
Consumer = aoc.Consumer
Hindsight = aoc.Hindsight
Ntfy = aoc.Ntfy
Vault = aoc.Vault
capability = aoc.capability

__all__ = [
    "Consumer",
    "Hindsight",
    "Ntfy",
    "RfdfConsumer",
    "RfdfEvidenceBundle",
    "RfdfRecorder",
    "Vault",
    "build_bundle",
    "capability",
]

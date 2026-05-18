"""Real orchestrator integration — imports ``ai_orchestrator_client``.

This is the **only** module under ``src/rfdf/`` that imports
``ai_orchestrator_client`` at module scope. Everything else reaches it
lazily through :func:`rfdf.orchestrator.__getattr__`, so a base install
(no ``[orchestrator]`` extra) never imports the optional dependency.

Importing this module when the extra is absent raises ``ImportError`` —
callers go through :mod:`rfdf.orchestrator`, which checks
:func:`~rfdf.orchestrator.availability.is_available` first and raises a
friendly :class:`~rfdf.orchestrator.availability.OrchestratorNotAvailableError`.
"""

from __future__ import annotations

import ai_orchestrator_client as aoc

# Re-export the SDK's consumer-integration surface so callers use a
# single import root: ``from rfdf.orchestrator import Consumer``.
Consumer = aoc.Consumer
Hindsight = aoc.Hindsight
Ntfy = aoc.Ntfy
Vault = aoc.Vault
capability = aoc.capability

__all__ = ["Consumer", "Hindsight", "Ntfy", "Vault", "capability"]

"""Optional orchestrator integration for rfdf (Stage 7).

The platform is **standalone-first**: it runs fully without
``ai-orchestrator-client``. Installing the ``[orchestrator]`` extra adds
consumer registration, evidence-bundle production, Hindsight memory
writes, L5 vault notes, planner-dispatched GNU Radio flowgraphs, and
ntfy alerts.

``ai_orchestrator_client`` is imported **lazily** — only inside
:mod:`rfdf.orchestrator._real`, and only when one of the integration
classes below is first accessed. Importing this module does NOT pull in
the optional dependency, so::

    import rfdf.orchestrator            # always succeeds

while::

    rfdf.orchestrator.RfdfConsumer      # raises OrchestratorNotAvailableError
                                        # with a clear install hint when the
                                        # extra is missing

Use :func:`is_available` to gate orchestrator code paths without
forcing the import.
"""

from __future__ import annotations

from typing import Any

from .availability import OrchestratorNotAvailableError, is_available

# OrchestratorNotAvailableError / is_available are eager; the rest are
# lazily resolved via __getattr__ and present only with [orchestrator].
__all__ = [
    "Consumer",
    "Hindsight",
    "Ntfy",
    "OrchestratorNotAvailableError",
    "RfdfConsumer",
    "RfdfEvidenceBundle",
    "Vault",
    "build_bundle",
    "capability",
    "is_available",
]

# Names resolved on first access from rfdf.orchestrator._real, which is
# the only module that imports ai_orchestrator_client. Grows as the
# Stage 7 sub-features land (RfdfConsumer, EvidenceBundle, FlowgraphBridge, …).
_LAZY_NAMES = frozenset(
    {
        "Consumer",
        "Hindsight",
        "Ntfy",
        "RfdfConsumer",
        "RfdfEvidenceBundle",
        "Vault",
        "build_bundle",
        "capability",
    }
)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for orchestrator integration names."""
    if name in _LAZY_NAMES:
        if not is_available():
            raise OrchestratorNotAvailableError(
                f"rfdf.orchestrator.{name} requires the "
                f"`ai-orchestrator-client` package. Install it with "
                f"`pip install rfdf[orchestrator]`."
            )
        from . import _real

        return getattr(_real, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Optional-dependency gate for ``rfdf.orchestrator``.

Importing :mod:`rfdf.orchestrator` never raises — it stays importable
for type-checking and feature detection even when the optional
``ai-orchestrator-client`` package is absent. Code paths that actually
need the orchestrator gate themselves on :func:`is_available` or catch
:class:`OrchestratorNotAvailableError`.
"""

from __future__ import annotations


def is_available() -> bool:
    """Return True if ``ai-orchestrator-client`` is importable.

    This is the standalone-vs-orchestrator switch: the platform runs
    fully without the optional dependency, and every orchestrator code
    path checks this first.
    """
    try:
        import ai_orchestrator_client  # noqa: F401
    except ImportError:
        return False
    return True


class OrchestratorNotAvailableError(RuntimeError):
    """Raised when an orchestrator feature is used without the extra.

    Install the optional dependency with ``pip install rfdf[orchestrator]``.
    """

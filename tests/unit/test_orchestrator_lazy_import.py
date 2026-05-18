"""Tests for the rfdf.orchestrator lazy-import wrapper (Stage 7).

The standalone-first contract: ``import rfdf.orchestrator`` always
succeeds; accessing an integration class without the ``[orchestrator]``
extra raises ``OrchestratorNotAvailableError`` with a clear install
hint.
"""

from __future__ import annotations

import pytest

import rfdf.orchestrator as orch
from rfdf.orchestrator.availability import (
    OrchestratorNotAvailableError,
    is_available,
)


def test_import_orchestrator_module_always_succeeds() -> None:
    # The module imported at the top of this file — reaching here proves it.
    assert hasattr(orch, "is_available")
    assert hasattr(orch, "OrchestratorNotAvailableError")


def test_is_available_returns_bool() -> None:
    assert isinstance(is_available(), bool)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = orch.does_not_exist


def test_lazy_name_raises_friendly_error_when_unavailable(monkeypatch) -> None:
    """With the optional dep reported absent, a lazy name raises the
    friendly error rather than a bare ImportError."""
    monkeypatch.setattr(orch, "is_available", lambda: False)
    with pytest.raises(OrchestratorNotAvailableError) as exc:
        _ = orch.Consumer
    assert "pip install rfdf[orchestrator]" in str(exc.value)


@pytest.mark.skipif(
    not is_available(),
    reason="ai-orchestrator-client not installed (base install)",
)
def test_lazy_names_resolve_when_available() -> None:
    """With the extra installed, the lazy names resolve to the SDK surface."""
    import ai_orchestrator_client as aoc

    assert orch.Consumer is aoc.Consumer
    assert orch.capability is aoc.capability
    assert orch.Hindsight is aoc.Hindsight
    assert orch.Vault is aoc.Vault
    assert orch.Ntfy is aoc.Ntfy

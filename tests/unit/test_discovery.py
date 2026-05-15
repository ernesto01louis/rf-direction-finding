"""Unit tests for the backend discovery helper.

The three failure modes documented in ``src/rfdf/hal/discovery.py`` MUST surface as
WARN-level logs + skip the backend (never raise out of ``discover_backends``):

1. Entry point ``.load()`` raises ``ImportError`` (missing optional dep).
2. The loaded target is not callable.
3. Two backends register the same name (first-wins, both distributions logged).

A fourth mode, factory raises at call time, is tested via ``load_backend`` (where
``BackendLoadError`` MUST wrap the original exception).
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint
from types import ModuleType
from typing import Any

import pytest

from rfdf.hal import BackendLoadError, discover_backends, list_backends, load_backend
from rfdf.hal.discovery import BACKEND_GROUPS


def _make_entry_point(name: str, value: str, group: str) -> EntryPoint:
    """Construct an EntryPoint that pytest can dispatch through entry_points()."""
    return EntryPoint(name=name, value=value, group=group)


def test_backend_groups_contains_all_four() -> None:
    """The canonical groups must include the four HAL Protocols."""
    assert set(BACKEND_GROUPS) == {
        "rfdf.backends.sdr",
        "rfdf.backends.rotator",
        "rfdf.backends.geometry",
        "rfdf.backends.compute",
    }


def test_discover_backends_unknown_group_returns_empty() -> None:
    """A group with no registered entry points returns an empty dict, not an error."""
    result = discover_backends("rfdf.backends.does-not-exist")
    assert result == {}


def test_list_backends_returns_all_groups() -> None:
    """list_backends emits one entry per canonical group, even if empty."""
    catalog = list_backends()
    assert set(catalog) == set(BACKEND_GROUPS)
    for value in catalog.values():
        assert isinstance(value, list)


def test_load_backend_unknown_name_raises_keyerror() -> None:
    """Asking for a name that isn't registered yields KeyError (not BackendLoadError)."""
    with pytest.raises(KeyError, match="not-a-real-backend"):
        load_backend("rfdf.backends.sdr", "not-a-real-backend")


# ---------------------------------------------------------------------------
# Failure mode 1: entry point .load() raises ImportError
# ---------------------------------------------------------------------------


def test_discover_skips_entry_point_when_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken entry point must NOT crash discovery — log + skip."""
    broken = _make_entry_point(
        "broken",
        "rfdf_broken_module:create",
        "rfdf.backends.sdr",
    )

    def fake_entry_points(*, group: str) -> tuple[EntryPoint, ...]:
        if group == "rfdf.backends.sdr":
            return (broken,)
        return ()

    monkeypatch.setattr("rfdf.hal.discovery.entry_points", fake_entry_points)
    catalog = discover_backends("rfdf.backends.sdr")
    assert catalog == {}


# ---------------------------------------------------------------------------
# Failure mode 2: entry point loads to a non-callable target
# ---------------------------------------------------------------------------


def test_discover_skips_non_callable_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry point pointing at a non-callable (e.g. typo, module attribute) is skipped."""
    fake_module = ModuleType("rfdf_fake_noncallable")
    fake_module.NOT_A_FUNCTION = "this is a string, not a callable"  # type: ignore[attr-defined]
    sys.modules["rfdf_fake_noncallable"] = fake_module

    ep = _make_entry_point(
        "typoed",
        "rfdf_fake_noncallable:NOT_A_FUNCTION",
        "rfdf.backends.sdr",
    )

    def fake_entry_points(*, group: str) -> tuple[EntryPoint, ...]:
        if group == "rfdf.backends.sdr":
            return (ep,)
        return ()

    monkeypatch.setattr("rfdf.hal.discovery.entry_points", fake_entry_points)
    try:
        catalog = discover_backends("rfdf.backends.sdr")
    finally:
        sys.modules.pop("rfdf_fake_noncallable", None)
    assert catalog == {}


# ---------------------------------------------------------------------------
# Failure mode 3: duplicate names — first-wins, both distributions logged
# ---------------------------------------------------------------------------


def test_discover_duplicate_names_first_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two entry points sharing a name: keep the first one, log a WARN."""
    fake_module = ModuleType("rfdf_fake_duplicates")

    def factory_a(**_: Any) -> str:
        return "first"

    def factory_b(**_: Any) -> str:
        return "second"

    fake_module.factory_a = factory_a  # type: ignore[attr-defined]
    fake_module.factory_b = factory_b  # type: ignore[attr-defined]
    sys.modules["rfdf_fake_duplicates"] = fake_module

    ep_a = _make_entry_point("dup", "rfdf_fake_duplicates:factory_a", "rfdf.backends.sdr")
    ep_b = _make_entry_point("dup", "rfdf_fake_duplicates:factory_b", "rfdf.backends.sdr")

    def fake_entry_points(*, group: str) -> tuple[EntryPoint, ...]:
        if group == "rfdf.backends.sdr":
            return (ep_a, ep_b)
        return ()

    monkeypatch.setattr("rfdf.hal.discovery.entry_points", fake_entry_points)
    try:
        catalog = discover_backends("rfdf.backends.sdr")
    finally:
        sys.modules.pop("rfdf_fake_duplicates", None)
    assert set(catalog) == {"dup"}
    # First wins: factory_a should be selected.
    assert catalog["dup"]() == "first"


# ---------------------------------------------------------------------------
# Failure mode 4: factory raises at call time
# ---------------------------------------------------------------------------


def test_load_backend_wraps_factory_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the factory itself raises, load_backend raises BackendLoadError + __cause__."""

    def broken_factory(**_: Any) -> Any:
        raise RuntimeError("simulated provider auth failure")

    fake_catalog = {"explode": broken_factory}

    def fake_discover(group: str) -> dict[str, Any]:
        return fake_catalog if group == "rfdf.backends.sdr" else {}

    monkeypatch.setattr("rfdf.hal.discovery.discover_backends", fake_discover)
    with pytest.raises(BackendLoadError) as exc_info:
        load_backend("rfdf.backends.sdr", "explode")
    assert exc_info.value.group == "rfdf.backends.sdr"
    assert exc_info.value.name == "explode"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "simulated provider auth failure" in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# Smoke: discover_backends returns callable factories on real entry points
# ---------------------------------------------------------------------------


def test_discover_returns_callable_factories_when_real_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: a well-formed entry point loads and is registered."""

    def synthetic_factory(**kwargs: Any) -> dict[str, Any]:
        return {"backend": "synthetic", **kwargs}

    fake_module = ModuleType("rfdf_fake_good")
    fake_module.create = synthetic_factory  # type: ignore[attr-defined]
    sys.modules["rfdf_fake_good"] = fake_module

    ep = _make_entry_point("synthetic", "rfdf_fake_good:create", "rfdf.backends.sdr")

    def fake_entry_points(*, group: str) -> tuple[EntryPoint, ...]:
        if group == "rfdf.backends.sdr":
            return (ep,)
        return ()

    monkeypatch.setattr("rfdf.hal.discovery.entry_points", fake_entry_points)
    try:
        catalog = discover_backends("rfdf.backends.sdr")
    finally:
        sys.modules.pop("rfdf_fake_good", None)
    assert "synthetic" in catalog
    instance = catalog["synthetic"](foo="bar")
    assert instance == {"backend": "synthetic", "foo": "bar"}

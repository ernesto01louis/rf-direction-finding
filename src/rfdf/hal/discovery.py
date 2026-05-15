"""Backend discovery via ``importlib.metadata`` entry-points.

Backends register under four groups (``rfdf.backends.{sdr,rotator,geometry,compute}``).
Each entry point's name is the backend's user-facing identifier (``mock``, ``b210``,
``runpod``, …) and its value is a factory callable returning a configured backend
instance.

Discovery is fail-tolerant:

* If an entry point's ``.load()`` raises (typically ``ImportError`` because an
  optional dependency is absent), the discovery layer logs a warning and skips it.
  A broken third-party backend package never breaks the platform.
* If the loaded object is not callable (typo in ``pyproject.toml``), same treatment.
* If a factory raises at call time, ``load_backend`` wraps the exception in
  :class:`rfdf.hal.types.BackendLoadError` so callers see the backend identity in
  the traceback chain.
* If two backends register the same name (typically editable + wheel installs of
  the same project shadowing each other), the first one wins and a warning lists
  both source distributions.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import Any

import structlog

from rfdf.hal.types import BackendLoadError

_LOG = structlog.get_logger(__name__)

#: Canonical entry-point groups for the four HAL Protocols.
BACKEND_GROUPS: tuple[str, ...] = (
    "rfdf.backends.sdr",
    "rfdf.backends.rotator",
    "rfdf.backends.geometry",
    "rfdf.backends.compute",
)


def _safe_load(ep: EntryPoint) -> Callable[..., Any] | None:
    """Load an entry point, logging + returning ``None`` on any failure."""
    try:
        target = ep.load()
    except Exception as exc:  # intentional broad catch for plugin robustness
        _LOG.warning(
            "backend_entry_point_load_failed",
            group=ep.group,
            name=ep.name,
            error_type=type(exc).__name__,
            error=str(exc).splitlines()[0] if str(exc) else "",
        )
        return None
    if not callable(target):
        _LOG.warning(
            "backend_entry_point_not_callable",
            group=ep.group,
            name=ep.name,
            target_type=type(target).__name__,
        )
        return None
    return target  # type: ignore[no-any-return]


def discover_backends(group: str) -> dict[str, Callable[..., Any]]:
    """Return ``{name: factory}`` for one entry-point group.

    Loads every entry point under ``group``, skipping any that fail to import or
    aren't callable. On duplicate names, first-wins and a warning is logged.

    Args:
        group: Entry-point group (e.g. ``"rfdf.backends.sdr"``).

    Returns:
        Mapping from backend name to the factory callable produced by the entry
        point. Empty when no backends are installed under ``group``.
    """
    catalog: dict[str, Callable[..., Any]] = {}
    seen_dists: dict[str, str] = {}
    for ep in entry_points(group=group):
        factory = _safe_load(ep)
        if factory is None:
            continue
        dist_name = ep.dist.name if ep.dist is not None else "<unknown>"
        if ep.name in catalog:
            _LOG.warning(
                "backend_duplicate_name_first_wins",
                group=group,
                name=ep.name,
                first_dist=seen_dists.get(ep.name, "<unknown>"),
                shadowed_dist=dist_name,
            )
            continue
        catalog[ep.name] = factory
        seen_dists[ep.name] = dist_name
    return catalog


def load_backend(group: str, name: str, **kwargs: Any) -> Any:
    """Discover ``group`` and call the ``name`` factory with ``**kwargs``.

    Args:
        group: Entry-point group (e.g. ``"rfdf.backends.sdr"``).
        name: Backend identifier (e.g. ``"mock"``).
        **kwargs: Forwarded to the backend factory.

    Returns:
        The backend instance returned by the factory.

    Raises:
        KeyError: If ``name`` is not registered under ``group``.
        BackendLoadError: If the factory raised an exception. The original
            exception is preserved as ``__cause__``.
    """
    catalog = discover_backends(group)
    if name not in catalog:
        available = sorted(catalog)
        raise KeyError(
            f"no backend named {name!r} in group {group!r} "
            f"(available: {available or 'none installed'})"
        )
    try:
        return catalog[name](**kwargs)
    except Exception as exc:
        raise BackendLoadError(group, name, exc) from exc


def list_backends() -> dict[str, list[str]]:
    """Catalog every backend across the four canonical groups.

    Returns:
        Mapping ``{group: [name, ...]}``. Groups with no installed backends map
        to an empty list. Useful for the ``rfdf hw list-backends`` CLI.
    """
    return {group: sorted(discover_backends(group)) for group in BACKEND_GROUPS}

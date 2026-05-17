"""Shared fixtures for the property-based contract suites.

Each suite parametrizes over its HAL group's registered backends. Backend
factories sometimes need configured-on-construction kwargs (e.g. the SigMF
file-replay backend needs a ``meta_path``). The factory wrappers below build a
ready-to-test instance for each backend name.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfdf.hal import discover_backends


def _filter_hardware(names: list[str]) -> list[str]:
    """Drop backends that require physical hardware from the contract suites.

    The contract suites build each backend with a no-arg (or minimal) factory
    call; the Stage 5 hardware backends need real devices (and site-specific
    construction kwargs), so they are excluded here and verified instead by the
    ``hardware``-marked suites under ``tests/hardware/``.
    """
    hardware_only = {
        "b210",
        "hackrf",
        "rtl-sdr",
        "rtlsdr",
        "krakensdr",
        "yaesu-rotctld",
        "antrunner",
        "grbl-linear",
    }
    return [n for n in names if n not in hardware_only]


def _sdr_factory_kwargs(name: str, tiny_sigmf: tuple[Path, Path]) -> dict[str, Any]:
    if name == "file-replay":
        meta_path, _ = tiny_sigmf
        return {"meta_path": meta_path, "block_samples": 64}
    return {"block_samples": 64}


@pytest.fixture(scope="session")
def sdr_factories() -> list[tuple[str, Callable[..., Any]]]:
    catalog = discover_backends("rfdf.backends.sdr")
    return [(name, catalog[name]) for name in sorted(_filter_hardware(list(catalog)))]


@pytest.fixture(scope="session")
def rotator_factories() -> list[tuple[str, Callable[..., Any]]]:
    catalog = discover_backends("rfdf.backends.rotator")
    return [(name, catalog[name]) for name in sorted(_filter_hardware(list(catalog)))]


@pytest.fixture(scope="session")
def geometry_factories() -> list[tuple[str, Callable[..., Any]]]:
    catalog = discover_backends("rfdf.backends.geometry")
    return [(name, catalog[name]) for name in sorted(_filter_hardware(list(catalog)))]


@pytest.fixture(scope="session")
def compute_factories() -> list[tuple[str, Callable[..., Any]]]:
    catalog = discover_backends("rfdf.backends.compute")
    return [(name, catalog[name]) for name in sorted(_filter_hardware(list(catalog)))]


@pytest.fixture
def sdr_factory_kwargs(tiny_sigmf: tuple[Path, Path]) -> Callable[[str], dict[str, Any]]:
    """Return a function that produces per-backend constructor kwargs."""

    def make(name: str) -> dict[str, Any]:
        return _sdr_factory_kwargs(name, tiny_sigmf)

    return make

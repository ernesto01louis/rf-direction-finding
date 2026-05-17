"""Property-based contract for :class:`rfdf.hal.SdrSource`.

Any backend conforming to the protocol MUST satisfy:

* ``num_channels`` is a positive int.
* ``stream()`` yields ``StreamBlock`` instances whose ``iq`` array has shape
  ``(num_channels, *)`` and dtype ``complex64``.
* Sequence numbers across blocks are monotonically increasing.
* ``tuning_range_hz`` is a valid interval and ``max_sample_rate_hz > 0``.

Hypothesis is used with ``derandomize=True`` to keep CI reproducible and a
constrained centre-frequency strategy to keep the array-factor mock from
spatial-aliasing edge cases (corner cases are Stage 3's problem).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rfdf.hal import SdrConfig, SdrSource


def _make_sdr(
    name: str,
    factory: Callable[..., Any],
    sdr_factory_kwargs: Callable[[str], dict[str, Any]],
) -> SdrSource:
    """Build an SDR via the factory + the per-backend kwarg builder."""
    return factory(**sdr_factory_kwargs(name))  # type: ignore[no-any-return]


def test_num_channels_positive(
    sdr_factories: list[tuple[str, Callable[..., Any]]],
    sdr_factory_kwargs: Callable[[str], dict[str, Any]],
) -> None:
    """Every conforming SDR reports num_channels > 0."""
    assert sdr_factories, "no SDR backends discovered — Stage 2 must register at least one"
    for name, factory in sdr_factories:
        sdr = _make_sdr(name, factory, sdr_factory_kwargs)
        assert sdr.num_channels > 0, f"{name}: num_channels {sdr.num_channels}"


def test_tuning_range_well_formed(
    sdr_factories: list[tuple[str, Callable[..., Any]]],
    sdr_factory_kwargs: Callable[[str], dict[str, Any]],
) -> None:
    """tuning_range_hz is (low, high) with low <= high and max_sample_rate > 0."""
    for name, factory in sdr_factories:
        sdr = _make_sdr(name, factory, sdr_factory_kwargs)
        low, high = sdr.tuning_range_hz
        assert low <= high, f"{name}: range {(low, high)}"
        assert sdr.max_sample_rate_hz > 0, name


@settings(
    derandomize=True,
    deadline=None,
    max_examples=8,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(rx_gain_db=st.floats(min_value=0.0, max_value=60.0))
def test_stream_blocks_are_well_formed(
    rx_gain_db: float,
    sdr_factories: list[tuple[str, Callable[..., Any]]],
    sdr_factory_kwargs: Callable[[str], dict[str, Any]],
    tiny_sigmf: tuple[Path, Path],
) -> None:
    """StreamBlock.iq has shape (num_channels, N), complex64, finite."""
    _ = tiny_sigmf  # session fixture used transitively via sdr_factory_kwargs
    for name, factory in sdr_factories:
        sdr = _make_sdr(name, factory, sdr_factory_kwargs)
        # Use a centre frequency / sample rate the recording can deliver. The
        # file-replay backend is pinned to its recording; the mock accepts a
        # wide range.
        if name == "file-replay":
            sample_rate = 2_560.0
            center_freq = 868e6
        else:
            sample_rate = 2e6
            center_freq = 868e6
        cfg = SdrConfig(
            center_freq_hz=center_freq,
            sample_rate_hz=sample_rate,
            rx_gain_db=rx_gain_db,
        )

        async def run(s: SdrSource = sdr, c: SdrConfig = cfg, n: str = name) -> None:
            await s.configure(c)
            await s.start()
            collected: list[int] = []
            async for block in s.stream():
                assert block.iq.shape[0] == s.num_channels, n
                assert block.iq.dtype == np.complex64, n
                assert np.all(np.isfinite(block.iq)), n
                collected.append(block.sequence_number)
                if len(collected) >= 3:
                    await s.stop()
                    break
            # Monotonic sequence numbers.
            assert collected == sorted(collected) and len(set(collected)) == len(collected), (
                n,
                collected,
            )

        asyncio.run(run())


def test_close_is_idempotent(
    sdr_factories: list[tuple[str, Callable[..., Any]]],
    sdr_factory_kwargs: Callable[[str], dict[str, Any]],
) -> None:
    """Repeated close() calls don't raise."""
    for name, factory in sdr_factories:
        sdr = _make_sdr(name, factory, sdr_factory_kwargs)

        async def run(s: SdrSource = sdr) -> None:
            await s.close()
            await s.close()

        asyncio.run(run())


def test_isinstance_protocol(
    sdr_factories: list[tuple[str, Callable[..., Any]]],
    sdr_factory_kwargs: Callable[[str], dict[str, Any]],
) -> None:
    """Every backend is a structural SdrSource."""
    for name, factory in sdr_factories:
        sdr = _make_sdr(name, factory, sdr_factory_kwargs)
        assert isinstance(sdr, SdrSource), name


def test_status_returns_dict(
    sdr_factories: list[tuple[str, Callable[..., Any]]],
    sdr_factory_kwargs: Callable[[str], dict[str, Any]],
) -> None:
    """status() returns a dict with a 'backend' key, callable before configure()."""
    for name, factory in sdr_factories:
        sdr = _make_sdr(name, factory, sdr_factory_kwargs)

        async def run(s: SdrSource = sdr, n: str = name) -> None:
            # Probed before configure() — status() is a health probe, not a
            # data path, and must not raise.
            report = await s.status()
            assert isinstance(report, dict), n
            assert isinstance(report.get("backend"), str), n

        asyncio.run(run())


def _unused_pytest_helper(_p: pytest.MonkeyPatch) -> None:  # pragma: no cover
    pass

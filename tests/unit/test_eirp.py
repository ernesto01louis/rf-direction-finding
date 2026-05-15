"""Unit tests for ``rfdf.core.eirp``.

Covers the cap helper (under + at + over), the override gate, and the decorator.
"""

from __future__ import annotations

import asyncio

import pytest

from rfdf.config import RfdfConfig
from rfdf.core.eirp import (
    DEFAULT_EIRP_CAP_DBM,
    EirpCapExceededError,
    enforce_eirp_cap,
    requires_eirp_check,
)


def _config(max_eirp_dbm: float, *, override: bool = False) -> RfdfConfig:
    cfg = RfdfConfig()
    cfg.eirp.max_eirp_dbm = max_eirp_dbm
    cfg.eirp.override_explicit = override
    return cfg


def test_default_cap_is_eu_srd_25_milliwatts() -> None:
    """14 dBm = 25 mW, the EU SRD general license-exempt limit."""
    assert pytest.approx(14.0) == DEFAULT_EIRP_CAP_DBM


def test_enforce_below_cap_passes() -> None:
    """Power under the cap is silently allowed."""
    enforce_eirp_cap(10.0, config=_config(14.0))


def test_enforce_at_cap_passes() -> None:
    """Power equal to the cap is allowed (cap is inclusive on the lower side)."""
    enforce_eirp_cap(14.0, config=_config(14.0))


def test_enforce_above_cap_raises() -> None:
    """Power above the cap raises EirpCapExceededError."""
    with pytest.raises(EirpCapExceededError) as exc_info:
        enforce_eirp_cap(20.0, config=_config(14.0))
    assert exc_info.value.power_dbm == pytest.approx(20.0)
    assert exc_info.value.cap_dbm == pytest.approx(14.0)


def test_override_explicit_lets_power_exceed_cap() -> None:
    """override_explicit=True permits >cap power (operator took the decision)."""
    # Higher than cap, but override is set — allowed.
    enforce_eirp_cap(36.0, config=_config(14.0, override=True))


def test_error_message_points_to_override_flag() -> None:
    """The exception message should tell operators how to authorize the request."""
    with pytest.raises(EirpCapExceededError, match="override_explicit"):
        enforce_eirp_cap(50.0, config=_config(14.0))


def test_decorator_blocks_above_cap() -> None:
    """The decorator raises before the wrapped function body runs."""
    calls: list[float] = []

    @requires_eirp_check
    async def fake_pilot(*, freq_hz: float, power_dbm: float, config: RfdfConfig) -> None:
        calls.append(power_dbm)

    with pytest.raises(EirpCapExceededError):
        asyncio.run(fake_pilot(freq_hz=868e6, power_dbm=20.0, config=_config(14.0)))
    assert calls == []


def test_decorator_passes_through_below_cap() -> None:
    """When the power is allowed, the wrapped function receives the call."""
    calls: list[float] = []

    @requires_eirp_check
    async def fake_pilot(*, freq_hz: float, power_dbm: float, config: RfdfConfig) -> str:
        calls.append(power_dbm)
        return "emitted"

    result = asyncio.run(fake_pilot(freq_hz=868e6, power_dbm=10.0, config=_config(14.0)))
    assert result == "emitted"
    assert calls == [10.0]


def test_decorator_skips_when_power_dbm_absent() -> None:
    """Without a power_dbm kwarg, the decorator forwards unchanged."""
    received: dict[str, float] = {}

    @requires_eirp_check
    async def no_power_method(*, freq_hz: float) -> None:
        received["freq_hz"] = freq_hz

    asyncio.run(no_power_method(freq_hz=2.4e9))
    assert received == {"freq_hz": 2.4e9}

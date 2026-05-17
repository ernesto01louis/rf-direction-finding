"""Unit tests for the RTL-SDR contrib backend (no dongle required)."""

from __future__ import annotations

import asyncio

import pytest
from rfdf_backend_rtlsdr.source import (
    RtlSdrError,
    RtlSdrSource,
    create,
)

from rfdf.hal import SdrConfig, SdrSource


def test_capabilities() -> None:
    """A constructed RTL-SDR backend reports single-channel capabilities."""
    sdr = create()
    assert sdr.num_channels == 1
    assert sdr.supports_coherent is False
    assert sdr.tuning_range_hz == (24e6, 1.766e9)
    assert sdr.max_sample_rate_hz == pytest.approx(2.56e6)


def test_rtlsdr_is_structural_sdr_source() -> None:
    """RtlSdrSource structurally satisfies the SdrSource Protocol."""
    assert isinstance(create(), SdrSource)


def test_configure_rejects_out_of_range_frequency() -> None:
    """A centre frequency outside the R820T2 range is rejected."""
    sdr = RtlSdrSource()

    async def run() -> None:
        with pytest.raises(RtlSdrError, match="tuning range"):
            await sdr.configure(SdrConfig(center_freq_hz=3e9, sample_rate_hz=2e6))

    asyncio.run(run())


def test_configure_rejects_excessive_sample_rate() -> None:
    """A sample rate above the stable limit is rejected."""
    sdr = RtlSdrSource()

    async def run() -> None:
        with pytest.raises(RtlSdrError, match="stable limit"):
            await sdr.configure(SdrConfig(center_freq_hz=100e6, sample_rate_hz=5e6))

    asyncio.run(run())


def test_status_before_open_is_safe() -> None:
    """status() is a health probe — safe before the dongle is opened."""
    report = asyncio.run(create().status())
    assert report["backend"] == "rtlsdr"
    assert report["reachable"] is False


def test_calibration_pilot_is_rx_only() -> None:
    """RTL-SDR cannot transmit — calibration_pilot raises."""

    async def run() -> None:
        with pytest.raises(NotImplementedError, match="RX only"):
            await create().calibration_pilot(100e6, 0.0)

    asyncio.run(run())

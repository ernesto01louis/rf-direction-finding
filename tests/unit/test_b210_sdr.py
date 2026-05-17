"""Unit tests for the B210 SDR backend — pure helpers + no-hardware paths.

The UHD device lifecycle (open / lock / retune / stream) needs physical
hardware and is covered by ``tests/hardware/test_b210_hardware.py`` under the
``hardware`` marker. These tests exercise everything that does NOT need a B210:
the data-rate envelope, USB-topology parsing, the pilot-tone estimator, the
"uhd not installed" path, factory validation, and Protocol conformance.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from rfdf.backends.sdr.b210 import (
    B210CalibrationError,
    B210Error,
    B210NotInstalledError,
    B210RateError,
    B210Source,
    _check_data_rate_envelope,
    _estimate_pilot_corrections,
    _parse_usb_topology,
    _require_uhd,
    create,
)
from rfdf.hal import SdrSource

_LSUSB_TWO_SS_BUSES = """\
/:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/6p, 5000M
    |__ Port 1: Dev 2, If 0, Class=Vendor Specific Class, Driver=, 5000M
/:  Bus 04.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/6p, 5000M
    |__ Port 2: Dev 3, If 0, Class=Vendor Specific Class, Driver=, 5000M
/:  Bus 01.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/12p, 480M
"""

_LSUSB_ONE_SS_BUS = """\
/:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/6p, 5000M
    |__ Port 1: Dev 2, If 0, Class=Vendor Specific Class, Driver=, 5000M
    |__ Port 2: Dev 3, If 0, Class=Vendor Specific Class, Driver=, 5000M
"""


# ---------------------------------------------------------------------------
# USB topology parsing
# ---------------------------------------------------------------------------


def test_usb_topology_ok_when_each_b210_on_own_controller() -> None:
    """Two 5000M root buses for two B210s passes the topology check."""
    result = _parse_usb_topology(_LSUSB_TWO_SS_BUSES, b210_count=2)
    assert result["ok"] is True
    assert result["root_controllers"] == 2


def test_usb_topology_flags_shared_controller() -> None:
    """Two B210s on one controller is flagged — the #1 instability cause."""
    result = _parse_usb_topology(_LSUSB_ONE_SS_BUS, b210_count=2)
    assert result["ok"] is False
    assert "separate controllers" in result["message"]


# ---------------------------------------------------------------------------
# Data-rate envelope
# ---------------------------------------------------------------------------


def test_data_rate_envelope_accepts_modest_request() -> None:
    """2 channels at 10 MS/s is within both the per-channel and aggregate caps."""
    _check_data_rate_envelope(num_channels=2, sample_rate_hz=10e6)


def test_data_rate_envelope_rejects_excessive_per_channel_rate() -> None:
    """A per-channel rate above 25 MS/s is rejected."""
    with pytest.raises(B210RateError, match="sustained limit"):
        _check_data_rate_envelope(num_channels=1, sample_rate_hz=30e6)


def test_data_rate_envelope_rejects_excessive_aggregate() -> None:
    """8 channels at 10 MS/s = 320 MB/s exceeds the 240 MB/s USB envelope."""
    with pytest.raises(B210RateError, match="USB envelope"):
        _check_data_rate_envelope(num_channels=8, sample_rate_hz=10e6)


# ---------------------------------------------------------------------------
# Pilot-tone estimator
# ---------------------------------------------------------------------------


def test_pilot_corrections_invert_channel_errors() -> None:
    """Per-channel gain/phase errors are recovered as their reciprocals."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    errors = np.array([1.0, 0.5j, -2.0, 0.8 - 0.6j], dtype=np.complex128)
    iq = np.outer(errors, base)
    corrections = _estimate_pilot_corrections(iq)
    # Applying the correction equalises every channel to channel 0.
    corrected = iq * corrections[:, None]
    for ch in range(1, 4):
        assert np.allclose(corrected[ch], corrected[0], atol=1e-9)


def test_pilot_corrections_reject_empty_iq() -> None:
    """Empty IQ raises a calibration error rather than dividing by zero."""
    with pytest.raises(B210CalibrationError, match="no IQ"):
        _estimate_pilot_corrections(np.zeros((4, 0), dtype=np.complex128))


def test_pilot_corrections_reject_dead_channel() -> None:
    """A channel with no pilot energy raises rather than producing inf gains."""
    iq = np.ones((3, 64), dtype=np.complex128)
    iq[1] = 0.0
    with pytest.raises(B210CalibrationError, match="no pilot energy"):
        _estimate_pilot_corrections(iq)


# ---------------------------------------------------------------------------
# SDK absence + factory validation
# ---------------------------------------------------------------------------


def test_require_uhd_raises_clean_error_when_absent() -> None:
    """Without the [sdr-uhd] extra, _require_uhd raises an actionable error."""
    with pytest.raises(B210NotInstalledError, match=r"rfdf\[sdr-uhd\]"):
        _require_uhd()


def test_create_requires_serial_numbers() -> None:
    """The factory rejects a call with no serial numbers."""
    with pytest.raises(B210Error, match="serial_numbers is required"):
        create()


def test_constructor_rejects_empty_serials() -> None:
    """B210Source rejects an empty serial list directly."""
    with pytest.raises(B210Error, match="at least one serial"):
        B210Source([])


# ---------------------------------------------------------------------------
# Construction + capabilities (no device session)
# ---------------------------------------------------------------------------


def test_construction_and_capabilities() -> None:
    """A constructed B210Source reports sane capabilities before configure()."""
    sdr = create(serial_numbers=["31D5A3B", "31D5A40"])
    assert sdr.num_channels == 4  # 2 RX channels per unit
    assert sdr.supports_coherent is True
    assert sdr.tuning_range_hz == (70e6, 6e9)
    assert sdr.max_sample_rate_hz == 25e6
    assert "Pilot-tone calibration" in sdr.coherent_caveats
    assert sdr.calibration is None


def test_b210_is_structural_sdr_source() -> None:
    """B210Source structurally satisfies the SdrSource Protocol."""
    sdr = create(serial_numbers=["31D5A3B"])
    assert isinstance(sdr, SdrSource)


def test_status_before_session_is_safe() -> None:
    """status() is a health probe — safe before any device session exists."""
    sdr = create(serial_numbers=["31D5A3B"])
    report = asyncio.run(sdr.status())
    assert report["backend"] == "b210"
    assert report["reachable"] is False
    assert report["calibrated"] is False


def test_calibration_pilot_is_rx_only() -> None:
    """The B210 backend cannot transmit — calibration_pilot raises."""
    sdr = create(serial_numbers=["31D5A3B"])

    async def run() -> None:
        with pytest.raises(NotImplementedError, match="RX-only"):
            await sdr.calibration_pilot(868e6, 0.0)

    asyncio.run(run())

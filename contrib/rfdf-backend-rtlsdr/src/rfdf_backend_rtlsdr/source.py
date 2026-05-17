"""RTL-SDR ``SdrSource`` backend — a contrib reference example.

A single-channel RTL2832U + R820T2 dongle (24 MHz - 1.766 GHz). Cheap and
ubiquitous; ideal for demos and as the canonical "how to write a contrib
backend" template.

This package is **not** a dependency of core ``rfdf``. It is installed
separately (``pip install -e contrib/rfdf-backend-rtlsdr/``) and registers
itself via the ``rfdf.backends.sdr`` entry-point group, so ``rfdf hw
list-backends`` discovers it once installed.

``pyrtlsdr`` is imported lazily so the module — and entry-point discovery —
work even when the dongle's driver is not installed.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from rfdf.hal.sdr import Recording, SdrConfig, StreamBlock

#: RTL-SDR R820T2 tuner range.
_TUNING_RANGE_HZ = (24e6, 1.766e9)
#: Practical sustained sample rate (the dongle quotes 3.2 MS/s, 2.56 is stable).
_MAX_SAMPLE_RATE_HZ = 2.56e6


class RtlSdrError(RuntimeError):
    """Base class for every error raised by the RTL-SDR backend."""


class RtlSdrNotInstalledError(RtlSdrError):
    """The ``pyrtlsdr`` package / RTL-SDR driver is not importable."""


def _require_rtlsdr() -> Any:
    """Import and return the ``rtlsdr`` module, or raise a clear install hint."""
    try:
        import rtlsdr
    except ImportError as exc:  # pragma: no cover - exercised only without the driver
        raise RtlSdrNotInstalledError(
            "The RTL-SDR backend requires pyrtlsdr and the librtlsdr driver; "
            "install with: pip install rfdf-backend-rtlsdr (and your distro's "
            "librtlsdr package)."
        ) from exc
    return rtlsdr


class RtlSdrSource:
    """Single-channel RTL-SDR backend implementing the ``SdrSource`` contract.

    Args:
        device_index: RTL-SDR device index (0 for the first dongle).
        block_samples: Samples per :class:`StreamBlock`.
    """

    supports_coherent = False
    tuning_range_hz = _TUNING_RANGE_HZ
    max_sample_rate_hz = _MAX_SAMPLE_RATE_HZ

    def __init__(self, *, device_index: int = 0, block_samples: int = 4096) -> None:
        """Capture configuration; the dongle is opened in ``configure()``."""
        self._device_index = int(device_index)
        self._block_samples = int(block_samples)
        self._sdr: Any | None = None
        self._config: SdrConfig | None = None
        self._sequence = 0
        self._running = False

    @property
    def num_channels(self) -> int:
        """RTL-SDR is single-channel."""
        return 1

    async def configure(self, config: SdrConfig) -> None:
        """Open the dongle and apply tuning + sampling configuration."""
        if config.sample_rate_hz > _MAX_SAMPLE_RATE_HZ:
            raise RtlSdrError(
                f"RTL-SDR: {config.sample_rate_hz / 1e6:.2f} MS/s exceeds the "
                f"stable limit of {_MAX_SAMPLE_RATE_HZ / 1e6:.2f} MS/s."
            )
        low, high = _TUNING_RANGE_HZ
        if not low <= config.center_freq_hz <= high:
            raise RtlSdrError(
                f"RTL-SDR: {config.center_freq_hz / 1e6:.1f} MHz outside the "
                f"tuning range {low / 1e6:.0f}-{high / 1e6:.0f} MHz."
            )
        rtlsdr = _require_rtlsdr()
        if self._sdr is None:
            self._sdr = rtlsdr.RtlSdr(device_index=self._device_index)
        self._sdr.sample_rate = config.sample_rate_hz
        self._sdr.center_freq = config.center_freq_hz
        self._sdr.gain = config.rx_gain_db
        self._config = config

    async def start(self) -> None:
        """Mark the backend as streaming; idempotent."""
        if self._sdr is None or self._config is None:
            raise RtlSdrError("RTL-SDR: call configure() before start()")
        self._running = True

    async def stop(self) -> None:
        """Stop streaming."""
        self._running = False

    async def stream(self) -> AsyncIterator[StreamBlock]:
        """Yield :class:`StreamBlock`s read from the dongle until ``stop()``."""
        if not self._running or self._sdr is None or self._config is None:
            raise RtlSdrError("RTL-SDR: call configure() + start() before stream()")
        try:
            while self._running:
                samples = await asyncio.to_thread(self._sdr.read_samples, self._block_samples)
                iq = np.asarray(samples, dtype=np.complex64).reshape(1, -1)
                yield StreamBlock(
                    iq=iq,
                    sample_rate_hz=self._config.sample_rate_hz,
                    center_freq_hz=self._config.center_freq_hz,
                    start_time_s=self._sequence * self._block_samples / self._config.sample_rate_hz,
                    sequence_number=self._sequence,
                    metadata={"backend": "rtlsdr"},
                )
                self._sequence += 1
        finally:
            self._running = False

    async def capture(self, duration_s: float) -> Recording:
        """Capture ``duration_s`` of IQ to a single-channel SigMF pair."""
        if self._sdr is None or self._config is None:
            raise RtlSdrError("RTL-SDR: call configure() before capture()")
        num_samples = round(duration_s * self._config.sample_rate_hz)
        samples = await asyncio.to_thread(self._sdr.read_samples, num_samples)
        iq = np.asarray(samples, dtype=np.complex64)
        capture_dir = Path.cwd() / ".rfdf-captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rtlsdr-{int(time.time() * 1000)}"
        data_path = capture_dir / f"{stem}.sigmf-data"
        meta_path = capture_dir / f"{stem}.sigmf-meta"
        iq.tofile(data_path)
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self._config.sample_rate_hz,
                "core:version": "1.0.0",
                "core:num_channels": 1,
                "core:hw": "RTL-SDR (RTL2832U + R820T2)",
            },
            "captures": [{"core:sample_start": 0, "core:frequency": self._config.center_freq_hz}],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return Recording(
            sigmf_meta_path=meta_path,
            sigmf_data_path=data_path,
            duration_s=duration_s,
            num_samples=len(iq),
            channels=1,
            sample_rate_hz=self._config.sample_rate_hz,
            center_freq_hz=self._config.center_freq_hz,
            metadata=meta,
        )

    async def status(self) -> dict[str, object]:
        """Report dongle health for ``rfdf hw selftest``."""
        return {
            "backend": "rtlsdr",
            "reachable": self._sdr is not None,
            "device_index": self._device_index,
            "num_channels": 1,
            "configured": self._config is not None,
            "streaming": self._running,
        }

    async def calibration_pilot(self, freq_hz: float, power_dbm: float) -> None:
        """RTL-SDR is RX-only — it cannot emit a pilot tone."""
        raise NotImplementedError("RTL-SDR: cannot emit a pilot tone — RX only")

    async def close(self) -> None:
        """Close the dongle and release the USB handle."""
        self._running = False
        if self._sdr is not None:
            self._sdr.close()
            self._sdr = None


def create(*, device_index: int = 0, block_samples: int = 4096, **_: Any) -> RtlSdrSource:
    """Factory wired into the ``rfdf.backends.sdr`` ``rtlsdr`` entry-point.

    Args:
        device_index: RTL-SDR device index.
        block_samples: Samples per streamed block.

    Returns:
        A configured (un-opened) :class:`RtlSdrSource`.
    """
    return RtlSdrSource(device_index=device_index, block_samples=block_samples)


__all__ = ["RtlSdrError", "RtlSdrNotInstalledError", "RtlSdrSource", "create"]

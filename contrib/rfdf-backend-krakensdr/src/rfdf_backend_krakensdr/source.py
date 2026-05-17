"""KrakenSDR ``SdrSource`` backend — a 5-channel coherent contrib example.

The KrakenSDR is five phase-coherent RTL-SDR receivers. Coherent capture is
handled by the **Heimdall DAQ daemon**; this backend consumes Heimdall's
aligned IQ frames through the :class:`HeimdallInterface` seam (see
``heimdall.py``) and presents them as the Stage-2 ``SdrSource`` HAL contract.

This is a contrib backend — a separate, pip-installable package, NOT a
dependency of core rfdf. Together with the RTL-SDR backend it is a canonical
example for community contributors: RTL-SDR shows the single-channel case,
KrakenSDR shows the coherent multi-channel case.
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
from rfdf_backend_krakensdr.heimdall import (
    HeimdallError,
    HeimdallInterface,
    HeimdallShmInterface,
    _coerce_interface,
)

#: KrakenSDR coherent tuning ceiling — the R820T2 tuners run coherently to
#: ~1.8 GHz; above that the per-tuner PLLs lose lock alignment.
_TUNING_RANGE_HZ = (24e6, 1.8e9)
#: Heimdall publishes coherent frames at up to ~2.56 MS/s per channel.
_MAX_SAMPLE_RATE_HZ = 2.56e6


class KrakenSdrError(RuntimeError):
    """Base class for every error raised by the KrakenSDR backend."""


class KrakenSdrSource:
    """5-channel coherent KrakenSDR backend implementing ``SdrSource``.

    Args:
        heimdall: The Heimdall DAQ interface frames are read through.
        block_samples: Samples per :class:`StreamBlock`.
    """

    supports_coherent = True
    coherent_caveats = (
        "Coherent alignment is maintained by the Heimdall DAQ noise-source "
        "calibration; re-run Heimdall calibration after a retune."
    )
    tuning_range_hz = _TUNING_RANGE_HZ
    max_sample_rate_hz = _MAX_SAMPLE_RATE_HZ

    def __init__(self, *, heimdall: HeimdallInterface, block_samples: int = 4096) -> None:
        """Capture the Heimdall interface; the daemon link opens in configure()."""
        self._heimdall = _coerce_interface(heimdall)
        self._block_samples = int(block_samples)
        self._config: SdrConfig | None = None
        self._sequence = 0
        self._running = False
        self._connected = False

    @property
    def num_channels(self) -> int:
        """KrakenSDR is a 5-channel coherent receiver."""
        return self._heimdall.num_channels

    async def configure(self, config: SdrConfig) -> None:
        """Connect to Heimdall and apply the tuning configuration."""
        if config.sample_rate_hz > _MAX_SAMPLE_RATE_HZ:
            raise KrakenSdrError(
                f"KrakenSDR: {config.sample_rate_hz / 1e6:.2f} MS/s exceeds the "
                f"coherent limit of {_MAX_SAMPLE_RATE_HZ / 1e6:.2f} MS/s."
            )
        low, high = _TUNING_RANGE_HZ
        if not low <= config.center_freq_hz <= high:
            raise KrakenSdrError(
                f"KrakenSDR: {config.center_freq_hz / 1e6:.1f} MHz outside the "
                f"coherent tuning range {low / 1e6:.0f}-{high / 1e6:.0f} MHz."
            )
        await asyncio.to_thread(
            self._heimdall.connect,
            center_freq_hz=config.center_freq_hz,
            sample_rate_hz=config.sample_rate_hz,
            gain_db=config.rx_gain_db,
        )
        self._connected = True
        self._config = config

    async def start(self) -> None:
        """Mark the backend as streaming; idempotent."""
        if not self._connected or self._config is None:
            raise KrakenSdrError("KrakenSDR: call configure() before start()")
        self._running = True

    async def stop(self) -> None:
        """Stop streaming."""
        self._running = False

    async def stream(self) -> AsyncIterator[StreamBlock]:
        """Yield coherent :class:`StreamBlock`s from Heimdall until ``stop()``."""
        if not self._running or self._config is None:
            raise KrakenSdrError("KrakenSDR: call configure() + start() before stream()")
        try:
            while self._running:
                frame = await asyncio.to_thread(self._heimdall.read_frame, self._block_samples)
                iq = np.asarray(frame, dtype=np.complex64)
                yield StreamBlock(
                    iq=iq,
                    sample_rate_hz=self._config.sample_rate_hz,
                    center_freq_hz=self._config.center_freq_hz,
                    start_time_s=self._sequence * self._block_samples / self._config.sample_rate_hz,
                    sequence_number=self._sequence,
                    metadata={"backend": "krakensdr", "coherent": True},
                )
                self._sequence += 1
        finally:
            self._running = False

    async def capture(self, duration_s: float) -> Recording:
        """Capture ``duration_s`` of coherent IQ to a multi-channel SigMF pair."""
        if self._config is None:
            raise KrakenSdrError("KrakenSDR: call configure() before capture()")
        num_samples = round(duration_s * self._config.sample_rate_hz)
        frame = await asyncio.to_thread(self._heimdall.read_frame, num_samples)
        iq = np.asarray(frame, dtype=np.complex64)
        capture_dir = Path.cwd() / ".rfdf-captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        stem = f"krakensdr-{int(time.time() * 1000)}"
        data_path = capture_dir / f"{stem}.sigmf-data"
        meta_path = capture_dir / f"{stem}.sigmf-meta"
        iq.tofile(data_path)
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self._config.sample_rate_hz,
                "core:version": "1.0.0",
                "core:num_channels": self.num_channels,
                "core:hw": "KrakenSDR (5x coherent RTL-SDR via Heimdall DAQ)",
                "rfdf:coherent": True,
            },
            "captures": [{"core:sample_start": 0, "core:frequency": self._config.center_freq_hz}],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return Recording(
            sigmf_meta_path=meta_path,
            sigmf_data_path=data_path,
            duration_s=duration_s,
            num_samples=iq.shape[1] if iq.ndim == 2 else 0,
            channels=self.num_channels,
            sample_rate_hz=self._config.sample_rate_hz,
            center_freq_hz=self._config.center_freq_hz,
            metadata=meta,
        )

    async def status(self) -> dict[str, object]:
        """Report KrakenSDR / Heimdall health for ``rfdf hw selftest``."""
        return {
            "backend": "krakensdr",
            "reachable": self._connected,
            "num_channels": self.num_channels,
            "coherent": True,
            "configured": self._config is not None,
            "streaming": self._running,
        }

    async def calibration_pilot(self, freq_hz: float, power_dbm: float) -> None:
        """KrakenSDR is RX-only — coherent calibration is Heimdall's noise source."""
        raise NotImplementedError(
            "KrakenSDR: cannot emit a pilot tone — RX only. Coherent calibration "
            "is performed by the Heimdall DAQ noise source."
        )

    async def close(self) -> None:
        """Stop streaming and release the Heimdall link."""
        self._running = False
        self._connected = False
        await asyncio.to_thread(self._heimdall.close)


def create(
    *,
    heimdall: HeimdallInterface | None = None,
    ctrl_path: str = "/tmp/krakensdr/_data_control",
    shm_name: str = "kraken_iq",
    block_samples: int = 4096,
    **_: Any,
) -> KrakenSdrSource:
    """Factory wired into the ``rfdf.backends.sdr`` ``krakensdr`` entry-point.

    Args:
        heimdall: A pre-built :class:`HeimdallInterface` (tests inject a fake);
            when omitted, a real :class:`HeimdallShmInterface` is constructed.
        ctrl_path: Heimdall control FIFO/socket path (real interface only).
        shm_name: Heimdall IQ shared-memory base name (real interface only).
        block_samples: Samples per streamed block.

    Returns:
        A configured (un-connected) :class:`KrakenSdrSource`.
    """
    interface = heimdall
    if interface is None:
        interface = HeimdallShmInterface(ctrl_path=ctrl_path, shm_name=shm_name)
    return KrakenSdrSource(heimdall=interface, block_samples=block_samples)


__all__ = ["HeimdallError", "KrakenSdrError", "KrakenSdrSource", "create"]

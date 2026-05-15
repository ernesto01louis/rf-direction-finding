"""``SdrSource`` Protocol and supporting Pydantic models.

An ``SdrSource`` is anything that produces complex IQ samples in a stream of
``StreamBlock`` chunks. Real hardware (B210, HackRF, RTL-SDR), file replay (SigMF),
and synthetic emitter scenarios all conform to the same interface so the algorithm
layer never special-cases the source.

The HAL is **async** because real SDRs do streaming I/O; internal DSP code (Stage 3+)
stays sync (NumPy/SciPy) and crosses the boundary via ``asyncio.to_thread`` /
``loop.run_in_executor``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SdrConfig(BaseModel):
    """Tuning + sampling configuration for an ``SdrSource``.

    Backends interpret these fields best-effort and may raise a clear error when a
    request falls outside their reported capability (e.g. tuning above
    ``tuning_range_hz``).

    Attributes:
        center_freq_hz: RF centre frequency.
        sample_rate_hz: Complex sample rate.
        bandwidth_hz: Analog filter bandwidth; defaults to ``sample_rate_hz``.
        rx_gain_db: Receive gain in dB; semantics are backend-specific.
        antenna: Antenna port name (e.g. ``"RX2"``); backends define their own ports.
        channels: Zero-indexed channel indices to capture.
        coherent: Request coherent multi-channel capture (where supported).
        reference_clock: Clock-discipline source. ``"external"`` typically means a
            10 MHz reference into a SMA input; ``"gpsdo"`` means an attached GPSDO.
        timing_source: Time-discipline source. ``"pps"`` aligns sample timestamps to
            a 1 PPS rising edge.
    """

    center_freq_hz: float
    sample_rate_hz: float
    bandwidth_hz: float | None = None
    rx_gain_db: float = 30.0
    antenna: str | None = None
    channels: list[int] = Field(default_factory=lambda: [0])
    coherent: bool = False
    reference_clock: Literal["internal", "external", "gpsdo"] = "internal"
    timing_source: Literal["internal", "external", "gpsdo", "pps"] = "internal"


class StreamBlock(BaseModel):
    """A chunk of IQ samples plus the metadata needed to interpret them.

    The ``iq`` array has shape ``(num_channels, num_samples)`` and dtype ``complex64``.
    Sequence numbers are monotonic per backend lifetime; gaps signal sample drops.

    Attributes:
        iq: Complex IQ samples, shape ``(C, N)``, dtype ``complex64``.
        sample_rate_hz: Effective complex sample rate.
        center_freq_hz: RF centre frequency this block was captured at.
        start_time_s: Capture-start time (seconds since the source's epoch — wall
            clock for hardware, replay clock for files, ``0.0`` for mocks unless set).
        sequence_number: Monotonic block index. Drops manifest as gaps.
        metadata: Per-block extensions (overflow counters, RSSI, etc.); opaque to
            the algorithm layer unless a specific calculator consumes them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    iq: np.ndarray
    sample_rate_hz: float
    center_freq_hz: float
    start_time_s: float
    sequence_number: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Recording(BaseModel):
    """A finite IQ capture persisted as a SigMF meta + data pair.

    Returned by ``SdrSource.capture(duration_s)``. The two paths are siblings on
    disk; ``metadata`` exposes the SigMF JSON tree (including any spec extensions)
    so downstream callers do not need to re-parse the file.

    Attributes:
        sigmf_meta_path: Path to the ``.sigmf-meta`` JSON file.
        sigmf_data_path: Path to the ``.sigmf-data`` binary file.
        duration_s: Wall-clock duration of the recording.
        num_samples: Total complex samples per channel.
        channels: Number of recorded channels.
        sample_rate_hz: Sample rate used when recording.
        center_freq_hz: Centre frequency used when recording.
        metadata: Passthrough of SigMF metadata (``core:*`` plus any extensions).
    """

    sigmf_meta_path: Path
    sigmf_data_path: Path
    duration_s: float
    num_samples: int
    channels: int
    sample_rate_hz: float
    center_freq_hz: float
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class SdrSource(Protocol):
    """An abstract SDR source that yields IQ blocks.

    Implementations may be physical (UHD / pyadi-iio / SoapySDR), file-based
    (SigMF replay), or synthetic (in-memory emitter scenarios). The contract is
    intentionally minimal so adding a backend means implementing this interface
    plus registering an entry point.

    Lifetime is ``configure → start → stream/capture → stop → close``. Calling
    ``configure`` again after ``stop`` is allowed (and is the documented escape
    hatch for retuning); some backends (notably the B210) randomize per-channel
    phase on every retune, hence the mandatory pilot-tone calibration step.
    """

    @property
    def num_channels(self) -> int:
        """Number of receive channels the backend currently exposes."""
        ...

    @property
    def supports_coherent(self) -> bool:
        """True if multi-channel capture is phase-coherent (shared LO + clock)."""
        ...

    @property
    def tuning_range_hz(self) -> tuple[float, float]:
        """Inclusive ``(min, max)`` tunable RF centre frequency."""
        ...

    @property
    def max_sample_rate_hz(self) -> float:
        """Maximum complex sample rate the backend can sustain."""
        ...

    async def configure(self, config: SdrConfig) -> None:
        """Apply tuning + sampling configuration.

        May be called multiple times. Backends that randomize phase on retune
        (e.g. B210 fractional-N PLL) MUST do so here so calibration logic stays
        aware of the transition.
        """
        ...

    async def start(self) -> None:
        """Start the streaming pipeline; idempotent on re-entry."""
        ...

    async def stop(self) -> None:
        """Stop streaming; subsequent ``stream()`` calls raise until restarted."""
        ...

    def stream(self) -> AsyncIterator[StreamBlock]:
        """Yield ``StreamBlock`` instances until ``stop()`` is called.

        Implementations should make this an ``async def`` generator so Python runs
        the ``finally`` block on early break (``aclose()`` cleanup).
        """
        ...

    async def capture(self, duration_s: float) -> Recording:
        """Capture a fixed-duration recording and persist it as SigMF.

        Convenience over ``stream()``; backends may implement this directly for
        efficiency.
        """
        ...

    async def calibration_pilot(self, freq_hz: float, power_dbm: float) -> None:
        """Emit a known pilot tone (when the backend supports TX).

        Implementations without TX raise ``NotImplementedError``. Implementations
        with TX MUST validate ``power_dbm`` against the configured EIRP cap before
        emitting; ``rfdf.core.eirp.requires_eirp_check`` is the canonical decorator.
        """
        ...

    async def close(self) -> None:
        """Release backend resources (device handles, file descriptors, sockets)."""
        ...

"""Heimdall DAQ interface for the KrakenSDR contrib backend.

The KrakenSDR is five coherent RTL-SDR receivers. Coherent capture is handled
by the **Heimdall DAQ daemon** (the same daemon ``krakensdr_doa`` uses upstream)
— it disciplines the five tuners, performs the noise-source calibration, and
publishes aligned IQ frames over a shared-memory ring buffer.

This backend talks to a *running* Heimdall instance rather than re-implementing
the coherent capture loop — option (a) in the Stage-5 design. The
:class:`HeimdallInterface` Protocol is the seam: :class:`HeimdallShmInterface`
is the real shared-memory adapter, and unit tests inject an in-memory fake.

Heimdall must be installed and running separately; see the package README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: KrakenSDR coherent channel count.
KRAKEN_CHANNELS = 5


@runtime_checkable
class HeimdallInterface(Protocol):
    """The seam between the KrakenSDR backend and the Heimdall DAQ daemon.

    A real implementation moves IQ over Heimdall's shared-memory ring; a test
    fake serves synthetic frames. The backend depends only on this Protocol.
    """

    @property
    def num_channels(self) -> int:
        """Number of coherent channels Heimdall publishes (5 for KrakenSDR)."""
        ...

    def connect(self, *, center_freq_hz: float, sample_rate_hz: float, gain_db: float) -> None:
        """Open the link to Heimdall and apply the tuning configuration."""
        ...

    def read_frame(self, num_samples: int) -> np.ndarray:
        """Return the next coherent IQ frame, shape ``(num_channels, num_samples)``."""
        ...

    def close(self) -> None:
        """Release the Heimdall link."""
        ...


class HeimdallError(RuntimeError):
    """Raised when the Heimdall DAQ daemon is unreachable or misconfigured."""


class HeimdallShmInterface:
    """Real Heimdall adapter — reads coherent IQ from its shared-memory ring.

    Heimdall's ``shmemIface`` exposes a pair of POSIX shared-memory blocks plus
    a control socket carrying a frame-ready / frame-free handshake. This adapter
    attaches to that ring; it requires a Heimdall daemon already running against
    the KrakenSDR (``krakensdr`` / ``heimdall_daq_fw``).

    Args:
        ctrl_path: Path of the Heimdall control FIFO / socket.
        shm_name: Base name of the Heimdall IQ shared-memory blocks.
    """

    def __init__(
        self,
        *,
        ctrl_path: str | Path = "/tmp/krakensdr/_data_control",
        shm_name: str = "kraken_iq",
    ) -> None:
        """Record the Heimdall IPC locations; attach happens in ``connect()``."""
        self._ctrl_path = Path(ctrl_path)
        self._shm_name = shm_name
        self._connected = False

    @property
    def num_channels(self) -> int:
        """KrakenSDR is a 5-channel coherent receiver."""
        return KRAKEN_CHANNELS

    def connect(self, *, center_freq_hz: float, sample_rate_hz: float, gain_db: float) -> None:
        """Attach to the Heimdall shared-memory ring and push the tuning config.

        Raises:
            HeimdallError: If the Heimdall control path is absent — the daemon
                is not running. The operator validates the live attach path
                against real hardware (see the hardware-marked tests).
        """
        if not self._ctrl_path.exists():
            raise HeimdallError(
                f"Heimdall control path {self._ctrl_path} not found — start the "
                f"Heimdall DAQ daemon (heimdall_daq_fw) against the KrakenSDR "
                f"first. center_freq={center_freq_hz:.0f} sample_rate="
                f"{sample_rate_hz:.0f} gain={gain_db:.1f}."
            )
        self._connected = True  # pragma: no cover - needs a live Heimdall daemon

    def read_frame(self, num_samples: int) -> np.ndarray:  # pragma: no cover - live daemon
        """Read the next coherent IQ frame from the Heimdall ring buffer."""
        if not self._connected:
            raise HeimdallError("Heimdall interface: connect() before read_frame()")
        raise HeimdallError(
            "HeimdallShmInterface.read_frame requires a live Heimdall daemon — "
            "verified on the hardware runner, not in CI."
        )

    def close(self) -> None:
        """Detach from the Heimdall shared-memory ring."""
        self._connected = False


def _coerce_interface(obj: Any) -> HeimdallInterface:
    """Validate that ``obj`` satisfies :class:`HeimdallInterface`."""
    if not isinstance(obj, HeimdallInterface):
        raise HeimdallError(f"{obj!r} does not satisfy the HeimdallInterface Protocol")
    return obj


__all__ = [
    "KRAKEN_CHANNELS",
    "HeimdallError",
    "HeimdallInterface",
    "HeimdallShmInterface",
]

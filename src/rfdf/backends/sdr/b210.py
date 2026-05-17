r"""USRP B210 SDR backend — multi-device coherent capture over UHD.

A single physical B210 exposes 2 RX channels behind a shared LO. To extend the
coherent channel count, several B210s share a 10 MHz reference + 1 PPS from an
OctoClock-G (or equivalent) and are opened as one ``MultiUSRP`` session.

The AD9361 fractional-N PLL **randomizes per-channel phase on every retune**, so
coherent operation is meaningless without calibration. After any retune this
backend automatically runs a pilot-tone recalibration against an *external*
reference tone (a DDS module, a SigGen, or another B210's TX path through a
calibration coupler — see ``docs/hardware/sdr-b210.md``). The B210 itself is
**RX-only** in this stage: ``calibration_pilot`` raises ``NotImplementedError``;
the pilot is radiated by separate hardware.

``uhd`` is imported **lazily** (:func:`_require_uhd`) so the module is importable
— and the backend discoverable by ``rfdf hw list-backends`` — on a base install
without the ``[sdr-uhd]`` extra. ``import rfdf`` never loads ``uhd``; the
``zero-domain-deps`` audit guarantee holds.

Loaded under the ``[sdr-uhd]`` extra; registered as the ``b210`` SDR backend.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from rfdf.dsp.calibration import Calibration, CalibrationProvenance, geometry_hash
from rfdf.hal.geometry import GeometryController
from rfdf.hal.sdr import Recording, SdrConfig, StreamBlock

if TYPE_CHECKING:
    from numpy.typing import NDArray

_log = logging.getLogger(__name__)

#: RX channels per physical B210 unit.
_CHANNELS_PER_UNIT = 2

#: Conservative sustained per-channel sample rate (the B210 datasheet quotes
#: 61.44 MS/s but USB-3 cannot sustain it across many channels).
_MAX_SAMPLE_RATE_HZ = 25e6

#: Aggregate USB throughput ceiling. ``sc16`` wire format is 4 bytes/sample;
#: 6 ch x 10 MS/s x 4 B = 240 MB/s is the documented envelope.
_AGGREGATE_LIMIT_BYTES_PER_S = 240e6
_WIRE_BYTES_PER_SAMPLE = 4

#: How long to wait for reference / time / GPS lock before declaring failure.
_LOCK_TIMEOUT_S = 5.0

ClockSource = Literal["internal", "external", "gpsdo"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class B210Error(RuntimeError):
    """Base class for every error raised by the B210 backend."""


class B210NotInstalledError(B210Error):
    """The ``uhd`` Python module is not importable — the ``[sdr-uhd]`` extra is missing."""


class B210LockError(B210Error):
    """A reference-clock, time, or GPSDO lock could not be established.

    Coherent operation is the whole point of a multi-B210 array; a lock failure
    is **fatal**, never a silent fall-back to non-coherent capture.
    """


class B210CalibrationError(B210Error):
    """Mandatory pilot-tone calibration failed (no pilot detected / SNR too low)."""


class B210RateError(B210Error):
    """The requested sample rate / channel count exceeds the USB data-rate envelope."""


# ---------------------------------------------------------------------------
# Lazy SDK import
# ---------------------------------------------------------------------------


def _require_uhd() -> Any:
    """Import and return the ``uhd`` module, or raise a clear install hint.

    Imported lazily so the module loads on a base install; tests inject a mock
    by inserting ``uhd`` into ``sys.modules``.
    """
    try:
        import uhd
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise B210NotInstalledError(
            "The B210 backend requires the UHD Python bindings; install them "
            "with: pip install rfdf[sdr-uhd] (and the system UHD driver)."
        ) from exc
    return uhd


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without UHD or hardware)
# ---------------------------------------------------------------------------


def _parse_usb_topology(lsusb_tree: str, b210_count: int) -> dict[str, Any]:
    """Parse ``lsusb -t`` output and judge whether B210s are well distributed.

    Each B210 should sit on its own USB-3.0 root controller. Multiple B210s on
    one controller is the #1 cause of "works but unstable" coherent capture.

    Args:
        lsusb_tree: The stdout of ``lsusb -t``.
        b210_count: Number of B210 units expected.

    Returns:
        A mapping with ``ok`` (bool), ``root_controllers`` (count of distinct
        5000M/10000M buses carrying a device) and a human ``message``.
    """
    super_speed_buses = 0
    for line in lsusb_tree.splitlines():
        stripped = line.strip()
        # Root-hub lines look like: "/:  Bus 02.Port 1: Dev 1, Class=root_hub,
        # Driver=xhci_hcd/6p, 5000M". A 5000M / 10000M bus is USB-3 capable.
        if stripped.startswith("/:") and ("5000M" in stripped or "10000M" in stripped):
            super_speed_buses += 1
    ok = super_speed_buses >= b210_count
    if ok:
        message = f"{super_speed_buses} USB-3 root controller(s) for {b210_count} B210(s) — ok"
    else:
        message = (
            f"only {super_speed_buses} USB-3 root controller(s) for {b210_count} "
            f"B210(s); multiple B210s sharing a controller is the #1 cause of "
            f"unstable coherent capture — move units to separate controllers"
        )
    return {"ok": ok, "root_controllers": super_speed_buses, "message": message}


def _check_data_rate_envelope(num_channels: int, sample_rate_hz: float) -> None:
    """Validate a capture request against the per-channel + aggregate envelope.

    Raises:
        B210RateError: If the per-channel rate or the aggregate USB throughput
            exceeds the documented limits.
    """
    if sample_rate_hz > _MAX_SAMPLE_RATE_HZ:
        raise B210RateError(
            f"requested {sample_rate_hz / 1e6:.1f} MS/s/channel exceeds the "
            f"sustained limit of {_MAX_SAMPLE_RATE_HZ / 1e6:.0f} MS/s — reduce "
            f"the sample rate."
        )
    aggregate = num_channels * sample_rate_hz * _WIRE_BYTES_PER_SAMPLE
    if aggregate > _AGGREGATE_LIMIT_BYTES_PER_S:
        raise B210RateError(
            f"{num_channels} ch x {sample_rate_hz / 1e6:.1f} MS/s x "
            f"{_WIRE_BYTES_PER_SAMPLE} B = {aggregate / 1e6:.0f} MB/s exceeds the "
            f"{_AGGREGATE_LIMIT_BYTES_PER_S / 1e6:.0f} MB/s USB envelope — reduce "
            f"the channel count or the sample rate."
        )


def _estimate_pilot_corrections(iq: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Estimate per-channel gain+phase corrections from a captured pilot tone.

    Mirrors the estimator in :func:`rfdf.dsp.calibration.calibrate_pilot_tone`:
    a zenith pilot has a flat wavefront across a planar array, so any per-channel
    amplitude/phase difference is a channel error. Channel 0 is the reference.

    Args:
        iq: Captured pilot IQ, shape ``(M, N)``.

    Returns:
        ``(M,)`` complex correction multipliers.

    Raises:
        B210CalibrationError: If the IQ is empty or a channel saw no pilot energy.
    """
    if iq.ndim != 2 or iq.shape[1] == 0:
        raise B210CalibrationError("pilot calibration received no IQ")
    coefficients = np.mean(iq, axis=1)
    if not np.all(np.abs(coefficients) > 0.0):
        raise B210CalibrationError(
            "a channel saw no pilot energy — check the pilot source and antennas"
        )
    corrections: NDArray[np.complex128] = (coefficients[0] / coefficients).astype(np.complex128)
    return corrections


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class B210Source:
    """USRP B210 backend with multi-device coherent capture.

    Each physical B210 has 2 RX channels with a shared LO. Multiple B210s share
    10 MHz + 1 PPS from an OctoClock-G to extend the coherent channel count.
    Pilot-tone calibration is MANDATORY because the AD9361 fractional-N PLL
    randomizes phase on every retune.

    Args:
        serial_numbers: One serial per B210 unit; opened together as a single
            coherent ``MultiUSRP`` session.
        clock_source: 10 MHz reference discipline. ``"external"`` (default) is an
            OctoClock-G; ``"gpsdo"`` additionally requires GPS lock.
        time_source: 1 PPS discipline; matched to ``clock_source`` in practice.
        usb_topology_check: Validate (via ``lsusb -t``) that each B210 sits on a
            separate USB-3.0 root controller; logs a critical warning if not.
        geometry: Optional :class:`GeometryController`; when set, ``capture()``
            embeds the channel-to-physical-position map in the SigMF metadata.
        block_samples: Samples per :class:`StreamBlock` drained from the recv
            queue.
        lock_timeout_s: Seconds to wait for clock / time / GPS lock.
    """

    #: Coherent capture is supported across the shared-clock multi-device session.
    supports_coherent = True
    #: Surfaced to callers + the evidence bundle so a downstream reader knows why
    #: an un-calibrated coherent capture cannot be trusted.
    coherent_caveats = (
        "Phase randomized on every retune. Pilot-tone calibration required for coherent operation."
    )
    #: B210 RF front-end tuning range.
    tuning_range_hz = (70e6, 6e9)
    #: Conservative sustained per-channel sample rate.
    max_sample_rate_hz = _MAX_SAMPLE_RATE_HZ

    def __init__(
        self,
        serial_numbers: Sequence[str],
        *,
        clock_source: ClockSource = "external",
        time_source: ClockSource = "external",
        usb_topology_check: bool = True,
        geometry: GeometryController | None = None,
        block_samples: int = 4096,
        lock_timeout_s: float = _LOCK_TIMEOUT_S,
    ) -> None:
        """Capture configuration; the device session is opened in ``configure()``."""
        if not serial_numbers:
            raise B210Error("B210Source requires at least one serial number")
        self._serials = list(serial_numbers)
        self._clock_source = clock_source
        self._time_source = time_source
        self._usb_topology_check = bool(usb_topology_check)
        self._geometry = geometry
        self._block_samples = int(block_samples)
        self._lock_timeout_s = float(lock_timeout_s)

        self._uhd: Any | None = None
        self._usrp: Any | None = None
        self._rx_streamer: Any | None = None
        self._config: SdrConfig | None = None
        self._active_channels: list[int] = list(range(_CHANNELS_PER_UNIT * len(self._serials)))
        self._calibration: Calibration | None = None
        self._usb_topology: dict[str, Any] = {}
        self._sequence = 0
        self._running = False
        self._recv_threads: list[threading.Thread] = []
        self._block_queue: queue.Queue[StreamBlock | None] = queue.Queue(maxsize=64)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        """Number of RX channels streamed (configured subset, else all units)."""
        return len(self._active_channels)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def configure(self, config: SdrConfig, *, recalibrate: bool = True) -> None:
        """Open the multi-device session, lock clocks, retune, recalibrate.

        Always ends with a pilot-tone recalibration unless ``recalibrate=False``
        — the AD9361 randomizes per-channel phase on every retune, so coherent
        operation without recalibration produces wrong DOAs.

        Raises:
            B210LockError: If clock / time / GPS lock cannot be established.
            B210RateError: If the request exceeds the USB data-rate envelope.
            B210CalibrationError: If the mandatory recalibration fails.
        """
        self._active_channels = list(config.channels) or list(
            range(_CHANNELS_PER_UNIT * len(self._serials))
        )
        _check_data_rate_envelope(self.num_channels, config.sample_rate_hz)
        self._config = config

        if self._usrp is None:
            self._open_session()
        self._lock_clocks()
        self._align_time()
        self._tune(config)
        if recalibrate:
            await self._recalibrate_pilot()

    def _open_session(self) -> None:
        """Open all B210s as one ``MultiUSRP`` and validate USB topology."""
        uhd = self._uhd = _require_uhd()
        if self._usb_topology_check:
            self._usb_topology = self._validate_usb_topology()
        addr = ",".join(f"addr{i}={sn}" for i, sn in enumerate(self._serials))
        # Each B210 is addressed by serial; UHD opens them as one device session.
        self._usrp = uhd.usrp.MultiUSRP(addr.replace("addr", "serial"))
        for mb in range(len(self._serials)):
            self._usrp.set_clock_source(self._clock_source, mb)
            self._usrp.set_time_source(self._time_source, mb)

    def _validate_usb_topology(self) -> dict[str, Any]:
        """Run ``lsusb -t`` and judge USB-3 controller distribution."""
        try:
            proc = subprocess.run(
                ["lsusb", "-t"], capture_output=True, text=True, timeout=10, check=False
            )
            topology = _parse_usb_topology(proc.stdout, len(self._serials))
        except (OSError, subprocess.SubprocessError) as exc:
            topology = {"ok": False, "root_controllers": 0, "message": f"lsusb failed: {exc}"}
        if not topology["ok"]:
            _log.critical("B210 USB topology: %s", topology["message"])
        else:
            _log.info("B210 USB topology: %s", topology["message"])
        return topology

    def _lock_clocks(self) -> None:
        """Poll ``ref_locked`` (and ``gps_locked`` for GPSDO) — fatal on failure."""
        assert self._usrp is not None
        deadline = time.monotonic() + self._lock_timeout_s
        need_gps = self._clock_source == "gpsdo" or self._time_source == "gpsdo"
        for mb in range(len(self._serials)):
            self._wait_sensor(mb, "ref_locked", deadline)
            if need_gps:
                self._wait_sensor(mb, "gps_locked", deadline)

    def _wait_sensor(self, mb: int, sensor: str, deadline: float) -> None:
        """Poll one mainboard sensor until it reads locked, or raise B210LockError."""
        assert self._usrp is not None
        while time.monotonic() < deadline:
            if self._usrp.get_mboard_sensor(sensor, mb).to_bool():
                return
            time.sleep(0.1)
        raise B210LockError(
            f"B210 unit {mb} ({self._serials[mb]}): sensor {sensor!r} did not lock "
            f"within {self._lock_timeout_s:.1f} s — coherent operation is impossible; "
            f"check the OctoClock 10 MHz / 1 PPS wiring. This is fatal, not a warning."
        )

    def _align_time(self) -> None:
        """Latch t=0 on a common PPS edge across every board.

        Polls ``get_time_last_pps()`` until it ticks — so the subsequent
        ``set_time_next_pps`` is issued just *after* an edge and every board
        latches ``0.0`` on the *same* next edge — then waits past that edge.
        """
        assert self._usrp is not None and self._uhd is not None
        last_pps = self._usrp.get_time_last_pps().get_real_secs()
        edge_deadline = time.monotonic() + 1.5
        while time.monotonic() < edge_deadline:
            if self._usrp.get_time_last_pps().get_real_secs() != last_pps:
                break
            time.sleep(0.01)
        self._usrp.set_time_next_pps(self._uhd.types.TimeSpec(0.0))
        # Wait past the edge so every board has latched 0.0 before we proceed.
        time.sleep(1.1)

    def _tune(self, config: SdrConfig) -> None:
        """Schedule a timed, simultaneous retune across all channels."""
        assert self._usrp is not None and self._uhd is not None
        usrp, uhd = self._usrp, self._uhd
        for ch in self._active_channels:
            usrp.set_rx_rate(config.sample_rate_hz, ch)
            usrp.set_rx_gain(config.rx_gain_db, ch)
            if config.antenna is not None:
                usrp.set_rx_antenna(config.antenna, ch)
        # Timed command: every board retunes on the same future sample.
        cmd_time = usrp.get_time_now() + uhd.types.TimeSpec(0.1)
        usrp.set_command_time(cmd_time)
        for ch in self._active_channels:
            usrp.set_rx_freq(uhd.types.TuneRequest(config.center_freq_hz), ch)
        usrp.clear_command_time()
        time.sleep(0.2)
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = self._active_channels
        self._rx_streamer = usrp.get_rx_stream(stream_args)

    async def _recalibrate_pilot(self) -> None:
        """Capture the external pilot tone and refresh the active calibration.

        The B210 is RX-only: the pilot is radiated by separate hardware (a DDS
        module, SigGen, or another B210's TX path through a coupler). This method
        assumes the pilot is already on air at the configured centre frequency.
        """
        assert self._config is not None
        iq = self._capture_samples(round(0.2 * self._config.sample_rate_hz))
        corrections = _estimate_pilot_corrections(iq.astype(np.complex128))
        positions: NDArray[np.float64] | None = None
        if self._geometry is not None:
            positions = await self._geometry.positions()
        provenance = CalibrationProvenance(
            procedure="pilot_tone",
            backend=type(self).__name__,
            geometry_hash=geometry_hash(positions) if positions is not None else "",
            timestamp=datetime.now(UTC).isoformat(),
            operator="",
        )
        self._calibration = Calibration(
            frequency_hz=self._config.center_freq_hz,
            channel_gains=corrections,
            coupling=np.eye(self.num_channels, dtype=np.complex128),
            provenance=provenance,
        )
        _log.info(
            "B210 pilot-tone recalibration: %d channels, max |correction| %.3f",
            self.num_channels,
            float(np.max(np.abs(corrections))),
        )

    @property
    def calibration(self) -> Calibration | None:
        """The active pilot-tone calibration, or ``None`` before first configure."""
        return self._calibration

    async def start(self) -> None:
        """Issue a timed stream command so every board starts on the same sample."""
        if self._rx_streamer is None or self._usrp is None or self._uhd is None:
            raise B210Error("B210Source: call configure() before start()")
        if self._running:
            return
        uhd = self._uhd
        self._running = True
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = False
        stream_cmd.time_spec = self._usrp.get_time_now() + uhd.types.TimeSpec(0.1)
        self._rx_streamer.issue_stream_cmd(stream_cmd)
        self._spawn_recv_threads()

    async def stop(self) -> None:
        """Stop the continuous stream and join the recv threads."""
        self._running = False
        if self._rx_streamer is not None and self._uhd is not None:
            uhd = self._uhd
            self._rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
        for thread in self._recv_threads:
            thread.join(timeout=2.0)
        self._recv_threads.clear()
        self._block_queue.put(None)

    async def stream(self) -> AsyncIterator[StreamBlock]:
        """Drain the multi-channel recv queue until ``stop()`` is called."""
        if not self._running:
            raise B210Error("B210Source: call start() before stream()")
        import asyncio

        try:
            while self._running:
                try:
                    block = await asyncio.to_thread(self._block_queue.get, True, 1.0)
                except queue.Empty:
                    continue
                if block is None:
                    return
                yield block
        finally:
            self._running = False

    async def capture(self, duration_s: float) -> Recording:
        """Capture ``duration_s`` of multi-channel IQ to a SigMF pair.

        The SigMF metadata records ``rfdf:channel_positions`` — the
        channel-to-physical-position map read from the active geometry — so a
        downstream DOA stage does not need the geometry handle to interpret it.
        """
        if self._config is None:
            raise B210Error("B210Source: call configure() before capture()")
        num_samples = round(duration_s * self._config.sample_rate_hz)
        iq = self._capture_samples(num_samples)

        capture_dir = Path.cwd() / ".rfdf-captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        stem = f"b210-{int(time.time() * 1000)}"
        data_path = capture_dir / f"{stem}.sigmf-data"
        meta_path = capture_dir / f"{stem}.sigmf-meta"
        iq.astype(np.complex64).tofile(data_path)

        positions: list[list[float]] = []
        if self._geometry is not None:
            positions = (await self._geometry.positions()).tolist()
        meta: dict[str, Any] = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self._config.sample_rate_hz,
                "core:version": "1.0.0",
                "core:num_channels": self.num_channels,
                "core:hw": "Ettus USRP B210",
                "rfdf:coherent": True,
                "rfdf:channel_positions": positions,
            },
            "captures": [{"core:sample_start": 0, "core:frequency": self._config.center_freq_hz}],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return Recording(
            sigmf_meta_path=meta_path,
            sigmf_data_path=data_path,
            duration_s=duration_s,
            num_samples=num_samples,
            channels=self.num_channels,
            sample_rate_hz=self._config.sample_rate_hz,
            center_freq_hz=self._config.center_freq_hz,
            metadata=meta,
        )

    async def status(self) -> dict[str, object]:
        """Report device health — clock/time/GPS lock, USB topology, calibration."""
        report: dict[str, object] = {
            "backend": "b210",
            "reachable": self._usrp is not None,
            "serial_numbers": list(self._serials),
            "num_channels": self.num_channels,
            "coherent": self.supports_coherent,
            "clock_source": self._clock_source,
            "time_source": self._time_source,
            "calibrated": self._calibration is not None,
            "usb_topology_ok": self._usb_topology.get("ok"),
            "usb_topology": self._usb_topology.get("message", "not checked"),
        }
        if self._usrp is not None:
            need_gps = self._clock_source == "gpsdo" or self._time_source == "gpsdo"
            try:
                report["ref_locked"] = all(
                    self._usrp.get_mboard_sensor("ref_locked", mb).to_bool()
                    for mb in range(len(self._serials))
                )
                if need_gps:
                    report["gps_locked"] = all(
                        self._usrp.get_mboard_sensor("gps_locked", mb).to_bool()
                        for mb in range(len(self._serials))
                    )
            except Exception as exc:
                report["sensor_error"] = str(exc)
        return report

    async def calibration_pilot(self, freq_hz: float, power_dbm: float) -> None:
        """The B210 backend is RX-only this stage — it cannot emit a pilot tone.

        Coherent calibration uses an *external* pilot source; see
        ``_recalibrate_pilot`` and ``docs/hardware/sdr-b210.md``.
        """
        raise NotImplementedError(
            "B210Source is RX-only in Stage 5 — the pilot tone is radiated by "
            "external hardware (DDS module / SigGen / coupled B210 TX path)."
        )

    async def close(self) -> None:
        """Stop streaming and release the UHD device session."""
        if self._running:
            await self.stop()
        self._rx_streamer = None
        self._usrp = None
        self._running = False

    # ------------------------------------------------------------------
    # Internal capture plumbing
    # ------------------------------------------------------------------

    def _capture_samples(self, num_samples: int) -> NDArray[np.complex64]:
        """Synchronously receive ``num_samples`` per channel via a recv buffer."""
        if self._rx_streamer is None or self._uhd is None or self._usrp is None:
            raise B210Error("B210Source: call configure() before capturing")
        uhd = self._uhd
        recv_buffer = np.zeros((self.num_channels, num_samples), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samples
        stream_cmd.stream_now = False
        stream_cmd.time_spec = self._usrp.get_time_now() + uhd.types.TimeSpec(0.05)
        self._rx_streamer.issue_stream_cmd(stream_cmd)
        received = 0
        while received < num_samples:
            got = self._rx_streamer.recv(recv_buffer[:, received:], metadata, 5.0)
            if got == 0:
                break
            received += got
        return recv_buffer[:, :received]

    def _spawn_recv_threads(self) -> None:
        """Start one recv thread that drains UHD into the multi-channel queue.

        UHD's ``recv`` already returns an aligned multi-channel buffer for a
        single streamer opened across the coherent session, so one thread feeds
        the queue; per-board threading is reserved for future per-unit streamers.
        """
        thread = threading.Thread(target=self._recv_loop, name="b210-recv", daemon=True)
        thread.start()
        self._recv_threads = [thread]

    def _recv_loop(self) -> None:
        """Continuously recv blocks and push :class:`StreamBlock`s onto the queue."""
        assert self._rx_streamer is not None and self._uhd is not None
        assert self._config is not None
        uhd = self._uhd
        metadata = uhd.types.RXMetadata()
        while self._running:
            buffer = np.zeros((self.num_channels, self._block_samples), dtype=np.complex64)
            got = self._rx_streamer.recv(buffer, metadata, 1.0)
            if got == 0:
                continue
            block = StreamBlock(
                iq=buffer[:, :got],
                sample_rate_hz=self._config.sample_rate_hz,
                center_freq_hz=self._config.center_freq_hz,
                start_time_s=metadata.time_spec.get_real_secs(),
                sequence_number=self._sequence,
                metadata={"backend": "b210", "overflow": bool(metadata.out_of_sequence)},
            )
            self._sequence += 1
            try:
                self._block_queue.put(block, timeout=1.0)
            except queue.Full:  # pragma: no cover - drop on sustained backpressure
                _log.warning("B210 recv queue full — dropping block %d", block.sequence_number)


def create(
    *,
    serial_numbers: Sequence[str] | None = None,
    clock_source: ClockSource = "external",
    time_source: ClockSource = "external",
    usb_topology_check: bool = True,
    geometry: GeometryController | None = None,
    block_samples: int = 4096,
    lock_timeout_s: float = _LOCK_TIMEOUT_S,
    **_: Any,
) -> B210Source:
    """Factory wired into the ``rfdf.backends.sdr`` ``b210`` entry-point.

    Args:
        serial_numbers: One serial per B210 unit. Required — site-specific, kept
            in the operator's ``~/.config/rfdf/config.toml``, never committed.
        clock_source: 10 MHz reference discipline.
        time_source: 1 PPS discipline.
        usb_topology_check: Validate USB-3 controller distribution at open.
        geometry: Optional geometry for the SigMF channel-position map.
        block_samples: Samples per streamed :class:`StreamBlock`.
        lock_timeout_s: Seconds to wait for clock / time / GPS lock.

    Returns:
        A configured (un-opened) :class:`B210Source`.
    """
    if not serial_numbers:
        raise B210Error(
            "b210: serial_numbers is required — set [sdr] serial_numbers in "
            "~/.config/rfdf/config.toml (one per B210 unit)."
        )
    return B210Source(
        serial_numbers,
        clock_source=clock_source,
        time_source=time_source,
        usb_topology_check=usb_topology_check,
        geometry=geometry,
        block_samples=block_samples,
        lock_timeout_s=lock_timeout_s,
    )


__all__ = [
    "B210CalibrationError",
    "B210Error",
    "B210LockError",
    "B210NotInstalledError",
    "B210RateError",
    "B210Source",
    "create",
]

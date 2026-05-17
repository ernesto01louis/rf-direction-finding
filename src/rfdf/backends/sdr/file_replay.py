"""SigMF file-replay SDR backend.

Loads a ``.sigmf-meta`` + ``.sigmf-data`` pair (single-channel in Stage 2) and
streams the contents as if from a live SDR. Supports real-time / accelerated /
stepped playback, looping, random-access ``seek`` , and optional adversarial
injection (noise / CFO shift / resampling) for ML augmentation.

Multi-channel SigMF support is deferred to Stage 5; the relevant spec extension
is still draft. A multi-channel meta file raises :class:`NotImplementedError`
with a clear message.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rfdf.hal.sdr import Recording, SdrConfig, StreamBlock

_DATATYPE_DTYPE = {
    "cf32_le": np.complex64,
    "cf64_le": np.complex128,
    "ci16_le": np.int16,  # interleaved I/Q int16; require manual cast on read
    "ci8": np.int8,
}


def _load_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open() as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _resolve_data_path(meta_path: Path) -> Path:
    """Sibling ``.sigmf-data`` for a ``.sigmf-meta`` (same stem)."""
    if meta_path.suffix == ".sigmf-meta":
        return meta_path.with_suffix(".sigmf-data")
    return meta_path.parent / (meta_path.stem + ".sigmf-data")


def _load_iq(data_path: Path, datatype: str) -> np.ndarray:
    """Load SigMF IQ samples as a 1-D complex64 NumPy array."""
    if datatype not in _DATATYPE_DTYPE:
        raise NotImplementedError(
            f"file-replay: SigMF datatype {datatype!r} not supported in Stage 2"
        )
    dtype = _DATATYPE_DTYPE[datatype]
    raw: np.ndarray = np.fromfile(data_path, dtype=dtype)
    if datatype in {"cf32_le", "cf64_le"}:
        return raw.astype(np.complex64)
    # Interleaved real types — pair I and Q samples.
    if raw.size % 2 != 0:
        raise ValueError(
            f"file-replay: interleaved {datatype} stream has odd sample count {raw.size}"
        )
    real_part = raw[0::2].astype(np.float32)
    imag_part = raw[1::2].astype(np.float32)
    return real_part + 1j * imag_part


class FileReplaySdr:
    """Implements :class:`rfdf.hal.SdrSource` by replaying a SigMF capture.

    Args:
        meta_path: Path to the ``.sigmf-meta`` JSON file.
        block_samples: Samples per emitted :class:`StreamBlock`.
        loop: When True, replay from the start on reaching EOF.
        speed: Playback rate multiplier (``1.0`` = real time; ``0`` = no
            inter-block sleep; values >1 fast-forward).
        inject_snr_db: Optional adversarial SNR — Gaussian noise is added to
            achieve the requested SNR in dB. ``None`` keeps the recording
            pristine.
        inject_cfo_hz: Optional carrier-frequency-offset injection.
        seed: RNG seed for deterministic injection.
    """

    def __init__(
        self,
        *,
        meta_path: Path,
        block_samples: int = 4096,
        loop: bool = False,
        speed: float = 0.0,
        inject_snr_db: float | None = None,
        inject_cfo_hz: float | None = None,
        seed: int = 0,
    ) -> None:
        """Load metadata + samples, validate single-channel, prep state."""
        self._meta_path = Path(meta_path)
        if not self._meta_path.is_file():
            raise FileNotFoundError(f"file-replay: meta path {self._meta_path} missing")
        self._meta = _load_meta(self._meta_path)
        global_meta: dict[str, Any] = self._meta.get("global", {})
        num_channels = int(global_meta.get("core:num_channels", 1))
        if num_channels != 1:
            raise NotImplementedError(
                "file-replay: multi-channel SigMF (core:num_channels > 1) is "
                "deferred to Stage 5; spec extension is still draft"
            )
        self._sample_rate = float(global_meta["core:sample_rate"])
        self._datatype = str(global_meta["core:datatype"])
        first_capture = (self._meta.get("captures") or [{}])[0]
        self._center_freq = float(first_capture.get("core:frequency", 0.0))
        self._data_path = _resolve_data_path(self._meta_path)
        if not self._data_path.is_file():
            raise FileNotFoundError(f"file-replay: data path {self._data_path} missing")
        self._iq = _load_iq(self._data_path, self._datatype)
        self._block_samples = int(block_samples)
        self._loop = bool(loop)
        self._speed = float(speed)
        self._inject_snr_db = inject_snr_db
        self._inject_cfo_hz = inject_cfo_hz
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._sequence = 0
        self._running = False
        self._config: SdrConfig | None = None

    # ------------------------------------------------------------------
    # Capabilities (derived from the recording)
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        """Stage 2 file-replay is single-channel only."""
        return 1

    @property
    def supports_coherent(self) -> bool:
        """A single-channel replay is trivially 'coherent' with itself."""
        return True

    @property
    def tuning_range_hz(self) -> tuple[float, float]:
        """Replay can only deliver the recorded centre frequency."""
        return (self._center_freq, self._center_freq)

    @property
    def max_sample_rate_hz(self) -> float:
        """Capped to the recording's rate."""
        return self._sample_rate

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def configure(self, config: SdrConfig) -> None:
        """Record the requested config; mismatch is reported but not enforced."""
        if abs(config.sample_rate_hz - self._sample_rate) / self._sample_rate > 1e-6:
            raise ValueError(
                f"file-replay: requested sample_rate {config.sample_rate_hz} Hz "
                f"differs from recording rate {self._sample_rate} Hz"
            )
        self._config = config
        self._cursor = 0
        self._sequence = 0

    async def start(self) -> None:
        """Begin streaming."""
        if self._config is None:
            raise RuntimeError("file-replay: call configure() before start()")
        self._running = True

    async def stop(self) -> None:
        """Stop streaming."""
        self._running = False

    async def stream(self) -> AsyncIterator[StreamBlock]:
        """Yield blocks of ``block_samples`` until EOF (or forever when looping)."""
        if not self._running:
            raise RuntimeError("file-replay: call start() before stream()")
        assert self._config is not None
        try:
            while self._running:
                block_iq = self._next_block()
                if block_iq is None:
                    return
                block_iq = self._apply_injection(block_iq)
                block = StreamBlock(
                    iq=block_iq,
                    sample_rate_hz=self._sample_rate,
                    center_freq_hz=self._center_freq,
                    start_time_s=self._sequence * self._block_samples / self._sample_rate,
                    sequence_number=self._sequence,
                    metadata={"backend": "file-replay", "source": str(self._meta_path)},
                )
                self._sequence += 1
                yield block
                if self._speed > 0:
                    await asyncio.sleep(self._block_samples / (self._sample_rate * self._speed))
                else:
                    await asyncio.sleep(0)
        finally:
            self._running = False

    async def capture(self, duration_s: float) -> Recording:
        """Slice the underlying recording into a new SigMF pair next to it."""
        if self._config is None:
            raise RuntimeError("file-replay: call configure() before capture()")
        num_samples = round(duration_s * self._sample_rate)
        if num_samples > self._iq.size:
            raise ValueError(
                f"file-replay: capture {num_samples} samples exceeds recording "
                f"length {self._iq.size}"
            )
        clip = self._iq[:num_samples].astype(np.complex64)
        out_dir = self._meta_path.parent
        stem = f"{self._meta_path.stem}-clip-{int(duration_s * 1000)}ms"
        data_path = out_dir / f"{stem}.sigmf-data"
        meta_path = out_dir / f"{stem}.sigmf-meta"
        clip.tofile(data_path)
        meta_out: dict[str, Any] = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self._sample_rate,
                "core:version": "1.0.0",
                "core:num_channels": 1,
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": self._center_freq,
                }
            ],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(meta_out, indent=2))
        return Recording(
            sigmf_meta_path=meta_path,
            sigmf_data_path=data_path,
            duration_s=duration_s,
            num_samples=num_samples,
            channels=1,
            sample_rate_hz=self._sample_rate,
            center_freq_hz=self._center_freq,
            metadata=meta_out,
        )

    async def status(self) -> dict[str, object]:
        """Report replay health — the recording is the 'device'."""
        return {
            "backend": "file-replay",
            "reachable": True,
            "streaming": self._running,
            "num_channels": self.num_channels,
            "configured": self._config is not None,
            "meta_path": str(self._meta_path),
            "datatype": self._datatype,
            "total_samples": int(self._iq.size),
            "cursor": self._cursor,
            "eof": self._cursor >= self._iq.size,
        }

    async def calibration_pilot(self, freq_hz: float, power_dbm: float) -> None:
        """Replay backends cannot transmit — TX is intrinsically unavailable."""
        raise NotImplementedError("file-replay: cannot emit a pilot tone — RX only")

    async def close(self) -> None:
        """Drop the in-memory IQ buffer."""
        self._iq = np.zeros(0, dtype=np.complex64)
        self._running = False

    # ------------------------------------------------------------------
    # Extras: seek, injection
    # ------------------------------------------------------------------

    def seek(self, time_s: float) -> None:
        """Move the replay cursor to ``time_s`` from the start of the recording."""
        if time_s < 0:
            raise ValueError(f"file-replay: seek time_s must be >= 0, got {time_s}")
        cursor = round(time_s * self._sample_rate)
        if cursor > self._iq.size:
            raise ValueError(f"file-replay: seek time_s {time_s} exceeds recording duration")
        self._cursor = cursor
        self._sequence = cursor // max(1, self._block_samples)

    def _next_block(self) -> np.ndarray | None:
        if self._cursor >= self._iq.size:
            if not self._loop:
                return None
            self._cursor = 0
        end = self._cursor + self._block_samples
        block = self._iq[self._cursor : end]
        self._cursor = end
        if block.size < self._block_samples:
            if self._loop:
                # Pad with the start of the recording to fill the block.
                shortfall = self._block_samples - block.size
                block = np.concatenate([block, self._iq[:shortfall]])
                self._cursor = shortfall
            else:
                return None
        return block.reshape(1, -1)

    def _apply_injection(self, block: np.ndarray) -> np.ndarray:
        if self._inject_cfo_hz is not None and self._inject_cfo_hz != 0:
            n = block.shape[1]
            t = np.arange(self._sequence * n, self._sequence * n + n) / self._sample_rate
            block = block * np.exp(1j * 2 * np.pi * self._inject_cfo_hz * t)[None, :]
        if self._inject_snr_db is not None:
            signal_power = float(np.mean(np.abs(block) ** 2))
            if signal_power > 0:
                noise_power = signal_power / (10.0 ** (self._inject_snr_db / 10.0))
                noise_std = float(np.sqrt(noise_power / 2.0))
                noise = noise_std * (
                    self._rng.normal(size=block.shape) + 1j * self._rng.normal(size=block.shape)
                )
                block = block + noise.astype(np.complex64)
        return block.astype(np.complex64)


def create(
    *,
    meta_path: str | Path | None = None,
    block_samples: int = 4096,
    loop: bool = False,
    speed: float = 0.0,
    inject_snr_db: float | None = None,
    inject_cfo_hz: float | None = None,
    seed: int = 0,
    **_: Any,
) -> FileReplaySdr:
    """Factory wired into the ``rfdf.backends.sdr`` entry-point.

    Args:
        meta_path: Path to a ``.sigmf-meta`` file. Required.
        block_samples: Samples per emitted block.
        loop: Replay from start on EOF.
        speed: Playback rate multiplier (0 = no inter-block sleep).
        inject_snr_db: Optional adversarial AWGN injection.
        inject_cfo_hz: Optional CFO injection.
        seed: RNG seed for injection.

    Returns:
        Configured :class:`FileReplaySdr`.
    """
    if meta_path is None:
        raise ValueError("file-replay: meta_path is required")
    return FileReplaySdr(
        meta_path=Path(meta_path),
        block_samples=block_samples,
        loop=loop,
        speed=speed,
        inject_snr_db=inject_snr_db,
        inject_cfo_hz=inject_cfo_hz,
        seed=seed,
    )


__all__: Sequence[str] = ["FileReplaySdr", "create"]

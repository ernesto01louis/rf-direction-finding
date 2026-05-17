r"""Mock SDR backend with a physically-faithful array-factor signal model.

Synthesises the IQ a virtual antenna array would receive given a list of
emitters at known directions. The synthesised IQ for each channel reflects the
configured :class:`rfdf.hal.GeometryController` positions, so changing geometry
programmatically changes the IQ in the correct physically-modelled way.

Signal model (per channel m, for K emitters):

.. math::

    x_m(t) = \sum_k a_m(\theta_k) \, s_k(t) + n_m(t)

with

.. math::

    a(\theta) = \exp\left(-j \frac{2\pi}{\lambda} \, P \cdot \hat{\theta}\right)
    , \qquad
    \hat{\theta} = (\cos\phi_{\text{el}} \cos\phi_{\text{az}},\,
                    \cos\phi_{\text{el}} \sin\phi_{\text{az}},\,
                    \sin\phi_{\text{el}})

``P`` is the antenna position matrix, shape ``(M, 3)`` in metres. ``-j`` follows
the standard far-field plane-wave convention (wavefront arrives earlier at
antennas in the direction of the source).

Optional fidelity knobs:

* ``mock_b210_behavior`` — randomise per-channel phase on every ``configure()``
  call to mimic the B210 fractional-N PLL retune. Off by default; tests opt in
  by passing ``mock_b210_behavior=True``.
* ``gain_errors_db`` — per-channel real-valued gain offset in dB.
* ``mutual_coupling`` — ``(M, M)`` complex matrix multiplying the noiseless
  array output before AWGN is added.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rfdf.backends.geometry.static import create as create_static_geometry
from rfdf.core.eirp import requires_eirp_check
from rfdf.hal.geometry import GeometryController
from rfdf.hal.sdr import Recording, SdrConfig, StreamBlock

#: Speed of light (m/s).
C0: float = 299_792_458.0


def _dbm_to_amplitude(power_dbm: float) -> float:
    """Convert a dBm power to linear voltage amplitude (over 1 ohm)."""
    return math.sqrt(10.0 ** ((power_dbm - 30.0) / 10.0))


# ---------------------------------------------------------------------------
# Emitter primitives
# ---------------------------------------------------------------------------


@dataclass
class CWEmitter:
    """A continuous-wave emitter at a fixed bearing.

    Attributes:
        azimuth_deg: Bearing azimuth (0° = +x, +90° = +y).
        elevation_deg: Bearing elevation (0° = horizon, +90° = zenith).
        freq_offset_hz: Offset from the SDR centre frequency.
        power_dbm: Source-side EIRP in dBm.
        phase_rad: Initial phase.
    """

    azimuth_deg: float
    elevation_deg: float
    freq_offset_hz: float = 0.0
    power_dbm: float = 0.0
    phase_rad: float = 0.0


@dataclass
class PilotTone:
    """A CW emitter pinned to the SDR centre frequency (calibration reference)."""

    azimuth_deg: float
    elevation_deg: float
    power_dbm: float = 0.0
    phase_rad: float = 0.0


@dataclass
class NoiseEmitter:
    """A wideband AWGN interferer at a fixed bearing.

    Attributes:
        azimuth_deg: Bearing azimuth.
        elevation_deg: Bearing elevation.
        power_dbm: Interferer power in dBm.
        bandwidth_hz: Reserved for future use (Stage 3 filtering).
    """

    azimuth_deg: float
    elevation_deg: float
    power_dbm: float = 0.0
    bandwidth_hz: float | None = None


Emitter = CWEmitter | PilotTone | NoiseEmitter


@dataclass
class MockSdrScenario:
    """Bundles emitters + global noise so tests can compose scenarios concisely."""

    emitters: list[Emitter] = field(default_factory=list)
    snr_db: float = 20.0


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def _direction_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert ``(az, el)`` in degrees to a unit vector in the rotator-local frame."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )


class MockSdr:
    """Synthesises array IQ from a configurable :class:`MockSdrScenario`.

    Args:
        geometry: A :class:`rfdf.hal.GeometryController` whose ``positions()``
            yields the antenna locations (m).
        scenario: Initial emitter scenario; mutable post-construction via
            ``add_emitter`` / ``clear_emitters``.
        block_samples: Samples per :class:`StreamBlock` emitted by ``stream``.
        mock_b210_behavior: When True, randomise per-channel phase on every
            ``configure()`` call. Default False.
        gain_errors_db: Optional per-channel gain offsets in dB.
        mutual_coupling: Optional ``(M, M)`` complex mutual-coupling matrix.
        seed: RNG seed for deterministic noise + B210-phase simulation.
    """

    def __init__(
        self,
        *,
        geometry: GeometryController,
        scenario: MockSdrScenario | None = None,
        block_samples: int = 4096,
        mock_b210_behavior: bool = False,
        gain_errors_db: np.ndarray | None = None,
        mutual_coupling: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        """Capture configuration + reset state."""
        self._geometry = geometry
        self._scenario = scenario or MockSdrScenario()
        self._block_samples = int(block_samples)
        self._mock_b210 = bool(mock_b210_behavior)
        self._gain_errors_db = gain_errors_db
        self._mutual_coupling = mutual_coupling
        self._rng = np.random.default_rng(seed)
        self._config: SdrConfig | None = None
        self._positions: np.ndarray | None = None
        self._channel_phase: np.ndarray | None = None
        self._sequence = 0
        self._running = False
        self._start_time_s = 0.0

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        """Channel count tracks the configured geometry."""
        return self._geometry.num_antennas

    @property
    def supports_coherent(self) -> bool:
        """Mock is intrinsically coherent — all channels share a single clock."""
        return True

    @property
    def tuning_range_hz(self) -> tuple[float, float]:
        """Reference B210-style range; mock supports a wide span."""
        return (70e6, 6e9)

    @property
    def max_sample_rate_hz(self) -> float:
        """Reference B210-style max rate."""
        return 56e6

    # ------------------------------------------------------------------
    # Scenario management
    # ------------------------------------------------------------------

    def add_emitter(self, emitter: Emitter) -> None:
        """Append ``emitter`` to the active scenario."""
        self._scenario.emitters.append(emitter)

    def clear_emitters(self) -> None:
        """Remove all emitters."""
        self._scenario.emitters.clear()

    def set_snr_db(self, snr_db: float) -> None:
        """Override the per-channel signal-to-noise ratio."""
        self._scenario.snr_db = float(snr_db)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def configure(self, config: SdrConfig) -> None:
        """Apply tuning + sampling configuration; refresh cached geometry + phase."""
        self._config = config
        self._positions = await self._geometry.positions()
        if self._mock_b210:
            # Simulate fractional-N PLL retune: new random per-channel phase.
            self._channel_phase = self._rng.uniform(
                low=-math.pi, high=math.pi, size=self.num_channels
            )
        else:
            self._channel_phase = np.zeros(self.num_channels, dtype=np.float64)

    async def start(self) -> None:
        """Begin streaming; idempotent."""
        if self._config is None:
            raise RuntimeError("MockSdr: call configure() before start()")
        self._running = True
        self._start_time_s = time.monotonic()

    async def stop(self) -> None:
        """Stop streaming; subsequent stream() calls raise until restart."""
        self._running = False

    async def stream(self) -> AsyncIterator[StreamBlock]:
        """Yield :class:`StreamBlock` objects of ``block_samples`` per call."""
        if self._config is None:
            raise RuntimeError("MockSdr: call configure() before stream()")
        if not self._running:
            raise RuntimeError("MockSdr: call start() before stream()")
        try:
            while self._running:
                iq = self._synthesise_block(self._block_samples)
                block = StreamBlock(
                    iq=iq,
                    sample_rate_hz=self._config.sample_rate_hz,
                    center_freq_hz=self._config.center_freq_hz,
                    start_time_s=self._sequence * self._block_samples / self._config.sample_rate_hz,
                    sequence_number=self._sequence,
                    metadata={"backend": "mock", "snr_db": self._scenario.snr_db},
                )
                self._sequence += 1
                yield block
                # Yield control so async cancellation works promptly.
                await asyncio.sleep(0)
        finally:
            self._running = False

    async def capture(self, duration_s: float) -> Recording:
        """Capture ``duration_s`` of IQ and write a single-channel SigMF pair."""
        if self._config is None:
            raise RuntimeError("MockSdr: call configure() before capture()")
        num_samples = round(duration_s * self._config.sample_rate_hz)
        iq = self._synthesise_block(num_samples)

        # The Stage 2 file-replay backend supports single-channel only; export
        # the first channel for callers that round-trip captures through replay.
        first_channel = iq[0]
        capture_dir = Path.cwd() / ".rfdf-captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        stem = f"mock-{int(time.time() * 1000)}"
        data_path = capture_dir / f"{stem}.sigmf-data"
        meta_path = capture_dir / f"{stem}.sigmf-meta"

        # SigMF: interleaved I/Q float32 little-endian.
        first_channel.astype(np.complex64).tofile(data_path)
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self._config.sample_rate_hz,
                "core:version": "1.0.0",
                "core:num_channels": 1,
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": self._config.center_freq_hz,
                }
            ],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return Recording(
            sigmf_meta_path=meta_path,
            sigmf_data_path=data_path,
            duration_s=duration_s,
            num_samples=num_samples,
            channels=1,
            sample_rate_hz=self._config.sample_rate_hz,
            center_freq_hz=self._config.center_freq_hz,
            metadata=meta,
        )

    async def status(self) -> dict[str, object]:
        """Report mock device health (always 'reachable' — it is synthetic)."""
        return {
            "backend": "mock",
            "reachable": True,
            "streaming": self._running,
            "num_channels": self.num_channels,
            "configured": self._config is not None,
            "coherent": self.supports_coherent,
            "mock_b210_behavior": self._mock_b210,
            "num_emitters": len(self._scenario.emitters),
        }

    @requires_eirp_check
    async def calibration_pilot(self, freq_hz: float, power_dbm: float) -> None:
        """Add a :class:`PilotTone` emitter; EIRP cap enforced by the decorator."""
        # Pilot tone is added at zenith (90 deg elevation) — boresight by convention.
        self.add_emitter(PilotTone(azimuth_deg=0.0, elevation_deg=90.0, power_dbm=power_dbm))
        # ``freq_hz`` is consumed via the configured centre frequency; we keep the
        # signature for HAL conformance + future per-tone retuning.
        _ = freq_hz

    async def close(self) -> None:
        """Release backend resources (no-op for the mock)."""
        await self.stop()

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _synthesise_block(self, num_samples: int) -> np.ndarray:
        """Generate one ``(M, N)`` block of complex64 IQ."""
        assert self._config is not None
        assert self._positions is not None
        assert self._channel_phase is not None
        sample_rate = self._config.sample_rate_hz
        center_freq = self._config.center_freq_hz
        wavelength = C0 / center_freq
        t = np.arange(num_samples, dtype=np.float64) / sample_rate

        num_channels = self.num_channels
        signal = np.zeros((num_channels, num_samples), dtype=np.complex128)

        for emitter in self._scenario.emitters:
            theta_hat = _direction_unit_vector(
                emitter.azimuth_deg,
                emitter.elevation_deg,
            )
            # Steering vector: shape (M,).
            steering = np.exp(-1j * (2.0 * math.pi / wavelength) * (self._positions @ theta_hat))
            if isinstance(emitter, CWEmitter):
                amp = _dbm_to_amplitude(emitter.power_dbm)
                source = amp * np.exp(
                    1j * (2.0 * math.pi * emitter.freq_offset_hz * t + emitter.phase_rad)
                )
                signal += np.outer(steering, source)
            elif isinstance(emitter, PilotTone):
                amp = _dbm_to_amplitude(emitter.power_dbm)
                source = amp * np.exp(1j * emitter.phase_rad) * np.ones(num_samples)
                signal += np.outer(steering, source)
            elif isinstance(emitter, NoiseEmitter):
                amp = _dbm_to_amplitude(emitter.power_dbm)
                source = (
                    amp
                    * (self._rng.normal(size=num_samples) + 1j * self._rng.normal(size=num_samples))
                    / math.sqrt(2.0)
                )
                signal += np.outer(steering, source)
            # NoiseEmitter.bandwidth_hz reserved for Stage 3 spectral shaping.

        # Mutual coupling (optional).
        if self._mutual_coupling is not None:
            signal = self._mutual_coupling @ signal

        # Per-channel B210-style phase offset.
        signal = signal * np.exp(1j * self._channel_phase)[:, None]

        # Per-channel gain calibration error.
        if self._gain_errors_db is not None:
            gain_lin = 10.0 ** (np.asarray(self._gain_errors_db, dtype=np.float64) / 20.0)
            signal = signal * gain_lin[:, None]

        # Additive Gaussian noise at the configured SNR.
        signal_power = float(np.mean(np.abs(signal) ** 2)) if num_samples > 0 else 0.0
        if signal_power > 0.0:
            noise_power = signal_power / (10.0 ** (self._scenario.snr_db / 10.0))
        else:
            noise_power = 1.0  # noise-only block — keep a sensible scale.
        noise_std = math.sqrt(noise_power / 2.0)
        noise = noise_std * (
            self._rng.normal(size=(num_channels, num_samples))
            + 1j * self._rng.normal(size=(num_channels, num_samples))
        )
        return (signal + noise).astype(np.complex64)


def create(
    *,
    geometry: GeometryController | None = None,
    geometry_antennas: Sequence[Sequence[float]] | None = None,
    scenario: MockSdrScenario | None = None,
    block_samples: int = 4096,
    mock_b210_behavior: bool = False,
    gain_errors_db: Sequence[float] | None = None,
    mutual_coupling: np.ndarray | None = None,
    seed: int = 0,
    **_: Any,
) -> MockSdr:
    """Factory wired into the ``rfdf.backends.sdr.mock`` entry-point.

    Args:
        geometry: Pre-built :class:`GeometryController`; takes precedence.
        geometry_antennas: Antenna positions used to construct a ``static``
            geometry when ``geometry`` is None. Defaults to the pentagonal
            5-antenna reference array.
        scenario: Initial scenario; defaults to empty (caller adds emitters).
        block_samples: Samples per StreamBlock.
        mock_b210_behavior: Enable B210 fractional-N phase randomisation.
        gain_errors_db: Per-channel gain offsets in dB.
        mutual_coupling: ``(M, M)`` complex coupling matrix.
        seed: RNG seed.

    Returns:
        Configured :class:`MockSdr`.
    """
    if geometry is None:
        geometry = create_static_geometry(antennas=geometry_antennas)
    return MockSdr(
        geometry=geometry,
        scenario=scenario,
        block_samples=block_samples,
        mock_b210_behavior=mock_b210_behavior,
        gain_errors_db=(
            np.asarray(gain_errors_db, dtype=np.float64) if gain_errors_db is not None else None
        ),
        mutual_coupling=mutual_coupling,
        seed=seed,
    )

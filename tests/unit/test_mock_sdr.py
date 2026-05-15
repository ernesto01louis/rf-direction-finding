"""Unit tests for the mock SDR backend.

Covers the array-factor signal model (math via MUSIC sanity check), the
optional fidelity branches (mutual coupling, B210 phase, gain errors), the
EIRP-guarded ``calibration_pilot`` path, and the capture → SigMF artifact
flow. Property-based contract tests live in ``tests/contracts/``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from rfdf.backends.sdr.mock import (
    CWEmitter,
    MockSdrScenario,
    PilotTone,
    create,
)
from rfdf.core.eirp import EirpCapExceededError
from rfdf.hal import SdrConfig, SdrSource


def _basic_config(*, channels: int = 5) -> SdrConfig:
    return SdrConfig(
        center_freq_hz=868e6,
        sample_rate_hz=2e6,
        rx_gain_db=30.0,
        channels=list(range(channels)),
    )


def test_protocol_conformance() -> None:
    """MockSdr is a structural SdrSource."""
    sdr = create()
    assert isinstance(sdr, SdrSource)


def test_capabilities_match_geometry() -> None:
    """num_channels follows the underlying GeometryController."""
    sdr = create()
    assert sdr.num_channels == 5  # default pentagonal array
    assert sdr.supports_coherent is True
    low, high = sdr.tuning_range_hz
    assert low < high
    assert sdr.max_sample_rate_hz > 0


def test_capture_writes_sigmf_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """capture(0.05s) produces a sigmf-meta + sigmf-data pair."""
    monkeypatch.chdir(tmp_path)
    sdr = create()
    asyncio.run(sdr.configure(_basic_config()))
    sdr.add_emitter(CWEmitter(azimuth_deg=30.0, elevation_deg=0.0, power_dbm=0.0))
    recording = asyncio.run(sdr.capture(0.05))
    assert recording.sigmf_meta_path.exists()
    assert recording.sigmf_data_path.exists()
    assert recording.duration_s == pytest.approx(0.05)
    assert recording.channels == 1
    assert recording.sample_rate_hz == pytest.approx(2e6)


def test_stream_yields_correct_shape_and_dtype() -> None:
    """stream() blocks have shape (M, block_samples), dtype complex64."""
    sdr = create(block_samples=512)
    asyncio.run(sdr.configure(_basic_config()))
    asyncio.run(sdr.start())

    async def first_two() -> list[tuple[tuple[int, ...], np.dtype]]:  # type: ignore[type-arg]
        out: list[tuple[tuple[int, ...], np.dtype]] = []  # type: ignore[type-arg]
        i = 0
        async for block in sdr.stream():
            out.append((block.iq.shape, block.iq.dtype))
            i += 1
            if i >= 2:
                await sdr.stop()
        return out

    shapes = asyncio.run(first_two())
    assert len(shapes) == 2
    for shape, dtype in shapes:
        assert shape == (5, 512)
        assert dtype == np.complex64


def test_stream_sequence_numbers_are_monotonic() -> None:
    """Sequence numbers increment by 1 across consecutive blocks."""
    sdr = create(block_samples=256)
    asyncio.run(sdr.configure(_basic_config()))
    asyncio.run(sdr.start())

    async def collect() -> list[int]:
        seqs: list[int] = []
        async for block in sdr.stream():
            seqs.append(block.sequence_number)
            if len(seqs) >= 4:
                await sdr.stop()
        return seqs

    seqs = asyncio.run(collect())
    assert seqs == [0, 1, 2, 3]


def test_music_peaks_at_emitter_bearing() -> None:
    """The mock SDR's IQ satisfies the standard MUSIC sanity check.

    Configure a single CW emitter at a known bearing, run classical MUSIC on the
    sample covariance, and assert the pseudo-spectrum peak is within 2° of the
    true bearing. This is the cheapest end-to-end verification that the
    array-factor signal model is implemented correctly.
    """
    true_az = 35.0
    scenario = MockSdrScenario(
        emitters=[CWEmitter(azimuth_deg=true_az, elevation_deg=0.0, power_dbm=0.0)],
        snr_db=30.0,
    )
    sdr = create(scenario=scenario, block_samples=4096, seed=1)
    cfg = _basic_config()
    asyncio.run(sdr.configure(cfg))
    asyncio.run(sdr.start())

    async def grab_one() -> np.ndarray:
        async for block in sdr.stream():
            await sdr.stop()
            return block.iq.astype(np.complex128)
        raise AssertionError("no block")

    iq = asyncio.run(grab_one())

    # Sample covariance R = (1/N) X X^H.
    cov = (iq @ iq.conj().T) / iq.shape[1]
    eigvals, eigvecs = np.linalg.eigh(cov)
    # K=1 signal, M-K=4 noise eigenvectors (the smallest eigenvalues).
    noise_space = eigvecs[:, :-1]

    # Steering vector helper (matching the mock's convention).
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.17, 0.0, 0.0],
            [0.0, 0.17, 0.0],
            [-0.17, 0.0, 0.0],
            [0.0, -0.17, 0.0],
        ]
    )
    wavelength = 299_792_458.0 / cfg.center_freq_hz

    grid_az = np.arange(-180.0, 180.0, 0.5)
    spectrum = np.zeros_like(grid_az)
    for i, az_deg in enumerate(grid_az):
        az = np.deg2rad(az_deg)
        theta_hat = np.array([np.cos(az), np.sin(az), 0.0])
        a = np.exp(-1j * (2.0 * np.pi / wavelength) * (positions @ theta_hat))
        proj = a.conj() @ noise_space @ noise_space.conj().T @ a
        spectrum[i] = 1.0 / max(np.real(proj), 1e-12)

    peak_az = grid_az[int(np.argmax(spectrum))]
    assert abs(peak_az - true_az) <= 2.0, f"peak at {peak_az}, expected ~{true_az}"
    _ = eigvals  # suppress unused-variable lint when reading the eig spectrum


def test_b210_behavior_randomizes_phase_per_configure() -> None:
    """With mock_b210_behavior=True, channel phase changes on each configure."""
    sdr = create(mock_b210_behavior=True, seed=7)
    asyncio.run(sdr.configure(_basic_config()))
    first = sdr._channel_phase.copy()  # type: ignore[union-attr]
    asyncio.run(sdr.configure(_basic_config()))
    second = sdr._channel_phase  # type: ignore[union-attr]
    assert second is not None
    assert not np.allclose(first, second)


def test_b210_behavior_off_by_default() -> None:
    """Without the flag, channel phase is zero across all channels."""
    sdr = create()
    asyncio.run(sdr.configure(_basic_config()))
    assert sdr._channel_phase is not None  # type: ignore[union-attr]
    np.testing.assert_allclose(sdr._channel_phase, 0.0)  # type: ignore[arg-type]


def test_mutual_coupling_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-identity coupling matrix changes the IQ on a known emitter."""
    coupling = np.eye(5, dtype=np.complex128)
    coupling[0, 1] = 0.3  # leak from antenna 1 into 0
    sdr_a = create(seed=0)
    sdr_b = create(seed=0, mutual_coupling=coupling)
    cfg = _basic_config()
    for sdr in (sdr_a, sdr_b):
        asyncio.run(sdr.configure(cfg))
        sdr.add_emitter(CWEmitter(azimuth_deg=10.0, elevation_deg=0.0, power_dbm=0.0))

    async def one_block(s: object) -> np.ndarray:
        await s.start()  # type: ignore[attr-defined]
        async for block in s.stream():  # type: ignore[attr-defined]
            await s.stop()  # type: ignore[attr-defined]
            return block.iq.astype(np.complex128)
        raise AssertionError("no block")

    iq_a = asyncio.run(one_block(sdr_a))
    iq_b = asyncio.run(one_block(sdr_b))
    assert not np.allclose(iq_a, iq_b)


def test_gain_errors_applied() -> None:
    """A +6 dB gain on channel 0 roughly doubles its amplitude vs the rest."""
    sdr = create(seed=0, gain_errors_db=[6.0, 0.0, 0.0, 0.0, 0.0])
    cfg = _basic_config()
    asyncio.run(sdr.configure(cfg))
    sdr.add_emitter(CWEmitter(azimuth_deg=0.0, elevation_deg=90.0, power_dbm=0.0))
    sdr.set_snr_db(40.0)

    async def grab() -> np.ndarray:
        await sdr.start()
        async for block in sdr.stream():
            await sdr.stop()
            return block.iq.astype(np.complex128)
        raise AssertionError("no block")

    iq = asyncio.run(grab())
    rms = np.sqrt(np.mean(np.abs(iq) ** 2, axis=1))
    ratio = rms[0] / rms[1]
    # +6 dB means voltage x2; expect ratio in (1.5, 2.5) with noise.
    assert 1.5 <= ratio <= 2.5


def test_calibration_pilot_blocked_above_eirp_cap() -> None:
    """Decorator blocks calibration_pilot when power_dbm exceeds the default cap."""
    # Default cap from RfdfConfig is 14 dBm; request 30 dBm => raises.
    for key in list(os.environ.keys()):
        if key.startswith("RFDF_"):
            del os.environ[key]
    sdr = create()
    asyncio.run(sdr.configure(_basic_config()))
    with pytest.raises(EirpCapExceededError):
        asyncio.run(sdr.calibration_pilot(freq_hz=868e6, power_dbm=30.0))


def test_calibration_pilot_under_cap_adds_emitter() -> None:
    """Under the cap, calibration_pilot adds a PilotTone to the scenario."""
    sdr = create()
    asyncio.run(sdr.configure(_basic_config()))
    asyncio.run(sdr.calibration_pilot(freq_hz=868e6, power_dbm=0.0))
    assert any(isinstance(e, PilotTone) for e in sdr._scenario.emitters)  # type: ignore[union-attr]


def test_stream_without_start_raises() -> None:
    """stream() before start() raises a clear RuntimeError."""
    sdr = create()
    asyncio.run(sdr.configure(_basic_config()))

    async def go() -> None:
        async for _ in sdr.stream():
            return

    with pytest.raises(RuntimeError, match="start"):
        asyncio.run(go())


def test_capture_without_configure_raises() -> None:
    """capture() before configure() raises a clear RuntimeError."""
    sdr = create()
    with pytest.raises(RuntimeError, match="configure"):
        asyncio.run(sdr.capture(0.01))

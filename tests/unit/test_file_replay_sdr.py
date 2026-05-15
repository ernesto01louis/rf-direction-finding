"""Unit tests for the SigMF file-replay SDR backend."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from rfdf.backends.sdr.file_replay import create as create_replay
from rfdf.hal import SdrConfig, SdrSource


def _config_for_recording(meta_path: Path) -> SdrConfig:
    meta = json.loads(meta_path.read_text())
    sample_rate = float(meta["global"]["core:sample_rate"])
    center_freq = float(meta["captures"][0]["core:frequency"])
    return SdrConfig(center_freq_hz=center_freq, sample_rate_hz=sample_rate)


def test_protocol_conformance(tiny_sigmf: tuple[Path, Path]) -> None:
    """The replay backend is a structural SdrSource."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path)
    assert isinstance(sdr, SdrSource)


def test_capabilities_derive_from_recording(tiny_sigmf: tuple[Path, Path]) -> None:
    """num_channels=1, tuning_range pinned to the recording's centre freq."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path)
    assert sdr.num_channels == 1
    low, high = sdr.tuning_range_hz
    assert low == high
    assert sdr.max_sample_rate_hz == pytest.approx(2_560.0)


def test_stream_yields_correct_shape(tiny_sigmf: tuple[Path, Path]) -> None:
    """Stream blocks have shape (1, block_samples) and dtype complex64."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path, block_samples=64)
    asyncio.run(sdr.configure(_config_for_recording(meta_path)))
    asyncio.run(sdr.start())

    async def first_block() -> np.ndarray:
        async for block in sdr.stream():
            await sdr.stop()
            return block.iq
        raise AssertionError("no block")

    iq = asyncio.run(first_block())
    assert iq.shape == (1, 64)
    assert iq.dtype == np.complex64


def test_stream_iterates_full_recording(tiny_sigmf: tuple[Path, Path]) -> None:
    """Without looping, the stream terminates at EOF."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path, block_samples=64, loop=False)
    asyncio.run(sdr.configure(_config_for_recording(meta_path)))
    asyncio.run(sdr.start())

    async def collect() -> int:
        count = 0
        async for _ in sdr.stream():
            count += 1
            if count >= 100:  # safety guard
                break
        return count

    count = asyncio.run(collect())
    # 256 samples / 64 per block = 4 blocks before EOF.
    assert count == 4


def test_loop_replays_from_start(tiny_sigmf: tuple[Path, Path]) -> None:
    """With loop=True, the stream wraps around and never EOFs."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path, block_samples=64, loop=True)
    asyncio.run(sdr.configure(_config_for_recording(meta_path)))
    asyncio.run(sdr.start())

    async def collect() -> int:
        count = 0
        async for _ in sdr.stream():
            count += 1
            if count >= 10:
                await sdr.stop()
        return count

    assert asyncio.run(collect()) == 10


def test_seek_resets_cursor(tiny_sigmf: tuple[Path, Path]) -> None:
    """seek(time_s) moves the cursor; next block starts at the new offset."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path, block_samples=64)
    asyncio.run(sdr.configure(_config_for_recording(meta_path)))
    sdr.seek(0.05)  # halfway in
    asyncio.run(sdr.start())

    async def first_block() -> np.ndarray:
        async for block in sdr.stream():
            await sdr.stop()
            return block.iq
        raise AssertionError("no block")

    iq = asyncio.run(first_block())
    assert iq.shape == (1, 64)


def test_seek_beyond_eof_raises(tiny_sigmf: tuple[Path, Path]) -> None:
    """seek past the recording's duration raises ValueError."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path)
    with pytest.raises(ValueError, match="exceeds recording"):
        sdr.seek(10.0)


def test_inject_snr_changes_samples(tiny_sigmf: tuple[Path, Path]) -> None:
    """Injecting AWGN modifies the IQ relative to clean replay."""
    meta_path, _ = tiny_sigmf
    sdr_clean = create_replay(meta_path=meta_path, block_samples=64, seed=0)
    sdr_noisy = create_replay(meta_path=meta_path, block_samples=64, inject_snr_db=0.0, seed=0)
    for sdr in (sdr_clean, sdr_noisy):
        asyncio.run(sdr.configure(_config_for_recording(meta_path)))
        asyncio.run(sdr.start())

    async def grab(s: object) -> np.ndarray:
        async for block in s.stream():  # type: ignore[attr-defined]
            await s.stop()  # type: ignore[attr-defined]
            return block.iq
        raise AssertionError("no block")

    iq_clean = asyncio.run(grab(sdr_clean))
    iq_noisy = asyncio.run(grab(sdr_noisy))
    assert not np.allclose(iq_clean, iq_noisy)


def test_calibration_pilot_raises(tiny_sigmf: tuple[Path, Path]) -> None:
    """File-replay is RX-only; calibration_pilot raises NotImplementedError."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path)
    with pytest.raises(NotImplementedError, match="RX only"):
        asyncio.run(sdr.calibration_pilot(freq_hz=868e6, power_dbm=0.0))


def test_multi_channel_meta_rejected(tmp_path: Path) -> None:
    """Multi-channel SigMF raises NotImplementedError (Stage 2 limitation)."""
    meta_path = tmp_path / "multi.sigmf-meta"
    data_path = tmp_path / "multi.sigmf-data"
    np.zeros(128, dtype=np.complex64).tofile(data_path)
    meta_path.write_text(
        json.dumps(
            {
                "global": {
                    "core:datatype": "cf32_le",
                    "core:sample_rate": 2.0e6,
                    "core:version": "1.0.0",
                    "core:num_channels": 4,
                },
                "captures": [{"core:sample_start": 0, "core:frequency": 868e6}],
                "annotations": [],
            }
        )
    )
    with pytest.raises(NotImplementedError, match="multi-channel"):
        create_replay(meta_path=meta_path)


def test_missing_meta_file_raises(tmp_path: Path) -> None:
    """A non-existent meta_path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="meta path"):
        create_replay(meta_path=tmp_path / "missing.sigmf-meta")


def test_unsupported_datatype_raises(tmp_path: Path) -> None:
    """An exotic SigMF datatype raises NotImplementedError."""
    meta_path = tmp_path / "weird.sigmf-meta"
    data_path = tmp_path / "weird.sigmf-data"
    data_path.write_bytes(b"\x00" * 8)
    meta_path.write_text(
        json.dumps(
            {
                "global": {
                    "core:datatype": "ru32_be",  # not supported
                    "core:sample_rate": 1e6,
                    "core:version": "1.0.0",
                    "core:num_channels": 1,
                },
                "captures": [{"core:sample_start": 0, "core:frequency": 868e6}],
                "annotations": [],
            }
        )
    )
    with pytest.raises(NotImplementedError, match="datatype"):
        create_replay(meta_path=meta_path)


def test_capture_slices_recording(tiny_sigmf: tuple[Path, Path]) -> None:
    """capture(duration_s) writes a new SigMF pair sibling to the source."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path)
    asyncio.run(sdr.configure(_config_for_recording(meta_path)))
    recording = asyncio.run(sdr.capture(0.05))  # 128 samples
    assert recording.sigmf_meta_path.exists()
    assert recording.sigmf_data_path.exists()
    assert recording.num_samples == 128
    assert recording.duration_s == pytest.approx(0.05)


def test_configure_sample_rate_mismatch_raises(tiny_sigmf: tuple[Path, Path]) -> None:
    """A requested sample rate that disagrees with the recording raises."""
    meta_path, _ = tiny_sigmf
    sdr = create_replay(meta_path=meta_path)
    bad = SdrConfig(center_freq_hz=868e6, sample_rate_hz=1e6)  # not 2560
    with pytest.raises(ValueError, match="sample_rate"):
        asyncio.run(sdr.configure(bad))

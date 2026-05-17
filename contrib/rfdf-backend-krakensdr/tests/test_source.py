"""Unit tests for the KrakenSDR contrib backend (no hardware, no Heimdall).

A fake :class:`HeimdallInterface` stands in for the Heimdall DAQ daemon so the
backend logic is fully exercised without a KrakenSDR or a running daemon.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from rfdf_backend_krakensdr.heimdall import KRAKEN_CHANNELS
from rfdf_backend_krakensdr.source import KrakenSdrError, KrakenSdrSource, create

from rfdf.hal import SdrConfig, SdrSource


class FakeHeimdall:
    """In-memory stand-in for the Heimdall DAQ — serves synthetic coherent frames."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    @property
    def num_channels(self) -> int:
        """KrakenSDR coherent channel count."""
        return KRAKEN_CHANNELS

    def connect(self, *, center_freq_hz: float, sample_rate_hz: float, gain_db: float) -> None:
        """Record that the (fake) daemon link was opened."""
        _ = (center_freq_hz, sample_rate_hz, gain_db)
        self.connected = True

    def read_frame(self, num_samples: int) -> np.ndarray:
        """Return a synthetic coherent IQ frame."""
        rng = np.random.default_rng(0)
        real = rng.standard_normal((KRAKEN_CHANNELS, num_samples))
        imag = rng.standard_normal((KRAKEN_CHANNELS, num_samples))
        return (real + 1j * imag).astype(np.complex64)

    def close(self) -> None:
        """Record that the (fake) daemon link was released."""
        self.closed = True


def test_capabilities() -> None:
    """A constructed KrakenSDR backend reports a 5-channel coherent receiver."""
    sdr = create(heimdall=FakeHeimdall())
    assert sdr.num_channels == 5
    assert sdr.supports_coherent is True
    assert sdr.tuning_range_hz == (24e6, 1.8e9)


def test_krakensdr_is_structural_sdr_source() -> None:
    """KrakenSdrSource structurally satisfies the SdrSource Protocol."""
    assert isinstance(create(heimdall=FakeHeimdall()), SdrSource)


def test_configure_rejects_out_of_range_frequency() -> None:
    """A centre frequency above the coherent ceiling is rejected."""
    sdr = KrakenSdrSource(heimdall=FakeHeimdall())

    async def run() -> None:
        with pytest.raises(KrakenSdrError, match="coherent tuning range"):
            await sdr.configure(SdrConfig(center_freq_hz=2.4e9, sample_rate_hz=2e6))

    asyncio.run(run())


def test_stream_yields_coherent_blocks() -> None:
    """stream() yields 5-channel complex64 blocks via the Heimdall interface."""
    fake = FakeHeimdall()
    sdr = KrakenSdrSource(heimdall=fake, block_samples=128)

    async def run() -> None:
        await sdr.configure(SdrConfig(center_freq_hz=433e6, sample_rate_hz=2e6))
        assert fake.connected is True
        await sdr.start()
        collected = 0
        async for block in sdr.stream():
            assert block.iq.shape == (5, 128)
            assert block.iq.dtype == np.complex64
            collected += 1
            if collected >= 3:
                await sdr.stop()
                break
        await sdr.close()
        assert fake.closed is True

    asyncio.run(run())


def test_capture_writes_multichannel_sigmf(tmp_path, monkeypatch) -> None:
    """capture() writes a 5-channel SigMF pair."""
    monkeypatch.chdir(tmp_path)
    sdr = create(heimdall=FakeHeimdall())

    async def run() -> None:
        await sdr.configure(SdrConfig(center_freq_hz=433e6, sample_rate_hz=2e6))
        recording = await sdr.capture(0.001)
        assert recording.channels == 5
        assert recording.sigmf_meta_path.is_file()
        assert recording.sigmf_data_path.is_file()

    asyncio.run(run())


def test_calibration_pilot_is_rx_only() -> None:
    """KrakenSDR cannot transmit — calibration_pilot raises."""

    async def run() -> None:
        with pytest.raises(NotImplementedError, match="RX only"):
            await create(heimdall=FakeHeimdall()).calibration_pilot(433e6, 0.0)

    asyncio.run(run())

"""DOA on a mock array — the canonical "is the platform working?" demo.

Configures a mock SDR with an 8-element uniform linear array and three emitters,
captures synthetic IQ with no hardware, runs MUSIC and ESPRIT, and reports the
recovered bearings against the Cramer-Rao bound. Runs end-to-end in well under
30 seconds.

Run it from the repository root::

    python examples/01-doa-on-mock-array/demo.py
"""

from __future__ import annotations

import asyncio

import numpy as np

from rfdf.backends.geometry.static import create as create_static_geometry
from rfdf.backends.sdr.mock import CWEmitter, MockSdrScenario
from rfdf.backends.sdr.mock import create as create_mock_sdr
from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.crlb import compute_crlb
from rfdf.dsp.doa.esprit import esprit
from rfdf.dsp.doa.music import music
from rfdf.dsp.geometry_presets import half_wavelength_spacing, ula
from rfdf.dsp.steering import build_manifold
from rfdf.hal import SdrConfig


def main() -> None:
    """Run the mock-array DOA demo and print the recovered bearings."""
    freq_hz = 868e6
    truths = (35.0, 80.0, 125.0)
    channels = 8
    positions = ula(channels, half_wavelength_spacing(freq_hz))
    geometry = create_static_geometry(antennas=positions.tolist())
    emitters = [
        CWEmitter(azimuth_deg=az, elevation_deg=0.0, power_dbm=0.0, freq_offset_hz=7e3 * index)
        for index, az in enumerate(truths)
    ]
    sdr = create_mock_sdr(
        geometry=geometry,
        scenario=MockSdrScenario(emitters=list(emitters), snr_db=20.0),
        block_samples=8192,
    )
    asyncio.run(
        sdr.configure(
            SdrConfig(center_freq_hz=freq_hz, sample_rate_hz=2e6, channels=list(range(channels)))
        )
    )

    async def _capture() -> np.ndarray:  # type: ignore[type-arg]
        await sdr.start()
        async for block in sdr.stream():
            await sdr.stop()
            return block.iq.astype(np.complex128)
        raise RuntimeError("the mock SDR produced no IQ")

    iq = asyncio.run(_capture())
    covariance = sample_covariance(iq)

    manifold = build_manifold(positions, np.arange(0.0, 180.0, 0.25), np.array([0.0]), freq_hz)
    music_estimate = music(covariance, manifold, num_signals=3)
    esprit_estimate = esprit(covariance, positions=positions, freq_hz=freq_hz, num_signals=3)

    print(f"simulated 3 emitters at azimuths {list(truths)} deg (8-element ULA, 20 dB SNR)")
    print(f"  MUSIC  recovered: {[round(az, 2) for az in sorted(music_estimate.azimuth_deg)]}")
    print(f"  ESPRIT recovered: {[round(az, 2) for az in sorted(esprit_estimate.azimuth_deg)]}")
    for truth in truths:
        crlb = compute_crlb(
            positions,
            freq_hz=freq_hz,
            snr_db=20.0,
            snapshots=iq.shape[1],
            direction_deg=truth,
        )
        print(f"  source at {truth:6.1f} deg -> Cramer-Rao std {np.sqrt(crlb):.4f} deg")

    for name, estimate in (("MUSIC", music_estimate), ("ESPRIT", esprit_estimate)):
        for truth, recovered in zip(truths, sorted(estimate.azimuth_deg), strict=True):
            error = abs(recovered - truth)
            if error >= 1.0:
                raise AssertionError(f"{name} recovered {recovered} for truth {truth}")
    print("demo: DOA pipeline PASS")


if __name__ == "__main__":
    main()

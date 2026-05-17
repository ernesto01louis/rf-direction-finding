"""Coherent capture + DOA on a real USRP B210 array.

=============================================================================
  HARDWARE REQUIRED. This example does NOT run on the demo-no-hardware CI
  gate. It needs:
    * one or more Ettus USRP B210s (serials in RFDF_B210_SERIALS, comma-sep)
    * an OctoClock-G (or equivalent) distributing 10 MHz + 1 PPS, if >1 B210
    * the [sdr-uhd] extra installed plus the system UHD driver
  With no B210 configured the script prints this banner and exits 0.
=============================================================================

What it does:

1. Configure the B210 backend with the configured physical units.
2. Verify reference-clock + time lock on every board (fatal if not locked).
3. Tune to 868 MHz and capture 5 seconds of coherent IQ.
4. Run a MUSIC DOA estimate on a real ambient signal (a LoRa gateway, a
   weather-satellite downlink, ...) — receive only, no transmit.
5. Report the recovered bearing.

Run it from the repository root::

    RFDF_B210_SERIALS=31D5A3B,31D5A40 python examples/05-real-b210-coherent-capture/demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np

from rfdf.backends.geometry.static import create as create_static_geometry
from rfdf.hal import SdrConfig

_CENTER_FREQ_HZ = 868e6
_SAMPLE_RATE_HZ = 2e6
_CAPTURE_S = 5.0


def _serials() -> list[str]:
    """Read B210 serials from the environment; empty when no hardware."""
    raw = os.environ.get("RFDF_B210_SERIALS", "").strip()
    return [s.strip() for s in raw.split(",") if s.strip()]


async def _run(serials: list[str]) -> None:
    """Configure the B210, capture coherent IQ, and run a DOA estimate."""
    from rfdf.backends.sdr.b210 import create as create_b210

    # A planar 5-element reference array — replace with your measured geometry.
    geometry = create_static_geometry()
    sdr = create_b210(serial_numbers=serials, geometry=geometry)

    print(f"configuring {len(serials)} B210(s): {', '.join(serials)} ...")
    await sdr.configure(SdrConfig(center_freq_hz=_CENTER_FREQ_HZ, sample_rate_hz=_SAMPLE_RATE_HZ))

    status = await sdr.status()
    print(f"  ref_locked={status.get('ref_locked')} calibrated={status.get('calibrated')}")
    if not status.get("ref_locked"):
        raise SystemExit("B210 reference clock is not locked — check the OctoClock wiring.")

    print(f"capturing {_CAPTURE_S:.0f} s of coherent IQ at {_CENTER_FREQ_HZ / 1e6:.0f} MHz ...")
    recording = await sdr.capture(_CAPTURE_S)
    print(
        f"  wrote {recording.num_samples} samples x {recording.channels} ch -> "
        f"{recording.sigmf_data_path.name}"
    )

    # Covariance + MUSIC on the captured block.
    from rfdf.dsp.covariance import sample_covariance
    from rfdf.dsp.doa import music
    from rfdf.dsp.steering import build_manifold

    iq = np.fromfile(recording.sigmf_data_path, dtype=np.complex64)
    iq = iq.reshape(recording.channels, -1)
    cov = sample_covariance(iq)
    positions = await geometry.positions()
    manifold = build_manifold(positions, np.arange(0.0, 360.0, 1.0), [0.0], _CENTER_FREQ_HZ)
    estimate = music(cov, manifold, num_signals=1)
    print(f"  MUSIC peak bearing: {estimate.azimuth_deg} deg azimuth")

    await sdr.close()
    print("demo: real-B210 coherent capture PASS")


def main() -> None:
    """Run the example, or print the hardware banner when no B210 is configured."""
    serials = _serials()
    if not serials:
        print(__doc__)
        print("No RFDF_B210_SERIALS set — hardware required, nothing to do.")
        sys.exit(0)
    asyncio.run(_run(serials))


if __name__ == "__main__":
    main()

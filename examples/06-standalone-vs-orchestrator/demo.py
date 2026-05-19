"""Standalone vs orchestrator-connected — the same DOA run, two modes.

The point of this example: rfdf is fully usable **standalone**. The
orchestrator integration is a bonus that adds context, traceability,
and memory — it never changes the answer.

Run it with just the base install to see standalone mode; install
``rfdf[orchestrator]`` to see what the orchestrator integration adds.

    python examples/06-standalone-vs-orchestrator/demo.py
"""

from __future__ import annotations

import asyncio
import json

from rfdf.backends.geometry.static import create as create_static_geometry
from rfdf.backends.sdr.mock import CWEmitter, MockSdrScenario
from rfdf.backends.sdr.mock import create as create_mock_sdr
from rfdf.dsp.doa.orchestration import Doa
from rfdf.dsp.doa.result import Algorithm
from rfdf.dsp.geometry_presets import half_wavelength_spacing, ula
from rfdf.hal import SdrConfig

_FREQ_HZ = 2.4e9
_TRUTH_AZIMUTHS = [35.0, 80.0]


async def _run_doa() -> dict:
    """Run one DOA estimate on a mock 8-element ULA. Pure standalone."""
    channels = 8
    positions = ula(channels, half_wavelength_spacing(_FREQ_HZ))
    geometry = create_static_geometry(antennas=positions.tolist())
    # A small per-emitter frequency offset decorrelates the sources so
    # the subspace estimator resolves them cleanly (coherent CW sources
    # at one frequency would need spatial smoothing).
    emitters = [
        CWEmitter(
            azimuth_deg=az,
            elevation_deg=0.0,
            power_dbm=0.0,
            freq_offset_hz=5e3 * index,
        )
        for index, az in enumerate(_TRUTH_AZIMUTHS)
    ]
    sdr = create_mock_sdr(
        geometry=geometry,
        scenario=MockSdrScenario(emitters=emitters, snr_db=25.0),
    )
    await sdr.configure(
        SdrConfig(
            center_freq_hz=_FREQ_HZ,
            sample_rate_hz=2.0e6,
            channels=list(range(channels)),
        )
    )
    estimate = await Doa(sdr, geometry, algorithm=Algorithm.MUSIC).run(
        duration_s=0.05, num_signals=len(_TRUTH_AZIMUTHS)
    )
    return {
        "algorithm": str(estimate.algorithm),
        "azimuth_deg": sorted(float(a) for a in estimate.azimuth_deg),
    }


def standalone_mode() -> dict:
    """Standalone: the DOA result, on screen and saved to local JSON."""
    print("=== standalone mode ===")
    result = asyncio.run(_run_doa())
    print(f"DOA estimate: {result['azimuth_deg']} deg ({result['algorithm']})")
    print(json.dumps(result, indent=2))
    print("Saved locally. That's it — no orchestrator needed.\n")
    return result


def orchestrator_mode(result: dict) -> None:
    """Orchestrator: the SAME result, plus a citation-grade evidence bundle.

    With a reachable orchestrator this also writes a Hindsight memory
    entry, an L5 vault note, and an ntfy alert — see the README. The
    answer is identical; the orchestrator adds context + traceability.
    """
    print("=== orchestrator-connected mode ===")
    from rfdf.orchestrator import is_available

    if not is_available():
        print("rfdf[orchestrator] not installed — install it to see:")
        print("  - a citation-grade evidence bundle for this run")
        print("  - a Hindsight memory entry + an L5 vault note")
        print("  - capability discovery by the orchestrator's planner")
        print("Standalone is NOT degraded — it is just less context-rich.\n")
        return

    from rfdf.orchestrator import build_bundle
    from rfdf.orchestrator.evidence import save_local

    bundle = build_bundle(
        "doa",
        inputs={
            "freq_hz": _FREQ_HZ,
            "truth_azimuths": _TRUTH_AZIMUTHS,
            "algorithm": result["algorithm"],
        },
        process={"algorithm": result["algorithm"], "num_signals": 2},
        outputs=result,
    )
    path = save_local(bundle)
    print(f"evidence bundle: {bundle.bundle_id} (quality: {bundle.quality})")
    print(f"reproducibility hash: {bundle.reproducibility_hash[:16]}…")
    print(f"saved: {path}")
    print(
        "With a reachable orchestrator, RfdfRecorder + RfdfAlerts would "
        "also write a Hindsight memory entry, an L5 vault note, and an "
        "ntfy alert. Same DOA answer — more context.\n"
    )


def main() -> None:
    """Run the DOA estimate in both modes and compare."""
    result = standalone_mode()
    orchestrator_mode(result)
    print("Both modes produced the same DOA estimate. The orchestrator")
    print("integration is bonus context — never a requirement.")


if __name__ == "__main__":
    main()

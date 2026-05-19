"""``RfdfConsumer`` — rfdf registered as an orchestrator consumer (Stage 7).

Each ``@capability``-decorated method is a **thin adapter**: it unpacks a
JSON payload, delegates to an existing rfdf platform class
(:class:`~rfdf.dsp.doa.Doa`, :class:`~rfdf.ml.inference.Classifier`,
the geometry presets), and returns a JSON-serialisable dict. No new
DSP/ML business logic lives here.

Domain imports (numpy, ``rfdf.dsp``, ``rfdf.ml``) are deferred into the
methods so that merely importing ``RfdfConsumer`` stays light — the
heavy stack loads only when a capability actually runs.
"""

from __future__ import annotations

from typing import Any

from ai_orchestrator_client import Consumer, capability

# Generic data-plane grants declared alongside the rf.* capabilities so
# the orchestrator's /consumers/{id}/{memory,vault,notify,evidence}
# push endpoints accept this consumer.
DATA_PLANE_CAPABILITIES = [
    "memory.write",
    "vault.write",
    "notify.send",
    "evidence.push",
]


class RfdfConsumer(Consumer):
    """rfdf as an orchestrator consumer.

    Registers under the name ``rfdf`` and exposes its DOA, classification,
    and geometry capabilities for orchestrator-dispatched calls. The
    capability handlers delegate to the standalone platform API — the
    same code paths the CLI and examples use.
    """

    name = "rfdf"

    @capability("rf.doa.run")
    async def run_doa(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Estimate direction of arrival on a mock-SDR scenario.

        Payload keys (all optional): ``algorithm``, ``channels``,
        ``freq_hz``, ``azimuths`` (list of emitter bearings, deg),
        ``snr_db``, ``duration_s``, ``num_signals`` (0 auto-estimates).
        """
        from rfdf.backends.geometry.static import create as create_static_geometry
        from rfdf.backends.sdr.mock import CWEmitter, Emitter, MockSdrScenario
        from rfdf.backends.sdr.mock import create as create_mock_sdr
        from rfdf.dsp.doa.orchestration import Doa
        from rfdf.dsp.doa.result import Algorithm
        from rfdf.dsp.geometry_presets import half_wavelength_spacing, ula
        from rfdf.hal import SdrConfig

        algorithm = Algorithm(payload.get("algorithm", "music"))
        channels = int(payload.get("channels", 8))
        freq_hz = float(payload.get("freq_hz", 2.4e9))
        azimuths = [float(a) for a in payload.get("azimuths", [30.0])]
        snr_db = float(payload.get("snr_db", 20.0))
        duration_s = float(payload.get("duration_s", 0.05))
        num_signals = int(payload.get("num_signals", 0)) or None

        positions = ula(channels, half_wavelength_spacing(freq_hz))
        geometry = create_static_geometry(antennas=positions.tolist())
        emitters: list[Emitter] = [
            CWEmitter(
                azimuth_deg=az,
                elevation_deg=0.0,
                power_dbm=0.0,
                freq_offset_hz=5e3 * index,
            )
            for index, az in enumerate(azimuths)
        ]
        sdr = create_mock_sdr(
            geometry=geometry,
            scenario=MockSdrScenario(emitters=emitters, snr_db=snr_db),
        )
        await sdr.configure(
            SdrConfig(
                center_freq_hz=freq_hz,
                sample_rate_hz=2.0e6,
                channels=list(range(channels)),
            )
        )
        doa = Doa(sdr, geometry, algorithm=algorithm)
        estimate = await doa.run(duration_s=duration_s, num_signals=num_signals)
        bearings = sorted(float(a) for a in estimate.azimuth_deg)
        return {
            "algorithm": str(estimate.algorithm),
            "azimuth_deg": bearings,
            "num_sources": len(bearings),
        }

    @capability("rf.classify")
    def classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Classify an IQ recording with a registered model.

        Payload keys: ``model_id`` (required), ``backend`` (default
        ``torch``), ``iq`` (required — a nested list of complex IQ, or
        ``[real, imag]`` pairs). Requires the ``[ml]`` extra and a model
        in the registry; raises a clear error otherwise.
        """
        import importlib

        import numpy as np

        # Imported dynamically: rfdf.ml pulls the heavy [ml] stack, which
        # is needed only when this capability actually runs.
        inference = importlib.import_module("rfdf.ml.inference")

        model_id = payload["model_id"]
        backend = payload.get("backend", "torch")
        raw = payload["iq"]
        iq = np.asarray(raw, dtype=np.complex64)

        classifier = inference.Classifier.from_registry(model_id, backend=backend)
        result = classifier.predict(iq)
        return {
            "model_id": model_id,
            "backend": backend,
            "top_k_classes": list(result.top_k_classes),
            "top_k_probabilities": [float(p) for p in result.top_k_probabilities],
            "inference_time_ms": float(result.inference_time_ms),
        }

    @capability("rf.geometry.morph")
    def geometry_morph(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return antenna positions for a geometry preset.

        Payload keys: ``preset`` (``ula`` | ``planar_cross``),
        ``num_elements`` (ULA only), ``spacing_m`` (defaults to λ/2 for
        ``freq_hz``), ``freq_hz``.
        """
        from rfdf.dsp.geometry_presets import (
            half_wavelength_spacing,
            planar_cross,
            ula,
        )

        preset = payload.get("preset", "ula")
        freq_hz = float(payload.get("freq_hz", 2.4e9))
        spacing_m = float(payload.get("spacing_m") or half_wavelength_spacing(freq_hz))
        if preset == "ula":
            positions = ula(int(payload.get("num_elements", 5)), spacing_m)
        elif preset == "planar_cross":
            positions = planar_cross(spacing_m)
        else:
            raise ValueError(f"unknown geometry preset {preset!r}; choose 'ula' or 'planar_cross'")
        return {
            "preset": preset,
            "spacing_m": spacing_m,
            "num_elements": int(positions.shape[0]),
            "positions_m": positions.tolist(),
        }

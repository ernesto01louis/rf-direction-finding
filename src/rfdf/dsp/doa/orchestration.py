"""High-level DOA orchestration — the ``Doa`` class."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.calibration import Calibration
from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.doa.bartlett import bartlett
from rfdf.dsp.doa.esprit import esprit, unitary_esprit
from rfdf.dsp.doa.music import music
from rfdf.dsp.doa.mvdr import mvdr
from rfdf.dsp.doa.result import Algorithm, DoaEstimate
from rfdf.dsp.doa.root_music import root_music
from rfdf.dsp.model_order import estimate_num_signals
from rfdf.dsp.steering import build_manifold
from rfdf.hal.geometry import GeometryController
from rfdf.hal.sdr import SdrSource

_GRID_ESTIMATORS: dict[Algorithm, Callable[..., DoaEstimate]] = {
    Algorithm.BARTLETT: bartlett,
    Algorithm.MVDR: mvdr,
    Algorithm.MUSIC: music,
}
_PARAMETRIC_ESTIMATORS: dict[Algorithm, Callable[..., DoaEstimate]] = {
    Algorithm.ROOT_MUSIC: root_music,
    Algorithm.ESPRIT: esprit,
    Algorithm.UNITARY_ESPRIT: unitary_esprit,
}


class Doa:
    """High-level direction-of-arrival orchestration.

    Wires an :class:`~rfdf.hal.SdrSource`, a :class:`~rfdf.hal.GeometryController`, an
    optional :class:`~rfdf.dsp.calibration.Calibration`, and an algorithm choice into a
    single :meth:`run` that captures IQ, builds the covariance, and returns a
    :class:`DoaEstimate`. This is the public DOA API the CLI and examples use.
    """

    def __init__(
        self,
        sdr: SdrSource,
        geometry: GeometryController,
        *,
        calibration: Calibration | None = None,
        algorithm: Algorithm = Algorithm.MUSIC,
    ) -> None:
        """Bind the hardware abstractions and the algorithm choice.

        Args:
            sdr: A configured SDR source.
            geometry: The antenna-geometry controller.
            calibration: Optional calibration applied to the captured IQ.
            algorithm: Which DOA estimator :meth:`run` should use.
        """
        self._sdr = sdr
        self._geometry = geometry
        self._calibration = calibration
        self._algorithm = Algorithm(algorithm)

    async def run(
        self,
        *,
        duration_s: float = 1.0,
        num_signals: int | None = None,
        grid_az_deg: ArrayLike | None = None,
        grid_el_deg: ArrayLike | None = None,
    ) -> DoaEstimate:
        """Capture IQ and estimate direction of arrival.

        Args:
            duration_s: How much IQ to integrate over.
            num_signals: Source count; ``None`` auto-estimates it with the MDL criterion.
            grid_az_deg: Azimuth scan grid for the grid estimators; defaults to a
                0.5-degree grid over [0, 180) — the unambiguous range of a ULA, and
                consistent with the parametric estimators. Pass a wider grid for a
                planar array with rear-hemisphere coverage.
            grid_el_deg: Elevation(s) for the scan; defaults to ``[0.0]``.

        Returns:
            A :class:`DoaEstimate` from the configured algorithm.
        """
        iq, freq_hz = await self._collect_iq(duration_s)
        positions = await self._geometry.positions()
        if self._calibration is not None:
            iq = np.asarray(self._calibration.apply(iq), dtype=np.complex128)
        covariance = sample_covariance(iq)

        num_channels = positions.shape[0]
        if num_signals is None:
            estimated = estimate_num_signals(covariance, snapshots=iq.shape[1], method="mdl")
            num_signals = min(max(estimated, 1), num_channels - 1)

        if self._algorithm in _GRID_ESTIMATORS:
            grid_az = (
                np.arange(0.0, 180.0, 0.5)
                if grid_az_deg is None
                else np.asarray(grid_az_deg, dtype=np.float64)
            )
            grid_el = (
                np.array([0.0])
                if grid_el_deg is None
                else np.asarray(grid_el_deg, dtype=np.float64)
            )
            manifold = build_manifold(positions, grid_az, grid_el, freq_hz)
            return _GRID_ESTIMATORS[self._algorithm](covariance, manifold, num_signals)
        return _PARAMETRIC_ESTIMATORS[self._algorithm](
            covariance, positions=positions, freq_hz=freq_hz, num_signals=num_signals
        )

    async def _collect_iq(self, duration_s: float) -> tuple[np.ndarray, float]:
        """Stream ``duration_s`` of IQ from the SDR; return the IQ and its frequency."""
        await self._sdr.start()
        blocks: list[np.ndarray] = []
        collected = 0
        freq_hz = 0.0
        target: int | None = None
        try:
            async for block in self._sdr.stream():
                blocks.append(block.iq.astype(np.complex128))
                collected += block.iq.shape[1]
                freq_hz = block.center_freq_hz
                if target is None:
                    target = max(1, int(duration_s * block.sample_rate_hz))
                if collected >= target:
                    break
        finally:
            await self._sdr.stop()
        if not blocks:
            raise RuntimeError("Doa.run: the SDR produced no IQ")
        return np.concatenate(blocks, axis=1), freq_hz

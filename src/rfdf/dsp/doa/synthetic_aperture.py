"""Position-domain synthetic aperture.

Fuses captures from several array positions ("stations") into one DOA estimate over a
larger virtual aperture. A morphing array on a motorised rail visits ``N`` positions;
``M`` antennas per station gives an ``M x N``-element virtual array, with proportionally
finer angular resolution.

Three fusion modes:

* ``coherent`` — phase-correct each station with its pilot reference, stack the IQ into
  one ``MN``-channel block, and estimate from the combined covariance. Delivers the full
  aperture gain; requires source coherence across the inter-station interval.
* ``incoherent`` — run a per-station estimate and average the pseudospectra in the log
  domain. Robust to bursty signals; no aperture gain.
* ``block-diagonal`` — assemble the per-station covariances into one block-diagonal
  virtual covariance and run a single MUSIC over the virtual array. (The optional
  pilot-phase cross-blocks, which would add partial coherent gain, are left zero — a
  documented simplification; see STAGE-3-OUTPUTS.)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.calibration import Calibration
from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.doa._common import peak_pick_1d, signal_noise_subspaces
from rfdf.dsp.doa.music import music
from rfdf.dsp.doa.mvdr import mvdr
from rfdf.dsp.doa.result import DoaEstimate
from rfdf.dsp.steering import build_manifold

_FUSIONS = ("coherent", "incoherent", "block-diagonal")
_ALGORITHMS: dict[str, Callable[..., DoaEstimate]] = {"music": music, "mvdr": mvdr}
_NULL_FLOOR: float = 1e-300


@dataclass(frozen=True)
class StationCapture:
    """One station's contribution to a synthetic aperture.

    Attributes:
        positions: This station's antenna positions, shape ``(M, 3)`` in metres.
        iq: This station's IQ block, shape ``(M, T)``.
        pilot_phase_rad: The station's residual phase from a pilot-tone measurement,
            used to align it with the other stations under coherent fusion.
    """

    positions: np.ndarray
    iq: np.ndarray
    pilot_phase_rad: float = 0.0


def synthetic_aperture_doa(
    captures: Sequence[StationCapture],
    *,
    calibration: Calibration,
    freq_hz: float,
    az_grid_deg: ArrayLike,
    algorithm: str = "music",
    fusion: str = "coherent",
    num_signals: int = 1,
    pilot_phase_corrections: Sequence[float] | None = None,
) -> DoaEstimate:
    """Estimate DOA by fusing several station captures into a synthetic aperture.

    Args:
        captures: One :class:`StationCapture` per array position.
        calibration: The per-station array calibration (applied to every station's IQ).
        freq_hz: RF frequency in hertz.
        az_grid_deg: 1-D azimuth grid in degrees.
        algorithm: ``"music"`` or ``"mvdr"`` — the estimator for coherent fusion and
            the per-station estimator for incoherent fusion.
        fusion: ``"coherent"``, ``"incoherent"``, or ``"block-diagonal"``.
        num_signals: Number of sources.
        pilot_phase_corrections: Optional per-station phases overriding each capture's
            ``pilot_phase_rad``.

    Returns:
        A :class:`DoaEstimate` with the recovered bearings.

    Raises:
        ValueError: If ``captures`` is empty, or ``algorithm`` / ``fusion`` is unknown.
    """
    if not captures:
        raise ValueError("synthetic_aperture_doa needs at least one StationCapture")
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"algorithm must be 'music' or 'mvdr', got {algorithm!r}")
    if fusion not in _FUSIONS:
        raise ValueError(f"fusion must be one of {_FUSIONS}, got {fusion!r}")
    estimator = _ALGORITHMS[algorithm]
    grid = np.asarray(az_grid_deg, dtype=np.float64)
    phases = (
        list(pilot_phase_corrections)
        if pilot_phase_corrections is not None
        else [capture.pilot_phase_rad for capture in captures]
    )
    corrected = [
        np.asarray(calibration.apply(capture.iq), dtype=np.complex128) for capture in captures
    ]

    if fusion == "incoherent":
        accumulated = np.zeros(grid.size, dtype=np.float64)
        for iq, capture in zip(corrected, captures, strict=True):
            covariance = sample_covariance(iq)
            manifold = build_manifold(capture.positions, grid, np.array([0.0]), freq_hz)
            spectrum_db = estimator(covariance, manifold, num_signals).pseudospectrum_db
            accumulated += np.asarray(spectrum_db, dtype=np.float64)
        spectrum = 10.0 ** (accumulated / len(captures) / 10.0)
        indices, azimuths, strengths, full_db = peak_pick_1d(spectrum, grid, num_signals)
        return DoaEstimate(
            algorithm="synthetic_aperture_incoherent",
            num_signals=num_signals,
            azimuth_deg=azimuths,
            elevation_deg=[0.0] * len(azimuths),
            pseudospectrum_db=full_db,
            peak_indices=indices,
            peak_strengths_db=strengths,
        )

    virtual_positions = np.vstack(
        [np.asarray(capture.positions, dtype=np.float64) for capture in captures]
    )
    manifold = build_manifold(virtual_positions, grid, np.array([0.0]), freq_hz)

    if fusion == "coherent":
        stacked = np.vstack(
            [iq * np.exp(-1j * phase) for iq, phase in zip(corrected, phases, strict=True)]
        )
        covariance = sample_covariance(stacked)
        estimate = estimator(covariance, manifold, num_signals)
        return estimate.model_copy(update={"algorithm": "synthetic_aperture_coherent"})

    # fusion == "block-diagonal"
    blocks = [sample_covariance(iq) for iq in corrected]
    channels = blocks[0].shape[0]
    total = channels * len(blocks)
    virtual = np.zeros((total, total), dtype=np.complex128)
    for index, block in enumerate(blocks):
        start = index * channels
        virtual[start : start + channels, start : start + channels] = block
    _, noise = signal_noise_subspaces(virtual, num_signals * len(blocks))
    projection = manifold.matrix @ noise.conj()
    null = np.real(np.sum(np.abs(projection) ** 2, axis=1))
    spectrum = 1.0 / np.maximum(null, _NULL_FLOOR)
    indices, azimuths, strengths, full_db = peak_pick_1d(spectrum, grid, num_signals)
    return DoaEstimate(
        algorithm="synthetic_aperture_block_diagonal",
        num_signals=num_signals,
        azimuth_deg=azimuths,
        elevation_deg=[0.0] * len(azimuths),
        pseudospectrum_db=full_db,
        peak_indices=indices,
        peak_strengths_db=strengths,
    )

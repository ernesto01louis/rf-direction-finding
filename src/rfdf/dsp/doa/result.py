"""Result types shared by the direction-of-arrival estimators."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Algorithm(StrEnum):
    """The classical DOA estimators rfdf provides."""

    BARTLETT = "bartlett"
    MVDR = "mvdr"
    MUSIC = "music"
    ROOT_MUSIC = "root_music"
    ESPRIT = "esprit"
    UNITARY_ESPRIT = "unitary_esprit"


class DoaEstimate(BaseModel):
    """The outcome of a direction-of-arrival estimate.

    Attributes:
        algorithm: Name of the estimator that produced this result.
        num_signals: Number of sources the estimate was solved for.
        azimuth_deg: Estimated source azimuths in degrees, ascending.
        elevation_deg: Estimated source elevations in degrees.
        pseudospectrum_db: The pseudospectrum in dB for grid estimators; ``None``
            for the parametric estimators (Root-MUSIC, ESPRIT).
        peak_indices: Grid indices of the chosen peaks (grid estimators only).
        peak_strengths_db: Pseudospectrum height at each peak, in dB.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    algorithm: str
    num_signals: int
    azimuth_deg: list[float]
    elevation_deg: list[float]
    pseudospectrum_db: np.ndarray | None = None
    peak_indices: list[int] = Field(default_factory=list)
    peak_strengths_db: list[float] = Field(default_factory=list)


class Doa2DResult(BaseModel):
    """The outcome of a 2-D (azimuth + elevation) direction-of-arrival estimate.

    Attributes:
        algorithm: Name of the estimator that produced this result.
        num_signals: Number of sources the estimate was solved for.
        azimuth_deg: Estimated source azimuths in degrees.
        elevation_deg: Estimated source elevations in degrees.
        pseudospectrum_db: The 2-D pseudospectrum surface in dB, shape
            ``(len(az_grid_deg), len(el_grid_deg))``.
        az_grid_deg: The azimuth grid the surface was evaluated over.
        el_grid_deg: The elevation grid the surface was evaluated over.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    algorithm: str
    num_signals: int
    azimuth_deg: list[float]
    elevation_deg: list[float]
    pseudospectrum_db: np.ndarray
    az_grid_deg: np.ndarray
    el_grid_deg: np.ndarray

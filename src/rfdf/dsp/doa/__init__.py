"""Classical direction-of-arrival estimators.

Grid estimators (:func:`bartlett`, :func:`mvdr`, :func:`music`) take a precomputed
:class:`~rfdf.dsp.steering.SteeringManifold`; the parametric ULA estimators
(:func:`root_music`, :func:`esprit`, :func:`unitary_esprit`) take antenna positions
directly. All return a :class:`DoaEstimate`.
"""

from __future__ import annotations

from rfdf.dsp.doa.bartlett import bartlett
from rfdf.dsp.doa.esprit import esprit, unitary_esprit
from rfdf.dsp.doa.music import music
from rfdf.dsp.doa.music_2d import music_2d
from rfdf.dsp.doa.mvdr import mvdr
from rfdf.dsp.doa.orchestration import Doa
from rfdf.dsp.doa.result import Algorithm, Doa2DResult, DoaEstimate
from rfdf.dsp.doa.root_music import root_music
from rfdf.dsp.doa.synthetic_aperture import StationCapture, synthetic_aperture_doa
from rfdf.dsp.doa.wideband import cssm, incoherent_wideband_music

__all__ = [
    "Algorithm",
    "Doa",
    "Doa2DResult",
    "DoaEstimate",
    "StationCapture",
    "bartlett",
    "cssm",
    "esprit",
    "incoherent_wideband_music",
    "music",
    "music_2d",
    "mvdr",
    "root_music",
    "synthetic_aperture_doa",
    "unitary_esprit",
]

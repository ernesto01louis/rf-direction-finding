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
from rfdf.dsp.doa.mvdr import mvdr
from rfdf.dsp.doa.result import Algorithm, DoaEstimate
from rfdf.dsp.doa.root_music import root_music

__all__ = [
    "Algorithm",
    "DoaEstimate",
    "bartlett",
    "esprit",
    "music",
    "mvdr",
    "root_music",
    "unitary_esprit",
]

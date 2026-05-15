"""Digital signal processing layer: classical direction-of-arrival estimation.

The ``rfdf.dsp`` package implements the Stage 3 DOA pipeline — steering manifolds,
covariance estimation, classical subspace and beamforming estimators (MUSIC, ESPRIT,
MVDR, Bartlett, Root-MUSIC), calibration, the Cramer-Rao lower bound, and the
position-domain synthetic aperture.

Every module here is pure NumPy/SciPy and synchronous. DOA code operates on plain
covariance matrices and antenna-position arrays; it never imports a hardware backend.
"""

from __future__ import annotations

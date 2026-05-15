"""Geometry backends implementing :class:`rfdf.hal.GeometryController`.

Stage 2 ships:

* :mod:`rfdf.backends.geometry.static` — fixed positions supplied at construction.
* :mod:`rfdf.backends.geometry.mock_morph` — morphing array with simulated
  positioning error + TOML-backed presets.

Real-rail backends (grbl-linear) land in Stage 5.
"""

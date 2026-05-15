"""Rotator backends implementing :class:`rfdf.hal.RotatorController`.

Stage 2 ships:

* :mod:`rfdf.backends.rotator.mock` — software-only rotator with realistic slew
  dynamics.

Real hardware backends (AntRunner via HTTP, hamlib rotctld over TCP, SPID,
Yaesu) land in Stage 5.
"""

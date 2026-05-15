"""Compute backends implementing :class:`rfdf.hal.ComputeBackend`.

Stage 2 ships:

* :mod:`rfdf.backends.compute.local` — in-process or subprocess execution on the
  caller's machine. CUDA / MPS / CPU auto-detected; no torch import at module
  load (lazy).

Remote-rental backends (RunPod, Modal, Vast.ai, SkyPilot) land in Stage 4
behind the corresponding optional extras.
"""

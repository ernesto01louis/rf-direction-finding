"""Hardware abstraction layer for rfdf.

Four Protocol classes form the contract every backend implements:

* :class:`SdrSource` — IQ capture, tuning, calibration.
* :class:`RotatorController` — mechanical AZ/EL pointing.
* :class:`GeometryController` — per-antenna 3D position.
* :class:`ComputeBackend` — ML / batch job dispatch.

Backends register via Python entry-points under ``rfdf.backends.{sdr,rotator,
geometry,compute}``; see :mod:`rfdf.hal.discovery` for the lookup helper.

This module imports nothing hardware-specific. Backends live in ``rfdf.backends``
and are imported lazily by the discovery helper, preserving the
``zero-domain-deps`` audit-lesson guarantee.
"""

from rfdf.hal.compute import ComputeBackend, ComputeJob, CostEstimate, JobHandle
from rfdf.hal.discovery import BACKEND_GROUPS, discover_backends, list_backends, load_backend
from rfdf.hal.geometry import GeometryController
from rfdf.hal.rotator import RotatorController
from rfdf.hal.sdr import Recording, SdrConfig, SdrSource, StreamBlock
from rfdf.hal.types import BackendLoadError, CalibrationReport, JobStatus

__all__ = [
    "BACKEND_GROUPS",
    "BackendLoadError",
    "CalibrationReport",
    "ComputeBackend",
    "ComputeJob",
    "CostEstimate",
    "GeometryController",
    "JobHandle",
    "JobStatus",
    "Recording",
    "RotatorController",
    "SdrConfig",
    "SdrSource",
    "StreamBlock",
    "discover_backends",
    "list_backends",
    "load_backend",
]

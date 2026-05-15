"""Shared types for the rfdf hardware abstraction layer.

These types are referenced by more than one Protocol (e.g. ``CalibrationReport`` is
returned by both ``RotatorController.calibrate`` and ``GeometryController.calibrate``).
Keeping them in a single module avoids circular imports and gives the public surface a
stable home.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Lifecycle states a ``ComputeBackend`` job can be in.

    The values are stable strings so JSON dumps round-trip cleanly. Backends MUST map
    their provider-specific states onto exactly one of these.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CalibrationReport(BaseModel):
    """Outcome of a ``calibrate()`` call on a rotator or geometry controller.

    The report is intentionally lean: every backend reports success / failure and a
    short human-readable note. Backend-specific telemetry (encoder counts, motor
    currents, rail end-stop offsets) lives in ``details`` so the contract stays small.

    Attributes:
        ok: True if the device returned to a known good zero / home position.
        message: One-line summary fit for a CLI status line.
        backend: Identifier of the backend that produced the report (e.g. ``"mock"``).
        timestamp: When the calibration completed (UTC).
        details: Backend-specific extra fields; opaque to ``rfdf`` core.
    """

    ok: bool
    message: str
    backend: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)


class BackendLoadError(RuntimeError):
    """Raised when a backend factory exists but fails to produce a configured instance.

    Wraps the original exception in ``__cause__`` so callers can pattern-match without
    losing the traceback. Distinct from a missing-entry-point error: missing backends
    surface as ``KeyError`` on the discovery catalog.

    Attributes:
        group: The entry-point group that owns the backend (e.g. ``"rfdf.backends.sdr"``).
        name: The backend's user-facing identifier (e.g. ``"mock"``).
    """

    def __init__(self, group: str, name: str, original: BaseException) -> None:
        """Initialise with the failing group + name + cause."""
        super().__init__(
            f"backend factory {group!r}/{name!r} raised {type(original).__name__}: {original}"
        )
        self.group = group
        self.name = name
        self.__cause__ = original

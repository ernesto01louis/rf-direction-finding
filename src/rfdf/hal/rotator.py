"""``RotatorController`` Protocol for AZ/EL mechanical pointing.

Implementations: ``mock`` (Stage 2), AntRunner / Yaesu G-5500DC / hamlib ``rotctld`` /
SPID (Stage 5+). Backend authors implement this Protocol and register an entry point
under ``rfdf.backends.rotator``; the discovery layer picks them up automatically.

Stage 2 deviation from the original Stage 2 PDF: ``position()`` and ``positions()``
are regular async methods rather than ``@property async def`` (which is not valid
Python — see ``STAGE-2-OUTPUTS.md`` §4). ``stream_position()`` is part of the
Protocol surface so the mock backend's slew-progress streaming can stay contract-
conformant.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from rfdf.hal.types import CalibrationReport


@runtime_checkable
class RotatorController(Protocol):
    """Drives a 2-axis (AZ/EL) rotator.

    Backends may expose only one axis (set the unsupported axis's ``supports_*``
    capability to ``False`` and raise on any movement command in the missing axis).
    Coordinate frame: azimuth 0° = magnetic North (or per-installation reference),
    elevation 0° = horizon, 90° = zenith. Backends document any deviation.
    """

    @property
    def supports_azimuth(self) -> bool:
        """True if the backend can change azimuth."""
        ...

    @property
    def supports_elevation(self) -> bool:
        """True if the backend can change elevation."""
        ...

    @property
    def azimuth_range_deg(self) -> tuple[float, float]:
        """Inclusive ``(min, max)`` reachable azimuth in degrees."""
        ...

    @property
    def elevation_range_deg(self) -> tuple[float, float]:
        """Inclusive ``(min, max)`` reachable elevation in degrees."""
        ...

    @property
    def max_speed_deg_per_s(self) -> float:
        """Maximum slew speed across both axes."""
        ...

    @property
    def positioning_accuracy_deg(self) -> float:
        """Worst-case absolute positioning error after a settled ``goto``."""
        ...

    async def goto(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Slew to ``(azimuth_deg, elevation_deg)`` and return when settled.

        Backends MUST validate the target is within their reported range and
        raise ``ValueError`` otherwise. The call completes only after the
        rotator reports the move has settled within ``positioning_accuracy_deg``.
        """
        ...

    async def park(self) -> None:
        """Slew to a backend-defined safe storage position."""
        ...

    async def stop(self) -> None:
        """Halt any in-flight motion immediately."""
        ...

    async def position(self) -> tuple[float, float]:
        """Return the current ``(azimuth_deg, elevation_deg)`` reading."""
        ...

    def stream_position(self) -> AsyncIterator[tuple[float, float]]:
        """Yield ``(az, el)`` samples during a slew until motion stops.

        Implementations should make this an ``async def`` generator so callers
        can break out cleanly; the generator terminates when the rotator reports
        the move has settled (i.e. ``position()`` is constant for the backend's
        settle window).
        """
        ...

    async def calibrate(self) -> CalibrationReport:
        """Run the backend's home / index / encoder-zero routine."""
        ...

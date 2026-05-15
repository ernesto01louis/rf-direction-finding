"""``GeometryController`` Protocol for per-antenna 3D position.

A geometry controller knows where each receive antenna physically is. Static arrays
(fixed LPDA cluster) report the same constant positions; morphing arrays (motorized
linear rails, Stewart platform) report current commanded positions. The mock SDR
consumes the geometry to synthesize per-channel IQ — changing geometry
programmatically changes the synthesized IQ in the physically correct way.

Stage 2 deviation from the original Stage 2 PDF: ``positions()`` is a regular async
method rather than ``@property async def`` (not valid Python). ``calibrate()`` is
included on the Protocol surface so morphing arrays with rail end-stops can report
zero-position calibration; static arrays MAY return ``ok=True`` with a no-op message.
See ``STAGE-2-OUTPUTS.md`` §4.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rfdf.hal.types import CalibrationReport


@runtime_checkable
class GeometryController(Protocol):
    """Reports antenna positions in the rotator-local coordinate frame.

    Frame convention: ``+x`` points forward along the rotator's azimuth zero,
    ``+y`` points to the left looking forward, ``+z`` points up. Units are
    meters throughout. Backends document any deviation from this convention.
    """

    @property
    def num_antennas(self) -> int:
        """Number of receive antennas this geometry exposes."""
        ...

    @property
    def is_morphable(self) -> bool:
        """True if positions can be commanded at runtime (e.g. motorized rails)."""
        ...

    @property
    def positioning_repeatability_mm(self) -> float:
        """Worst-case repeatability of a commanded position, in millimetres."""
        ...

    async def positions(self) -> np.ndarray:
        """Return current antenna positions in meters.

        Returns:
            Array of shape ``(num_antennas, 3)`` with ``(x, y, z)`` per row.
        """
        ...

    async def goto_preset(self, preset_name: str) -> None:
        """Move to a named preset. Static backends raise ``NotImplementedError``."""
        ...

    async def goto_positions(self, positions: np.ndarray) -> None:
        """Command the array to ``positions``; shape must match ``num_antennas``.

        Static backends raise ``NotImplementedError``. Morphing backends raise
        ``ValueError`` if any position is outside their reachable workspace.
        """
        ...

    async def list_presets(self) -> list[str]:
        """List names of presets known to this backend."""
        ...

    async def save_preset(self, name: str, positions: np.ndarray) -> None:
        """Persist ``positions`` under ``name`` for later ``goto_preset`` calls."""
        ...

    async def calibrate(self) -> CalibrationReport:
        """Run zero-position / rail-end-stop calibration.

        Static backends return ``ok=True`` immediately with a no-op message;
        morphing backends drive each axis to its limit switch and zero encoders.
        """
        ...

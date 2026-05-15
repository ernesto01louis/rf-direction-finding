"""Fixed-geometry backend.

A ``static`` geometry reports the antenna positions supplied at construction and
refuses morphing commands. The most common production deployment: a fixed LPDA
cluster bolted to a rotator plate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rfdf.hal.types import CalibrationReport


class StaticGeometry:
    """Implements :class:`rfdf.hal.GeometryController` for a fixed array.

    Args:
        antennas: Sequence of ``(x, y, z)`` tuples in meters, rotator-local frame.
    """

    def __init__(self, antennas: Sequence[Sequence[float]]) -> None:
        """Cache the supplied positions as a ``(N, 3)`` float64 array."""
        arr = np.asarray(antennas, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"static geometry: expected (N, 3) antenna positions, got shape {arr.shape}"
            )
        if arr.shape[0] == 0:
            raise ValueError("static geometry: at least one antenna position required")
        self._positions = arr

    @property
    def num_antennas(self) -> int:
        """Return the configured antenna count."""
        return int(self._positions.shape[0])

    @property
    def is_morphable(self) -> bool:
        """Always ``False`` — static arrays do not move."""
        return False

    @property
    def positioning_repeatability_mm(self) -> float:
        """Zero — positions are bolt-down constants."""
        return 0.0

    async def positions(self) -> np.ndarray:
        """Return a copy of the cached antenna positions."""
        return self._positions.copy()

    async def goto_preset(self, preset_name: str) -> None:
        """Always raises — static arrays don't support presets."""
        raise NotImplementedError(
            f"static geometry: goto_preset({preset_name!r}) — array is not morphable"
        )

    async def goto_positions(self, positions: np.ndarray) -> None:
        """Always raises — static arrays don't accept new positions."""
        raise NotImplementedError("static geometry: goto_positions() — array is not morphable")

    async def list_presets(self) -> list[str]:
        """Empty list — static arrays have no presets."""
        return []

    async def save_preset(self, name: str, positions: np.ndarray) -> None:
        """Always raises — static arrays don't store presets."""
        raise NotImplementedError(
            f"static geometry: save_preset({name!r}) — array is not morphable"
        )

    async def calibrate(self) -> CalibrationReport:
        """Trivial pass — static arrays are pre-calibrated by installation."""
        return CalibrationReport(
            ok=True,
            message="static geometry: positions fixed at construction; no calibration required",
            backend="static",
        )


def create(antennas: Sequence[Sequence[float]] | None = None, **_: Any) -> StaticGeometry:
    """Factory wired into the ``rfdf.backends.geometry`` entry-point.

    Args:
        antennas: Antenna positions. Falls back to the pentagonal 5-antenna
            cluster at λ/2 spacing for 868 MHz used by the demo-no-hardware
            smoke test when ``None``.

    Returns:
        Configured :class:`StaticGeometry` instance.
    """
    if antennas is None:
        antennas = [
            (0.0, 0.0, 0.0),
            (0.17, 0.0, 0.0),
            (0.0, 0.17, 0.0),
            (-0.17, 0.0, 0.0),
            (0.0, -0.17, 0.0),
        ]
    return StaticGeometry(antennas=antennas)

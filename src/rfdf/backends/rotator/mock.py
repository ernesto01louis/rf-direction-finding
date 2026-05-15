"""Mock rotator backend.

Simulates an AZ/EL rotator with constant-velocity slew, configurable settling
time, and Gaussian positioning error after settling. Useful for testing
campaign workflows that depend on rotator state without burning rotator hours.

``stream_position()`` yields intermediate positions during a slew so progress
UIs and any slew-aware DSP can be developed end-to-end.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from rfdf.hal.types import CalibrationReport


class MockRotator:
    """Implements :class:`rfdf.hal.RotatorController` purely in software.

    Args:
        azimuth_range_deg: Reachable azimuth interval.
        elevation_range_deg: Reachable elevation interval.
        max_speed_deg_per_s: Constant slew speed used to time ``goto`` returns.
        positioning_accuracy_deg: 1-sigma post-settle noise applied to the
            commanded position.
        seed: RNG seed for deterministic positioning noise.
        slew_tick_s: Sleep period between ``stream_position()`` updates.
    """

    def __init__(
        self,
        *,
        azimuth_range_deg: tuple[float, float] = (0.0, 360.0),
        elevation_range_deg: tuple[float, float] = (0.0, 90.0),
        max_speed_deg_per_s: float = 6.0,
        positioning_accuracy_deg: float = 0.5,
        seed: int = 0,
        slew_tick_s: float = 0.01,
    ) -> None:
        """Initialise capabilities + state."""
        if azimuth_range_deg[0] >= azimuth_range_deg[1]:
            raise ValueError(f"invalid azimuth_range_deg: {azimuth_range_deg}")
        if elevation_range_deg[0] >= elevation_range_deg[1]:
            raise ValueError(f"invalid elevation_range_deg: {elevation_range_deg}")
        if max_speed_deg_per_s <= 0:
            raise ValueError(f"max_speed_deg_per_s must be > 0, got {max_speed_deg_per_s}")
        self._az_range = azimuth_range_deg
        self._el_range = elevation_range_deg
        self._speed = float(max_speed_deg_per_s)
        self._accuracy = float(positioning_accuracy_deg)
        self._rng = np.random.default_rng(seed)
        self._slew_tick_s = float(slew_tick_s)
        self._az = azimuth_range_deg[0]
        self._el = elevation_range_deg[0]
        self._target_az = self._az
        self._target_el = self._el
        self._slew_lock = asyncio.Lock()
        self._stopped = False

    @property
    def supports_azimuth(self) -> bool:
        """Mock supports both axes."""
        return True

    @property
    def supports_elevation(self) -> bool:
        """Mock supports both axes."""
        return True

    @property
    def azimuth_range_deg(self) -> tuple[float, float]:
        """Reachable azimuth interval."""
        return self._az_range

    @property
    def elevation_range_deg(self) -> tuple[float, float]:
        """Reachable elevation interval."""
        return self._el_range

    @property
    def max_speed_deg_per_s(self) -> float:
        """Configured slew speed."""
        return self._speed

    @property
    def positioning_accuracy_deg(self) -> float:
        """Configured post-settle accuracy (1-sigma)."""
        return self._accuracy

    async def goto(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Slew to ``(azimuth_deg, elevation_deg)`` at max_speed, then settle with noise."""
        self._validate(azimuth_deg, elevation_deg)
        self._stopped = False
        async with self._slew_lock:
            self._target_az = azimuth_deg
            self._target_el = elevation_deg
            distance = math.hypot(
                azimuth_deg - self._az,
                elevation_deg - self._el,
            )
            slew_time_s = distance / self._speed
            await asyncio.sleep(slew_time_s)
            if self._stopped:
                return
            # Apply settle noise to model real positioning error.
            self._az = azimuth_deg + float(self._rng.normal(0.0, self._accuracy))
            self._el = elevation_deg + float(self._rng.normal(0.0, self._accuracy))

    async def park(self) -> None:
        """Park at the minimum azimuth / elevation corner."""
        await self.goto(self._az_range[0], self._el_range[0])

    async def stop(self) -> None:
        """Halt any pending slew. Idempotent."""
        self._stopped = True

    async def position(self) -> tuple[float, float]:
        """Return the current ``(azimuth_deg, elevation_deg)`` reading."""
        return (self._az, self._el)

    async def stream_position(self) -> AsyncIterator[tuple[float, float]]:
        """Yield ``(az, el)`` samples until the rotator settles.

        Interpolates linearly between current position and target at
        ``max_speed_deg_per_s`` with ``slew_tick_s`` resolution.
        """
        start_az, start_el = self._az, self._el
        target_az, target_el = self._target_az, self._target_el
        distance = math.hypot(target_az - start_az, target_el - start_el)
        if distance == 0.0:
            yield (self._az, self._el)
            return
        total_time = distance / self._speed
        steps = max(1, math.ceil(total_time / self._slew_tick_s))
        for i in range(steps + 1):
            if self._stopped:
                break
            t = min(1.0, i / steps)
            az = start_az + (target_az - start_az) * t
            el = start_el + (target_el - start_el) * t
            yield (az, el)
            await asyncio.sleep(self._slew_tick_s)

    async def calibrate(self) -> CalibrationReport:
        """Simulate a home routine: zero the position to the lower-left corner."""
        self._az = self._az_range[0]
        self._el = self._el_range[0]
        self._target_az = self._az
        self._target_el = self._el
        return CalibrationReport(
            ok=True,
            message=f"mock rotator: homed to ({self._az:.2f}, {self._el:.2f})",
            backend="mock",
            details={
                "az_range_deg": list(self._az_range),
                "el_range_deg": list(self._el_range),
            },
        )

    def _validate(self, az: float, el: float) -> None:
        if not (self._az_range[0] <= az <= self._az_range[1]):
            raise ValueError(f"mock rotator: azimuth {az} outside {self._az_range}")
        if not (self._el_range[0] <= el <= self._el_range[1]):
            raise ValueError(f"mock rotator: elevation {el} outside {self._el_range}")


def create(**kwargs: Any) -> MockRotator:
    """Factory wired into the ``rfdf.backends.rotator`` entry-point.

    Args:
        **kwargs: Forwarded to :class:`MockRotator`.

    Returns:
        Configured :class:`MockRotator`.
    """
    return MockRotator(**kwargs)

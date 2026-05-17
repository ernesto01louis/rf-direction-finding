"""AntRunner AZ/EL rotator backend.

The wuxx/AntRunner is a GRBL_ESP32-based AZ/EL rotator. Its firmware exposes a
CNC G-code dialect over an HTTP command endpoint; this backend wraps that in the
Stage-2 ``RotatorController`` HAL contract — the same contract the ``mock``
rotator passes.

Axis mapping: GRBL **X = azimuth**, **Y = elevation**, both in degrees. Moves
are absolute (``G90 G0 X<az> Y<el>``). With ``encoder_validation`` the backend
reads the controller's reported position back after every move and raises
:class:`RotatorPositionError` on a mismatch beyond ``positioning_accuracy_deg``.

The HTTP transport (``rfdf.backends._grbl``) lazy-imports ``httpx``; ``import
rfdf`` and base discovery work without the ``[rotator-antrunner]`` extra.

Registered as the ``antrunner`` rotator backend.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from rfdf.backends._grbl import GrblConnectionError, GrblHttpClient, GrblStatus
from rfdf.hal.types import CalibrationReport

_log = logging.getLogger(__name__)

#: How long to wait for a slew to settle (state returns to Idle) before failing.
_SETTLE_TIMEOUT_S = 120.0
#: Poll period while waiting for a slew to settle.
_POLL_S = 0.25


class AntRunnerError(RuntimeError):
    """Base class for every error raised by the AntRunner backend."""


class RotatorPositionError(AntRunnerError):
    """The controller's reported position disagrees with the commanded position.

    Raised by closed-loop validation after a ``goto`` — a mechanical slip, a
    lost step, or an encoder fault.
    """


class RotatorNotHomedError(AntRunnerError):
    """A move was requested before the mandatory startup homing cycle ran."""


class AntRunnerRotator:
    """wuxx/AntRunner GRBL_ESP32-based AZ/EL rotator.

    Args:
        host: IP or hostname of the AntRunner ESP32 controller.
        port: HTTP port (default 80).
        max_speed_deg_per_s: Reported slew speed (informational; the GRBL
            controller enforces its own feed rate).
        homing_required_on_startup: When True, ``goto`` raises until
            ``calibrate()`` (the homing cycle) has run.
        encoder_validation: When True, every ``goto`` reads the controller's
            position back and raises on a mismatch.
        positioning_accuracy_deg: Closed-loop agreement tolerance.
        command_path: GRBL_ESP32 command-endpoint template (``{cmd}`` is the
            URL-encoded command); varies across firmware forks.
    """

    #: Cable-management soft limits (degrees). +/-180 azimuth = +/-1 cable turn;
    #: see docs/hardware/rotator-antrunner.md. Pushed to GRBL $130/$131 by
    #: ``calibrate()``.
    soft_limits_az: tuple[float, float] = (-180.0, 180.0)
    soft_limits_el: tuple[float, float] = (0.0, 180.0)

    def __init__(
        self,
        host: str,
        *,
        port: int = 80,
        max_speed_deg_per_s: float = 6.0,
        homing_required_on_startup: bool = True,
        encoder_validation: bool = True,
        positioning_accuracy_deg: float = 0.1,
        command_path: str = "/command?commandText={cmd}",
    ) -> None:
        """Capture configuration; no network I/O until the first command."""
        if not host:
            raise AntRunnerError("AntRunnerRotator requires a controller host")
        self._grbl = GrblHttpClient(host=host, port=port, command_path=command_path)
        self._max_speed = float(max_speed_deg_per_s)
        self._homing_required = bool(homing_required_on_startup)
        self._encoder_validation = bool(encoder_validation)
        self._accuracy = float(positioning_accuracy_deg)
        self._homed = not self._homing_required

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def supports_azimuth(self) -> bool:
        """The AntRunner drives azimuth (GRBL X axis)."""
        return True

    @property
    def supports_elevation(self) -> bool:
        """The AntRunner drives elevation (GRBL Y axis)."""
        return True

    @property
    def azimuth_range_deg(self) -> tuple[float, float]:
        """Reachable azimuth — the cable-management soft limits."""
        return self.soft_limits_az

    @property
    def elevation_range_deg(self) -> tuple[float, float]:
        """Reachable elevation — the cable-management soft limits."""
        return self.soft_limits_el

    @property
    def max_speed_deg_per_s(self) -> float:
        """Reported slew speed."""
        return self._max_speed

    @property
    def positioning_accuracy_deg(self) -> float:
        """Closed-loop agreement tolerance."""
        return self._accuracy

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    async def goto(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Slew to ``(azimuth_deg, elevation_deg)`` and return once settled.

        Raises:
            ValueError: If the target is outside the soft limits.
            RotatorNotHomedError: If homing is required and has not run.
            RotatorPositionError: If closed-loop readback disagrees with the
                commanded position.
        """
        self._validate(azimuth_deg, elevation_deg)
        if not self._homed:
            raise RotatorNotHomedError(
                "AntRunner: homing is required before motion — call calibrate() first."
            )
        await self._grbl.send(f"G90 G0 X{azimuth_deg:.4f} Y{elevation_deg:.4f}")
        await self._wait_until_idle()
        if self._encoder_validation:
            actual_az, actual_el = await self.position()
            daz, dele = abs(actual_az - azimuth_deg), abs(actual_el - elevation_deg)
            if daz > self._accuracy or dele > self._accuracy:
                raise RotatorPositionError(
                    f"AntRunner closed-loop mismatch: commanded "
                    f"({azimuth_deg:.3f}, {elevation_deg:.3f}), reported "
                    f"({actual_az:.3f}, {actual_el:.3f}); delta "
                    f"({daz:.3f}, {dele:.3f}) exceeds {self._accuracy:.3f} deg."
                )

    async def park(self) -> None:
        """Slew to the safe park position — elevation straight up, azimuth 0."""
        await self.goto(0.0, 90.0)

    async def stop(self) -> None:
        """Halt motion immediately: GRBL feed-hold, then a soft reset."""
        await self._grbl.feed_hold()
        await self._grbl.soft_reset()

    async def position(self) -> tuple[float, float]:
        """Return the controller's current ``(azimuth, elevation)`` reading."""
        status = await self._grbl.query_status()
        return self._az_el(status)

    async def stream_position(self) -> AsyncIterator[tuple[float, float]]:
        """Yield ``(az, el)`` while a slew is in progress; stop once Idle."""
        while True:
            status = await self._grbl.query_status()
            yield self._az_el(status)
            if status.is_idle:
                return
            await asyncio.sleep(_POLL_S)

    async def calibrate(self) -> CalibrationReport:
        """Run the homing cycle and push the cable-management soft limits.

        Homing (``$H``) drives both axes to their limit switches and zeros the
        encoders. The soft limits are then written to GRBL ``$130``/``$131`` and
        soft limits are enabled (``$20=1``) so the firmware itself refuses a
        cable-wrapping move.
        """
        try:
            await self._grbl.home()
            await self._wait_until_idle(timeout_s=_SETTLE_TIMEOUT_S)
            # X (azimuth) and Y (elevation) max-travel soft limits.
            await self._grbl.set_setting(130, self.soft_limits_az[1])
            await self._grbl.set_setting(131, self.soft_limits_el[1])
            await self._grbl.set_setting(20, 1)  # enable soft limits
            self._homed = True
        except GrblConnectionError as exc:
            return CalibrationReport(
                ok=False,
                message=f"antrunner: homing failed — {exc}",
                backend="antrunner",
            )
        return CalibrationReport(
            ok=True,
            message="antrunner: homed; soft limits applied (AZ +/-180, EL 0-180)",
            backend="antrunner",
            details={
                "soft_limits_az": list(self.soft_limits_az),
                "soft_limits_el": list(self.soft_limits_el),
            },
        )

    async def status(self) -> dict[str, object]:
        """Return a GRBL health snapshot for ``rfdf hw selftest``."""
        try:
            grbl = await self._grbl.query_status()
        except GrblConnectionError as exc:
            return {"backend": "antrunner", "reachable": False, "error": str(exc)}
        return {
            "backend": "antrunner",
            "reachable": True,
            "state": grbl.state,
            "homed": self._homed,
            "machine_pos": list(grbl.machine_pos),
            "encoder_validation": self._encoder_validation,
        }

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._grbl.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate(self, az: float, el: float) -> None:
        """Reject a target outside the cable-management soft limits."""
        if not self.soft_limits_az[0] <= az <= self.soft_limits_az[1]:
            raise ValueError(f"AntRunner: azimuth {az} outside soft limits {self.soft_limits_az}")
        if not self.soft_limits_el[0] <= el <= self.soft_limits_el[1]:
            raise ValueError(f"AntRunner: elevation {el} outside soft limits {self.soft_limits_el}")

    @staticmethod
    def _az_el(status: GrblStatus) -> tuple[float, float]:
        """Extract ``(azimuth, elevation)`` from a GRBL status report."""
        pos = status.machine_pos
        az = pos[0] if len(pos) > 0 else 0.0
        el = pos[1] if len(pos) > 1 else 0.0
        return (az, el)

    async def _wait_until_idle(self, timeout_s: float = _SETTLE_TIMEOUT_S) -> None:
        """Poll the controller until it reports Idle, or raise on timeout/Alarm."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            status = await self._grbl.query_status()
            if status.is_alarm:
                raise AntRunnerError("AntRunner is in an Alarm state — re-home with calibrate().")
            if status.is_idle:
                return
            await asyncio.sleep(_POLL_S)
        raise AntRunnerError(f"AntRunner slew did not settle within {timeout_s:.0f} s.")


def create(
    *,
    host: str | None = None,
    port: int = 80,
    max_speed_deg_per_s: float = 6.0,
    homing_required_on_startup: bool = True,
    encoder_validation: bool = True,
    positioning_accuracy_deg: float = 0.1,
    command_path: str = "/command?commandText={cmd}",
    **_: Any,
) -> AntRunnerRotator:
    """Factory wired into the ``rfdf.backends.rotator`` ``antrunner`` entry-point.

    Args:
        host: Controller IP/hostname. Required — site-specific, kept in the
            operator's ``~/.config/rfdf/config.toml``, never committed.
        port: HTTP port.
        max_speed_deg_per_s: Reported slew speed.
        homing_required_on_startup: Require ``calibrate()`` before motion.
        encoder_validation: Validate closed-loop readback after every move.
        positioning_accuracy_deg: Closed-loop agreement tolerance.
        command_path: GRBL_ESP32 command-endpoint template.

    Returns:
        A configured :class:`AntRunnerRotator`.
    """
    if not host:
        raise AntRunnerError(
            "antrunner: host is required — set [rotator] host in ~/.config/rfdf/config.toml."
        )
    return AntRunnerRotator(
        host,
        port=port,
        max_speed_deg_per_s=max_speed_deg_per_s,
        homing_required_on_startup=homing_required_on_startup,
        encoder_validation=encoder_validation,
        positioning_accuracy_deg=positioning_accuracy_deg,
        command_path=command_path,
    )


__all__ = [
    "AntRunnerError",
    "AntRunnerRotator",
    "RotatorNotHomedError",
    "RotatorPositionError",
    "create",
]

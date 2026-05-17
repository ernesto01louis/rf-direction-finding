"""Shared GRBL-over-HTTP client for GRBL_ESP32 controllers.

Both the AntRunner AZ/EL rotator and the GRBL linear-rail geometry backend run
the GRBL_ESP32 firmware, which exposes a CNC G-code dialect over an HTTP command
endpoint (plus a WebSocket for high-rate streaming — not used here; the one-shot
HTTP API is simpler and stateless).

The status-report and settings-dump *parsers* are pure functions, unit-tested
without a controller. The :class:`GrblHttpClient` HTTP calls need a real
controller and are exercised by the ``hardware``-marked suites.

``httpx`` is imported **lazily** (:func:`_require_httpx`) so ``import rfdf`` —
and base backend discovery — works without the rotator/geometry extras.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

# GRBL 1.1 status report: "<State|MPos:0.000,0.000,0.000|FS:0,0|...>".
_STATUS_RE = re.compile(r"<([^>]*)>")
# Settings dump line: "$130=180.000".
_SETTING_RE = re.compile(r"^\$(\d+)\s*=\s*([-+0-9.]+)")

#: GRBL realtime command bytes.
GRBL_STATUS_QUERY = "?"
GRBL_FEED_HOLD = "!"
GRBL_RESUME = "~"
GRBL_SOFT_RESET = "\x18"


class GrblError(RuntimeError):
    """Base class for every error raised by the GRBL client."""


class GrblNotInstalledError(GrblError):
    """The ``httpx`` HTTP client is not importable — the backend extra is missing."""


class GrblConnectionError(GrblError):
    """The GRBL controller is unreachable or returned an unparseable response."""


class GrblAlarmError(GrblError):
    """The GRBL controller is in an Alarm state — homing or a soft reset is required."""


@dataclass(frozen=True)
class GrblStatus:
    """A parsed GRBL 1.1 status report.

    Attributes:
        state: Controller state — ``Idle`` / ``Run`` / ``Hold`` / ``Home`` /
            ``Alarm`` / ``Jog`` / ``Door`` / ``Check``.
        machine_pos: Machine coordinates per axis (the ``MPos`` field).
        work_pos: Work coordinates per axis (the ``WPos`` field), if present.
        raw: The raw report text, for diagnostics.
    """

    state: str
    machine_pos: tuple[float, ...]
    work_pos: tuple[float, ...] = ()
    raw: str = ""

    @property
    def is_idle(self) -> bool:
        """True when the controller has settled (no motion in flight)."""
        return self.state == "Idle"

    @property
    def is_alarm(self) -> bool:
        """True when the controller is latched in an Alarm state."""
        return self.state == "Alarm"


def parse_status_report(text: str) -> GrblStatus:
    """Parse a GRBL 1.1 ``?`` status report.

    Args:
        text: Raw controller output containing a ``<...>`` report.

    Returns:
        The parsed :class:`GrblStatus`.

    Raises:
        GrblConnectionError: If no ``<...>`` report is found in ``text``.
    """
    match = _STATUS_RE.search(text)
    if match is None:
        raise GrblConnectionError(f"no GRBL status report in response: {text!r}")
    body = match.group(1)
    fields = body.split("|")
    state = fields[0].strip()
    machine_pos: tuple[float, ...] = ()
    work_pos: tuple[float, ...] = ()
    for fld in fields[1:]:
        key, _, value = fld.partition(":")
        if key == "MPos":
            machine_pos = tuple(float(v) for v in value.split(",") if v)
        elif key == "WPos":
            work_pos = tuple(float(v) for v in value.split(",") if v)
    return GrblStatus(state=state, machine_pos=machine_pos, work_pos=work_pos, raw=text)


def parse_settings_dump(text: str) -> dict[int, float]:
    """Parse a GRBL ``$$`` settings dump into ``{setting_number: value}``.

    Args:
        text: Raw controller output, one ``$N=value`` per line.

    Returns:
        A mapping of GRBL setting number to its float value.
    """
    settings: dict[int, float] = {}
    for line in text.splitlines():
        match = _SETTING_RE.match(line.strip())
        if match is not None:
            settings[int(match.group(1))] = float(match.group(2))
    return settings


def _require_httpx() -> Any:
    """Import and return ``httpx``, or raise a clear install hint."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise GrblNotInstalledError(
            "GRBL backends require the httpx HTTP client; install the relevant "
            "extra (rfdf[rotator-antrunner] or rfdf[geometry-grbl])."
        ) from exc
    return httpx


@dataclass
class GrblHttpClient:
    """Minimal HTTP client for a GRBL_ESP32 controller.

    GRBL_ESP32 accepts G-code and realtime bytes via an HTTP command endpoint.
    The endpoint path varies across firmware forks; ``command_path`` is the
    template (``{cmd}`` is replaced with the URL-encoded command).

    Args:
        host: IP or hostname of the ESP32 controller.
        port: HTTP port (default 80).
        command_path: Command-endpoint template; ``{cmd}`` is substituted.
        timeout_s: Per-request HTTP timeout.
    """

    host: str
    port: int = 80
    command_path: str = "/command?commandText={cmd}"
    timeout_s: float = 5.0
    _client: Any = field(default=None, repr=False, compare=False)

    @property
    def base_url(self) -> str:
        """The ``http://host:port`` base URL."""
        return f"http://{self.host}:{self.port}"

    def _ensure_client(self) -> Any:
        """Lazily build the shared ``httpx.AsyncClient``."""
        if self._client is None:
            httpx = _require_httpx()
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    async def send(self, command: str) -> str:
        """Send a G-code line or realtime byte; return the controller's response.

        Raises:
            GrblConnectionError: If the controller is unreachable or errors.
        """
        httpx = _require_httpx()
        client = self._ensure_client()
        url = self.command_path.format(cmd=quote(command, safe=""))
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GrblConnectionError(
                f"GRBL controller {self.base_url} unreachable or errored: {exc}"
            ) from exc
        return str(response.text)

    async def query_status(self) -> GrblStatus:
        """Send ``?`` and parse the status report."""
        return parse_status_report(await self.send(GRBL_STATUS_QUERY))

    async def home(self) -> None:
        """Run the GRBL homing cycle (``$H``)."""
        await self.send("$H")

    async def feed_hold(self) -> None:
        """Issue a realtime feed-hold (``!``) — decelerate and pause."""
        await self.send(GRBL_FEED_HOLD)

    async def resume(self) -> None:
        """Issue a realtime cycle-resume (``~``)."""
        await self.send(GRBL_RESUME)

    async def soft_reset(self) -> None:
        """Issue a realtime soft reset (Ctrl-X)."""
        await self.send(GRBL_SOFT_RESET)

    async def dump_settings(self) -> dict[int, float]:
        """Send ``$$`` and parse the settings dump."""
        return parse_settings_dump(await self.send("$$"))

    async def set_setting(self, number: int, value: float) -> None:
        """Write a single GRBL ``$N=value`` setting."""
        await self.send(f"${number}={value}")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = [
    "GRBL_FEED_HOLD",
    "GRBL_RESUME",
    "GRBL_SOFT_RESET",
    "GRBL_STATUS_QUERY",
    "GrblAlarmError",
    "GrblConnectionError",
    "GrblError",
    "GrblHttpClient",
    "GrblNotInstalledError",
    "GrblStatus",
    "parse_settings_dump",
    "parse_status_report",
]

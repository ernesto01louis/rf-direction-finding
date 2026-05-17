r"""A minimal Hamlib ``rotctld`` TCP server backed by a ``RotatorController``.

Gpredict and other amateur-satellite trackers speak the Hamlib network-rotator
protocol. Exposing one lets any rfdf-supported rotator be driven by that whole
ecosystem. The server is optional — started with ``rfdf hw rotator-server``.

Protocol subset (enough for Gpredict):

* ``p`` / ``get_pos``      -> reply ``<az>\n<el>\n``
* ``P <az> <el>`` / ``set_pos`` -> slew, reply ``RPRT 0``
* ``S`` / ``stop``         -> halt, reply ``RPRT 0``
* ``q`` / ``Q``            -> close the connection

Any error replies ``RPRT -<errno>`` (the Hamlib convention).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfdf.hal import RotatorController

_log = logging.getLogger(__name__)

#: Default rotctld port (the Hamlib convention).
DEFAULT_ROTCTLD_PORT = 4533


class RotctldServer:
    """Serves the Hamlib rotctld protocol for one :class:`RotatorController`.

    Args:
        rotator: The backend the protocol drives.
        host: Bind address.
        port: Bind port (4533 by the Hamlib convention).
    """

    def __init__(
        self,
        rotator: RotatorController,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_ROTCTLD_PORT,
    ) -> None:
        """Capture the rotator + bind address."""
        self._rotator = rotator
        self._host = host
        self._port = port

    async def serve(self) -> None:
        """Run the server until cancelled."""
        server = await asyncio.start_server(self._handle, self._host, self._port)
        addr = ", ".join(str(s.getsockname()) for s in server.sockets)
        _log.info("rotctld serving on %s", addr)
        async with server:
            await server.serve_forever()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle one client connection."""
        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                reply = await self._dispatch(raw.decode("ascii", "replace").strip())
                if reply is None:
                    break
                writer.write(reply.encode("ascii"))
                await writer.drain()
        finally:
            writer.close()

    async def _dispatch(self, line: str) -> str | None:
        """Map one protocol line to a reply; ``None`` closes the connection."""
        if not line:
            return "RPRT 0\n"
        cmd, *args = line.split()
        if cmd in {"q", "Q"}:
            return None
        if cmd in {"p", "get_pos", "\\get_pos"}:
            try:
                az, el = await self._rotator.position()
            except Exception as exc:  # report to the client, keep serving
                _log.warning("rotctld get_pos failed: %s", exc)
                return "RPRT -1\n"
            return f"{az:.6f}\n{el:.6f}\n"
        if cmd in {"P", "set_pos", "\\set_pos"}:
            return await self._set_pos(args)
        if cmd in {"S", "stop", "\\stop"}:
            try:
                await self._rotator.stop()
            except Exception as exc:
                _log.warning("rotctld stop failed: %s", exc)
                return "RPRT -1\n"
            return "RPRT 0\n"
        # Unknown command — Hamlib's "not implemented" errno.
        return "RPRT -11\n"

    async def _set_pos(self, args: list[str]) -> str:
        """Handle a ``P <az> <el>`` set-position command."""
        if len(args) != 2:
            return "RPRT -1\n"
        try:
            az, el = float(args[0]), float(args[1])
        except ValueError:
            return "RPRT -1\n"
        try:
            await self._rotator.goto(az, el)
        except ValueError as exc:  # out of range
            _log.warning("rotctld set_pos rejected: %s", exc)
            return "RPRT -1\n"
        except Exception as exc:
            _log.warning("rotctld set_pos failed: %s", exc)
            return "RPRT -1\n"
        return "RPRT 0\n"


__all__ = ["DEFAULT_ROTCTLD_PORT", "RotctldServer"]

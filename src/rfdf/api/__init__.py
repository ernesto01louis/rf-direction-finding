"""Optional rfdf REST API (Stage 7 R7).

The API is **standalone-first** and **optional**: it ships behind the
``[api]`` extra. Importing :mod:`rfdf.api` never raises — FastAPI is
imported lazily inside :func:`create_app`.

The same app is two things at once:

* the standalone rfdf HTTP surface — ``GET /healthz``, ``GET /``,
  ``GET /capabilities``;
* the orchestrator's **capability callback target** — ``POST
  /capabilities/{capability}`` dispatches to :class:`RfdfConsumer`. The
  capability routes mount only when the ``[orchestrator]`` extra is
  installed, so a pure ``[api]`` install still serves ``/healthz``.

Serve it with the factory::

    uvicorn --factory rfdf.api:create_app

or via the CLI ``rfdf api serve``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

__all__ = ["create_app"]


def create_app() -> FastAPI:
    """Build and return the rfdf FastAPI application.

    Importing this module is safe without the ``[api]`` extra; *calling*
    ``create_app`` requires it (FastAPI is imported here).
    """
    from rfdf.api.app import build_app

    return build_app()

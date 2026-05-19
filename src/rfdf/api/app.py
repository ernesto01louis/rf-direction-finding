"""rfdf FastAPI application — health, discovery, and capability dispatch.

``build_app`` constructs the app; :func:`rfdf.api.create_app` is the
public factory. FastAPI is imported here, so this module is only
imported when the ``[api]`` extra is installed.

The capability routes (``POST /capabilities/{capability}``) are the
orchestrator's callback target — they mount only when the
``[orchestrator]`` extra is present. With ``RFDF_API_TOKEN`` set in the
environment, those routes require a matching ``Authorization: Bearer``
header (the token the orchestrator stored at registration).
``GET /healthz`` is always unauthenticated so infra liveness probes
work regardless.
"""

from __future__ import annotations

import os
import secrets
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from rfdf import __version__


def _check_token(authorization: str | None) -> None:
    """Enforce the bearer token on capability routes when one is set."""
    token = os.environ.get("RFDF_API_TOKEN", "").strip()
    if not token:
        return  # auth disabled — open (development default)
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    if not presented or not secrets.compare_digest(presented, token):
        raise HTTPException(status_code=401, detail="invalid or missing token")


def build_app() -> FastAPI:
    """Construct the rfdf FastAPI app."""
    app = FastAPI(
        title="rfdf",
        version=__version__,
        summary="Hardware-agnostic RF direction finding — REST surface.",
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness probe — always unauthenticated."""
        return {"status": "ok", "service": "rfdf", "version": __version__}

    @app.get("/")
    def root() -> dict[str, Any]:
        """Service banner."""
        return {
            "service": "rfdf",
            "version": __version__,
            "docs": "/docs",
            "healthz": "/healthz",
        }

    _mount_capabilities(app)
    return app


def _mount_capabilities(app: FastAPI) -> None:
    """Mount the capability routes when the [orchestrator] extra is present.

    Without ``ai-orchestrator-client`` the standalone API still serves
    ``/healthz`` and ``/`` — ``/capabilities`` simply reports that
    dispatch is unavailable.
    """
    from rfdf.orchestrator import is_available

    if not is_available():

        @app.get("/capabilities")
        def capabilities_unavailable() -> dict[str, Any]:
            return {
                "capabilities": [],
                "dispatch": "unavailable — install rfdf[orchestrator]",
            }

        return

    from ai_orchestrator_client import UnknownCapabilityError

    from rfdf.orchestrator import RfdfConsumer

    consumer = RfdfConsumer()

    @app.get("/capabilities")
    def list_capabilities() -> dict[str, Any]:
        """List the capabilities this consumer can be dispatched."""
        return {"capabilities": consumer.capabilities, "dispatch": "ready"}

    @app.post("/capabilities/{capability}")
    async def invoke_capability(
        capability: str,
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Dispatch a capability call — the orchestrator's callback target.

        ``payload`` is the JSON request body — the orchestrator's
        dispatch always sends one (``{}`` when empty).
        """
        _check_token(authorization)
        try:
            result = await consumer.dispatch(capability, payload)
        except UnknownCapabilityError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"missing payload key: {exc}") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"capability": capability, "result": result}

"""``rfdf hw`` CLI subcommands.

Two subcommands:

* ``rfdf hw list-backends`` — JSON dump of every entry-point-registered backend.
* ``rfdf hw selftest`` — exercises each configured backend for ~0.1 s and emits
  a JSON status report. Exit code 0 when every backend reports ``ok=true``,
  exit 1 on any failure.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import typer

from rfdf.config import load_config
from rfdf.hal import (
    SdrConfig,
    list_backends,
    load_backend,
)
from rfdf.hal.compute import ComputeJob

hw_app = typer.Typer(
    name="hw",
    help="Inspect + exercise hardware abstraction layer backends.",
    no_args_is_help=True,
    add_completion=False,
)


@hw_app.command("list-backends")
def list_backends_cmd() -> None:
    """Print the catalog of installed backends as JSON."""
    typer.echo(json.dumps(list_backends(), indent=2))


@hw_app.command("selftest")
def selftest() -> None:
    """Exercise each configured backend and emit a JSON status report."""
    cfg = load_config()
    report: dict[str, Any] = {}
    overall_ok = True

    _check_geometry(cfg, report)
    overall_ok &= report["geometry"]["ok"]
    _check_rotator(cfg, report)
    overall_ok &= report["rotator"]["ok"]
    _check_sdr(cfg, report)
    overall_ok &= report["sdr"]["ok"]
    _check_compute(cfg, report)
    overall_ok &= report["compute"]["ok"]

    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(code=0 if overall_ok else 1)


# ---------------------------------------------------------------------------
# Per-backend self-checks
# ---------------------------------------------------------------------------


def _safe_check(name: str, fn: Any) -> dict[str, Any]:
    """Run ``fn``, capturing exception / timing in a uniform dict."""
    start = time.monotonic()
    try:
        fn()
        return {
            "name": name,
            "ok": True,
            "latency_ms": (time.monotonic() - start) * 1000.0,
            "error_msg": None,
        }
    except Exception as exc:  # report all failures; CI sees the exit code
        return {
            "name": name,
            "ok": False,
            "latency_ms": (time.monotonic() - start) * 1000.0,
            "error_msg": f"{type(exc).__name__}: {exc}",
        }


def _check_geometry(cfg: Any, report: dict[str, Any]) -> None:
    """Load the configured geometry backend + read positions."""
    name = cfg.geometry.backend

    async def run() -> None:
        kwargs: dict[str, Any] = {}
        if name == "static":
            kwargs["antennas"] = cfg.geometry.antennas
        elif name == "mock-morph":
            kwargs["initial_positions"] = cfg.geometry.antennas
        geo = load_backend("rfdf.backends.geometry", name, **kwargs)
        positions = await geo.positions()
        assert positions.shape[1] == 3, "geometry positions must be (N, 3)"

    report["geometry"] = _safe_check(name, lambda: asyncio.run(run()))


def _check_rotator(cfg: Any, report: dict[str, Any]) -> None:
    """Load the configured rotator backend + do a no-op goto."""
    name = cfg.rotator.backend

    async def run() -> None:
        rot = load_backend("rfdf.backends.rotator", name)
        await rot.goto(0.0, 0.0)

    report["rotator"] = _safe_check(name, lambda: asyncio.run(run()))


def _check_sdr(cfg: Any, report: dict[str, Any]) -> None:
    """Load the configured SDR backend + capture a few samples."""
    name = cfg.sdr.backend

    async def run() -> None:
        kwargs: dict[str, Any] = {}
        if name == "mock":
            geo_name = cfg.geometry.backend
            geo_kwargs: dict[str, Any] = {}
            if geo_name == "static":
                geo_kwargs["antennas"] = cfg.geometry.antennas
            elif geo_name == "mock-morph":
                geo_kwargs["initial_positions"] = cfg.geometry.antennas
            kwargs["geometry"] = load_backend("rfdf.backends.geometry", geo_name, **geo_kwargs)
        sdr = load_backend("rfdf.backends.sdr", name, **kwargs)
        sdr_cfg = SdrConfig(
            center_freq_hz=cfg.sdr.center_freq_hz,
            sample_rate_hz=cfg.sdr.sample_rate_hz,
            rx_gain_db=cfg.sdr.rx_gain_db,
        )
        await sdr.configure(sdr_cfg)
        await sdr.start()
        async for block in sdr.stream():
            assert block.iq.shape[0] >= 1
            await sdr.stop()
            return

    report["sdr"] = _safe_check(name, lambda: asyncio.run(run()))


def _check_compute(cfg: Any, report: dict[str, Any]) -> None:
    """Load the configured compute backend + submit a trivial no-op job."""
    name = cfg.compute.backend

    async def run() -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        backend = load_backend("rfdf.backends.compute", name)
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "noop.py").write_text("print('selftest ok')\n")
            job = ComputeJob(entry_script="noop.py", working_dir=work)
            handle = await backend.submit(job)
            for _ in range(200):
                status = await backend.status(handle)
                if str(status) in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.02)
            assert str(status) == "completed", f"compute selftest finished with {status}"

    report["compute"] = _safe_check(name, lambda: asyncio.run(run()))

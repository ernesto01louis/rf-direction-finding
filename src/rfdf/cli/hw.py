"""``rfdf hw`` CLI subcommands.

* ``rfdf hw list-backends`` — JSON dump of every entry-point-registered backend.
* ``rfdf hw selftest`` — exercises each configured backend for ~0.1 s and emits
  a JSON status report. Exit code 0 when every backend reports ``ok=true``,
  exit 1 on any failure.
* ``rfdf hw udev {list,generate,install}`` — generate + install the udev rules
  that let a non-root process open a USB SDR.
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
from rfdf.hw import selftest as selftest_mod
from rfdf.hw import udev as udev_mod
from rfdf.hw.rotctld import DEFAULT_ROTCTLD_PORT, RotctldServer

hw_app = typer.Typer(
    name="hw",
    help="Inspect + exercise hardware abstraction layer backends.",
    no_args_is_help=True,
    add_completion=False,
)

udev_app = typer.Typer(
    name="udev",
    help="Generate + install udev rules for supported SDR hardware.",
    no_args_is_help=True,
    add_completion=False,
)
hw_app.add_typer(udev_app, name="udev")

geometry_app = typer.Typer(
    name="geometry",
    help="Inspect + drive the configured array-geometry backend.",
    no_args_is_help=True,
    add_completion=False,
)
hw_app.add_typer(geometry_app, name="geometry")

rotator_app = typer.Typer(
    name="rotator",
    help="Inspect + drive the configured rotator backend.",
    no_args_is_help=True,
    add_completion=False,
)
hw_app.add_typer(rotator_app, name="rotator")


def _geometry_kwargs(cfg: Any) -> dict[str, Any]:
    """Build the load_backend kwargs for the configured geometry backend."""
    name = cfg.geometry.backend
    if name == "static":
        return {"antennas": cfg.geometry.antennas}
    if name == "mock-morph":
        return {"initial_positions": cfg.geometry.antennas}
    return {}


@geometry_app.command("list-presets")
def geometry_list_presets() -> None:
    """List the named geometry presets the configured backend knows."""
    cfg = load_config()
    geo = load_backend("rfdf.backends.geometry", cfg.geometry.backend, **_geometry_kwargs(cfg))
    presets = asyncio.run(geo.list_presets())
    typer.echo(json.dumps(presets, indent=2))


@geometry_app.command("goto")
def geometry_goto(preset: str = typer.Argument(..., help="Preset name to move to.")) -> None:
    """Move the array to a named geometry preset."""
    cfg = load_config()
    geo = load_backend("rfdf.backends.geometry", cfg.geometry.backend, **_geometry_kwargs(cfg))
    try:
        asyncio.run(geo.goto_preset(preset))
    except Exception as exc:  # surface a clean message, non-zero exit
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"geometry: moved to preset {preset!r}")


@rotator_app.command("status")
def rotator_status() -> None:
    """Print the configured rotator's current position."""
    cfg = load_config()
    rot = load_backend("rfdf.backends.rotator", cfg.rotator.backend)
    try:
        az, el = asyncio.run(rot.position())
    except Exception as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"azimuth_deg": az, "elevation_deg": el}, indent=2))


@rotator_app.command("goto")
def rotator_goto(
    azimuth: float = typer.Argument(..., help="Target azimuth in degrees."),
    elevation: float = typer.Argument(..., help="Target elevation in degrees."),
) -> None:
    """Slew the configured rotator to (azimuth, elevation)."""
    cfg = load_config()
    rot = load_backend("rfdf.backends.rotator", cfg.rotator.backend)
    try:
        asyncio.run(rot.goto(azimuth, elevation))
    except Exception as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"rotator: slewed to ({azimuth:.2f}, {elevation:.2f})")


@rotator_app.command("park")
def rotator_park() -> None:
    """Park the configured rotator at its safe storage position."""
    cfg = load_config()
    rot = load_backend("rfdf.backends.rotator", cfg.rotator.backend)
    try:
        asyncio.run(rot.park())
    except Exception as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("rotator: parked")


@hw_app.command("rotator-server")
def rotator_server(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(DEFAULT_ROTCTLD_PORT, "--port", help="Bind port."),
) -> None:
    """Serve the Hamlib rotctld protocol for the configured rotator.

    Lets Gpredict and other amateur-satellite trackers point the rotator. Runs
    until interrupted.
    """
    cfg = load_config()
    rot = load_backend("rfdf.backends.rotator", cfg.rotator.backend)
    server = RotctldServer(rot, host=host, port=port)
    typer.echo(f"rotctld: serving {cfg.rotator.backend} rotator on {host}:{port} (Ctrl-C to stop)")
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:  # pragma: no cover - interactive
        typer.echo("rotctld: stopped")


@udev_app.command("list")
def udev_list() -> None:
    """List the devices rfdf ships udev rules for."""
    for rule in udev_mod.KNOWN_DEVICES:
        typer.echo(f"{rule.vendor_id}:{rule.product_id}  {rule.symlink:<14} {rule.description}")


@udev_app.command("generate")
def udev_generate() -> None:
    """Print the generated udev rules file to stdout."""
    typer.echo(udev_mod.render_rules_file(), nl=False)


@udev_app.command("install")
def udev_install(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Install the udev rules to /etc/udev/rules.d and reload udev (needs root)."""
    path = udev_mod.DEFAULT_RULES_PATH
    if not udev_mod.is_root():
        typer.echo(
            f"error: writing {path} requires root — re-run with sudo:\n  sudo rfdf hw udev install",
            err=True,
        )
        raise typer.Exit(code=1)
    if not yes and not typer.confirm(f"Install udev rules to {path} and reload udev?"):
        typer.echo("aborted.")
        raise typer.Exit(code=1)
    try:
        summary = udev_mod.install_rules(udev_mod.render_rules_file(), path=path)
    except PermissionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(summary)


@hw_app.command("list-backends")
def list_backends_cmd() -> None:
    """Print the catalog of installed backends as JSON."""
    typer.echo(json.dumps(list_backends(), indent=2))


@hw_app.command("selftest")
def selftest(
    fmt: str = typer.Option(
        "human", "--format", help="Output format: 'human' (colour tree) or 'json'."
    ),
) -> None:
    """Exercise each configured backend and report HAL-contract + smoke status.

    Runs the Stage-2 HAL contract exercise against every configured backend
    plus a device ``status()`` probe. Exit 0 when every backend is healthy,
    exit 1 on any failure.
    """
    if fmt not in {"human", "json"}:
        typer.echo(f"error: --format must be 'human' or 'json', got {fmt!r}", err=True)
        raise typer.Exit(code=2)
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

    if fmt == "json":
        typer.echo(selftest_mod.format_json(report))
    else:
        typer.echo(selftest_mod.format_human(report))
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

"""``rfdf doa`` CLI sub-commands: run, calibrate, benchmark, and morph-capture.

Every command operates on the mock SDR backend, so the full DOA pipeline is exercisable
with no hardware — the platform's load-bearing principle made interactive.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from rfdf.backends.geometry.static import create as create_static_geometry
from rfdf.backends.sdr.mock import CWEmitter, Emitter, MockSdrScenario
from rfdf.backends.sdr.mock import create as create_mock_sdr
from rfdf.dsp.calibration import calibrate_pilot_tone
from rfdf.dsp.covariance import sample_covariance
from rfdf.dsp.crlb import compute_crlb
from rfdf.dsp.doa.bartlett import bartlett
from rfdf.dsp.doa.esprit import esprit
from rfdf.dsp.doa.music import music
from rfdf.dsp.doa.mvdr import mvdr
from rfdf.dsp.doa.orchestration import Doa
from rfdf.dsp.doa.result import Algorithm, DoaEstimate
from rfdf.dsp.doa.root_music import root_music
from rfdf.dsp.geometry_presets import half_wavelength_spacing, ula
from rfdf.dsp.steering import build_manifold, steering_vector
from rfdf.hal import SdrConfig
from rfdf.hal.sdr import SdrSource

doa_app = typer.Typer(
    name="doa",
    help="Classical direction-of-arrival estimation on the mock SDR.",
    no_args_is_help=True,
)
_console = Console()
_SAMPLE_RATE_HZ = 2.0e6
_CWD = Path(".")


def _parse_algorithm(name: str) -> Algorithm:
    """Coerce a CLI algorithm string to an :class:`Algorithm`, or raise cleanly."""
    try:
        return Algorithm(name)
    except ValueError as exc:
        choices = ", ".join(member.value for member in Algorithm)
        raise typer.BadParameter(f"unknown algorithm {name!r}; choose from: {choices}") from exc


async def _capture_block(sdr: SdrSource) -> np.ndarray:
    """Start the SDR, return one StreamBlock's IQ, then stop."""
    await sdr.start()
    async for block in sdr.stream():
        await sdr.stop()
        return block.iq.astype(np.complex128)
    raise RuntimeError("capture produced no IQ")


def _simulate_covariance(
    positions: np.ndarray,
    *,
    azimuth_deg: float,
    freq_hz: float,
    snr_db: float,
    snapshots: int,
    seed: int,
) -> np.ndarray:
    """Synthesise a single-source ULA covariance for the benchmark sweep."""
    rng = np.random.default_rng(seed)
    response = steering_vector(positions, az_deg=azimuth_deg, el_deg=0.0, freq_hz=freq_hz)
    source = (rng.standard_normal(snapshots) + 1j * rng.standard_normal(snapshots)) / np.sqrt(2.0)
    signal = np.outer(response, source)
    noise_std = np.sqrt(1.0 / (10.0 ** (snr_db / 10.0)) / 2.0)
    noise = noise_std * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    return sample_covariance(signal + noise)


@doa_app.command("run")
def run(
    algorithm: Annotated[str, typer.Option(help="DOA estimator to use.")] = "music",
    azimuths: Annotated[
        str, typer.Option(help="Comma-separated emitter azimuths to simulate.")
    ] = "30",
    duration: Annotated[float, typer.Option(help="Capture duration in seconds.")] = 0.05,
    num_signals: Annotated[int, typer.Option(help="Source count; 0 auto-estimates via MDL.")] = 0,
    freq_hz: Annotated[float, typer.Option(help="RF centre frequency in hertz.")] = 2.4e9,
    snr_db: Annotated[float, typer.Option(help="Scenario signal-to-noise ratio in dB.")] = 20.0,
    channels: Annotated[int, typer.Option(help="Number of ULA antennas.")] = 8,
) -> None:
    """Estimate direction of arrival on a synthetic mock-SDR scenario."""
    estimator = _parse_algorithm(algorithm)
    truths = [float(value) for value in azimuths.split(",")]
    positions = ula(channels, half_wavelength_spacing(freq_hz))
    geometry = create_static_geometry(antennas=positions.tolist())
    emitters: list[Emitter] = [
        CWEmitter(azimuth_deg=az, elevation_deg=0.0, power_dbm=0.0, freq_offset_hz=5e3 * index)
        for index, az in enumerate(truths)
    ]
    sdr = create_mock_sdr(
        geometry=geometry, scenario=MockSdrScenario(emitters=emitters, snr_db=snr_db)
    )
    asyncio.run(
        sdr.configure(
            SdrConfig(
                center_freq_hz=freq_hz,
                sample_rate_hz=_SAMPLE_RATE_HZ,
                channels=list(range(channels)),
            )
        )
    )
    doa = Doa(sdr, geometry, algorithm=estimator)
    estimate = asyncio.run(doa.run(duration_s=duration, num_signals=num_signals or None))

    table = Table(title=f"DOA estimate ({estimate.algorithm})")
    table.add_column("source")
    table.add_column("azimuth (deg)", justify="right")
    for index, az in enumerate(sorted(estimate.azimuth_deg)):
        table.add_row(str(index + 1), f"{az:.2f}")
    _console.print(table)


@doa_app.command("calibrate")
def calibrate(
    name: Annotated[str, typer.Option(help="Calibration name (filename stem).")],
    directory: Annotated[
        Path, typer.Option(help="Directory to write the calibration into.")
    ] = _CWD,
    freq_hz: Annotated[float, typer.Option(help="Pilot frequency in hertz.")] = 2.4e9,
    channels: Annotated[int, typer.Option(help="Number of ULA antennas.")] = 8,
) -> None:
    """Run pilot-tone calibration against a mock SDR and save the result."""
    positions = ula(channels, half_wavelength_spacing(freq_hz))
    geometry = create_static_geometry(antennas=positions.tolist())
    sdr = create_mock_sdr(
        geometry=geometry,
        mock_b210_behavior=True,
        gain_errors_db=list(np.linspace(-3.0, 3.0, channels)),
    )
    asyncio.run(
        sdr.configure(
            SdrConfig(
                center_freq_hz=freq_hz,
                sample_rate_hz=_SAMPLE_RATE_HZ,
                channels=list(range(channels)),
            )
        )
    )
    calibration = asyncio.run(calibrate_pilot_tone(sdr, freq_hz=freq_hz, duration_s=0.05))
    saved = calibration.save(name, directory=directory)
    _console.print(f"calibration '{name}' saved to {saved}")


@doa_app.command("benchmark")
def benchmark(
    algorithms: Annotated[
        str, typer.Option(help="Comma-separated estimators to benchmark.")
    ] = "music,bartlett,mvdr,esprit,root_music",
    snr_range: Annotated[
        str, typer.Option(help="start:stop:step in dB (stop inclusive).")
    ] = "-5:25:10",
    trials: Annotated[int, typer.Option(help="Monte-Carlo trials per (algorithm, SNR).")] = 30,
    output: Annotated[Path | None, typer.Option(help="Optional path for an HTML report.")] = None,
    channels: Annotated[int, typer.Option(help="Number of ULA antennas.")] = 8,
) -> None:
    """Benchmark DOA estimators across an SNR sweep against the Cramer-Rao bound."""
    freq_hz = 2.4e9
    truth_az = 65.0
    snapshots = 1000
    positions = ula(channels, half_wavelength_spacing(freq_hz))
    manifold = build_manifold(positions, np.arange(0.0, 180.0, 0.1), np.array([0.0]), freq_hz)
    grid_fns: dict[str, Callable[..., DoaEstimate]] = {
        "music": music,
        "bartlett": bartlett,
        "mvdr": mvdr,
    }
    parametric_fns: dict[str, Callable[..., DoaEstimate]] = {
        "esprit": esprit,
        "root_music": root_music,
    }

    start, stop, step = (float(piece) for piece in snr_range.split(":"))
    snrs = list(np.arange(start, stop + step / 2.0, step))
    chosen = [name.strip() for name in algorithms.split(",")]

    table = Table(title=f"DOA benchmark (truth {truth_az} deg, {snapshots} snapshots)")
    table.add_column("SNR (dB)", justify="right")
    table.add_column("sqrt(CRLB) (deg)", justify="right")
    for name in chosen:
        table.add_column(f"{name} RMSE", justify="right")

    rows: list[list[str]] = []
    for snr_db in snrs:
        crlb = compute_crlb(
            positions, freq_hz=freq_hz, snr_db=snr_db, snapshots=snapshots, direction_deg=truth_az
        )
        row = [f"{snr_db:.0f}", f"{np.sqrt(crlb):.4f}"]
        for name in chosen:
            errors = []
            for seed in range(trials):
                cov = _simulate_covariance(
                    positions,
                    azimuth_deg=truth_az,
                    freq_hz=freq_hz,
                    snr_db=snr_db,
                    snapshots=snapshots,
                    seed=seed,
                )
                if name in grid_fns:
                    estimate = grid_fns[name](cov, manifold, num_signals=1)
                else:
                    estimate = parametric_fns[name](
                        cov, positions=positions, freq_hz=freq_hz, num_signals=1
                    )
                errors.append(estimate.azimuth_deg[0] - truth_az)
            row.append(f"{float(np.std(errors)):.4f}")
        rows.append(row)
        table.add_row(*row)
    _console.print(table)

    if output is not None:
        header = ["SNR (dB)", "sqrt(CRLB) (deg)", *[f"{name} RMSE" for name in chosen]]
        body = "\n".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
        )
        head = "".join(f"<th>{cell}</th>" for cell in header)
        output.write_text(
            f"<html><body><h1>rfdf DOA benchmark</h1><table border='1'>"
            f"<tr>{head}</tr>{body}</table></body></html>",
            encoding="utf-8",
        )
        _console.print(f"HTML report written to {output}")


@doa_app.command("morph-capture")
def morph_capture(
    directory: Annotated[Path, typer.Option(help="Directory to write per-station captures into.")],
    stations: Annotated[int, typer.Option(help="Number of rail stations to capture.")] = 6,
    azimuth: Annotated[float, typer.Option(help="Simulated emitter azimuth in degrees.")] = 60.0,
    freq_hz: Annotated[float, typer.Option(help="RF centre frequency in hertz.")] = 5.8e9,
    rail_span_m: Annotated[float, typer.Option(help="Total rail length the stations span.")] = 1.5,
) -> None:
    """Capture IQ across morphing-array rail stations for synthetic-aperture fusion."""
    directory.mkdir(parents=True, exist_ok=True)
    base = ula(5, half_wavelength_spacing(freq_hz))
    rail = np.linspace(0.0, rail_span_m, stations)
    for index in range(stations):
        positions = base + np.array([rail[index], 0.0, 0.0])
        geometry = create_static_geometry(antennas=positions.tolist())
        scenario = MockSdrScenario(
            emitters=[CWEmitter(azimuth_deg=azimuth, elevation_deg=0.0, power_dbm=0.0)],
            snr_db=25.0,
        )
        sdr = create_mock_sdr(geometry=geometry, scenario=scenario, seed=index)
        asyncio.run(
            sdr.configure(
                SdrConfig(
                    center_freq_hz=freq_hz,
                    sample_rate_hz=_SAMPLE_RATE_HZ,
                    channels=list(range(5)),
                )
            )
        )
        iq = asyncio.run(_capture_block(sdr))
        np.savez(directory / f"station_{index}.npz", positions=positions, iq=iq)
    _console.print(f"captured {stations} stations into {directory}")

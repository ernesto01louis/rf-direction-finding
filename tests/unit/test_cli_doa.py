"""Unit tests for the ``rfdf doa`` CLI sub-commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rfdf.cli.main import app

runner = CliRunner()


def test_doa_help_lists_all_subcommands() -> None:
    """``rfdf doa --help`` advertises run, calibrate, benchmark, morph-capture."""
    result = runner.invoke(app, ["doa", "--help"])
    assert result.exit_code == 0
    for command in ("run", "calibrate", "benchmark", "morph-capture"):
        assert command in result.output


def test_doa_run_prints_an_estimate() -> None:
    """``rfdf doa run`` estimates DOA on a synthetic scenario and prints a table."""
    result = runner.invoke(
        app, ["doa", "run", "--algorithm", "music", "--azimuths", "65", "--duration", "0.02"]
    )
    assert result.exit_code == 0, result.output
    assert "estimate" in result.output.lower()


def test_doa_run_rejects_an_unknown_algorithm() -> None:
    """An unknown algorithm name is rejected with a non-zero exit code."""
    result = runner.invoke(app, ["doa", "run", "--algorithm", "bogus"])
    assert result.exit_code != 0


def test_doa_calibrate_saves_a_calibration(tmp_path: Path) -> None:
    """``rfdf doa calibrate`` writes a calibration npz + toml pair."""
    result = runner.invoke(
        app, ["doa", "calibrate", "--name", "cband", "--directory", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cband.npz").exists()
    assert (tmp_path / "cband.toml").exists()


def test_doa_benchmark_runs_and_writes_html(tmp_path: Path) -> None:
    """``rfdf doa benchmark`` sweeps SNR and writes an HTML report."""
    report = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "doa",
            "benchmark",
            "--algorithms",
            "music,esprit",
            "--snr-range",
            "10:20:10",
            "--trials",
            "10",
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert report.exists()
    assert "<table" in report.read_text()


def test_doa_morph_capture_writes_station_files(tmp_path: Path) -> None:
    """``rfdf doa morph-capture`` writes one npz capture per rail station."""
    result = runner.invoke(
        app, ["doa", "morph-capture", "--directory", str(tmp_path), "--stations", "3"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "station_0.npz").exists()
    assert (tmp_path / "station_2.npz").exists()

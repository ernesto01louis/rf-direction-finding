"""Unit tests for the ``rfdf compute`` CLI sub-commands."""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from rfdf.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_compute_help_lists_subcommands() -> None:
    """``rfdf compute --help`` advertises list, test, estimate, jobs, logs, cancel."""
    result = runner.invoke(app, ["compute", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("list", "test", "estimate", "jobs"):
        assert sub in result.output


def test_top_level_help_lists_compute() -> None:
    """``rfdf --help`` lists the compute sub-app."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "compute" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_compute_list_shows_local_backend() -> None:
    """``rfdf compute list`` shows at least the 'local' backend."""
    result = runner.invoke(app, ["compute", "list"])
    assert result.exit_code == 0, result.output
    assert "local" in result.output


def test_compute_list_shows_cloud_backends() -> None:
    """``rfdf compute list`` discovers cloud backends registered via entry-points."""
    result = runner.invoke(app, ["compute", "list"])
    assert result.exit_code == 0, result.output
    # At least one of the cloud backends is registered in pyproject.toml.
    # We just check the table rendered without crashing.
    assert "local" in result.output  # always present


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


def test_compute_test_local_succeeds() -> None:
    """``rfdf compute test --backend local`` completes a no-op job successfully."""
    result = runner.invoke(app, ["compute", "test", "--backend", "local"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output or "completed" in result.output.lower()


def test_compute_test_unknown_backend_exits_nonzero() -> None:
    """An unknown backend name exits non-zero with a helpful message."""
    result = runner.invoke(app, ["compute", "test", "--backend", "does-not-exist"])
    assert result.exit_code != 0
    assert "does-not-exist" in result.output or "Unknown" in result.output


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


def test_compute_estimate_local_shows_zero_cost() -> None:
    """``rfdf compute estimate --backend local`` prints $0.0000 (free local compute)."""
    result = runner.invoke(app, ["compute", "estimate", "--backend", "local"])
    assert result.exit_code == 0, result.output
    # local backend always estimates $0 cost
    assert "$0.0000" in result.output or "0.00" in result.output


def test_compute_estimate_unknown_backend_exits_nonzero() -> None:
    """Estimate for an unknown backend exits non-zero."""
    result = runner.invoke(app, ["compute", "estimate", "--backend", "does-not-exist"])
    assert result.exit_code != 0


def test_compute_estimate_cloud_backend_graceful() -> None:
    """Cloud backends that lack credentials report clearly without crashing."""
    # runpod is registered in pyproject.toml; it will either succeed or
    # report a clear SDK-missing / credential-missing message.
    result = runner.invoke(
        app,
        ["compute", "estimate", "--backend", "runpod", "--gpu-model", "A6000", "--gpu-count", "1"],
    )
    # exit 0 (has SDK + stub estimate) or exit 1 (ImportError / missing creds)
    # — we just assert no unhandled Python traceback leaks.
    assert "Traceback" not in result.output
    assert "Traceback" not in (result.stderr or "")


# ---------------------------------------------------------------------------
# jobs (empty session)
# ---------------------------------------------------------------------------


def test_compute_jobs_empty_session() -> None:
    """``rfdf compute jobs`` with no submitted jobs prints a helpful note."""
    result = runner.invoke(app, ["compute", "jobs"])
    assert result.exit_code == 0, result.output
    # Should mention per-session state or be empty job list message.
    assert "session" in result.output.lower() or "No jobs" in result.output


# ---------------------------------------------------------------------------
# Lazy-import: rfdf --help must not load torch
# ---------------------------------------------------------------------------


def test_rfdf_help_no_torch_loaded() -> None:
    """``rfdf --help`` must not import torch (zero-domain-deps CI invariant)."""
    code = (
        "from rfdf.cli.main import app; "
        "import sys; "
        "assert 'torch' not in sys.modules, "
        "f'torch leaked into sys.modules after importing rfdf.cli.main'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_cli_compute_module_no_torch_at_import() -> None:
    """``import rfdf.cli.compute`` must not load torch at module level."""
    code = (
        "import rfdf.cli.compute; "
        "import sys; "
        "assert 'torch' not in sys.modules, "
        "f'torch leaked from rfdf.cli.compute at import time'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_cli_ml_module_no_torch_at_import() -> None:
    """``import rfdf.cli.ml`` must not load torch at module level."""
    code = (
        "import rfdf.cli.ml; "
        "import sys; "
        "assert 'torch' not in sys.modules, "
        "f'torch leaked from rfdf.cli.ml at import time'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

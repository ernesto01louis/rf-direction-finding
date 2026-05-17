"""Unit tests for the ``rfdf hw udev`` CLI subcommands."""

from __future__ import annotations

import subprocess
import sys


def _run_cli(*args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    """Run ``python -m rfdf.cli.main <args>`` and return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "rfdf.cli.main", *args],
        capture_output=True,
        text=True,
        input=stdin,
        check=False,
    )


def test_udev_list_shows_known_devices() -> None:
    """`rfdf hw udev list` lists the B210 and RTL-SDR rules."""
    result = _run_cli("hw", "udev", "list")
    assert result.returncode == 0
    assert "2500:0020" in result.stdout  # Ettus B210
    assert "0bda:2838" in result.stdout  # RTL-SDR


def test_udev_generate_emits_a_rules_file() -> None:
    """`rfdf hw udev generate` prints a valid udev rules file."""
    result = _run_cli("hw", "udev", "generate")
    assert result.returncode == 0
    assert result.stdout.startswith("# udev rules for rfdf")
    assert 'SUBSYSTEM=="usb"' in result.stdout


def test_udev_install_aborts_without_confirmation() -> None:
    """`rfdf hw udev install` without --yes and no input aborts (writes nothing)."""
    result = _run_cli("hw", "udev", "install", stdin="")
    # Either the non-root guard or the declined confirmation — both exit non-zero
    # and neither installs anything.
    assert result.returncode != 0

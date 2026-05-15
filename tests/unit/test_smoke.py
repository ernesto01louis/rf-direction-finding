"""Stage 1 smoke tests.

Verify the package imports, exposes a version string, and the CLI entry point answers
`--version`. Bigger suites land per stage.
"""

from __future__ import annotations

import subprocess
import sys


def test_package_imports() -> None:
    """`import rfdf` succeeds and exposes a version."""
    import rfdf

    assert isinstance(rfdf.__version__, str)
    assert rfdf.__version__  # non-empty


def test_version_string_is_pep440_shaped() -> None:
    """The version string roughly matches PEP 440 (digits + dots + optional pre-release)."""
    import re

    import rfdf

    assert re.match(r"^\d+\.\d+\.\d+", rfdf.__version__), rfdf.__version__


def test_cli_version_prints() -> None:
    """`python -m rfdf.cli.main --version` prints the rfdf version and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "rfdf.cli.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_no_domain_libs_loaded_on_import() -> None:
    """Importing rfdf must not pull in any RF / ML / compute domain libraries.

    This is the architectural commitment from VISION.md §3 made enforceable. The CI
    `zero-domain-deps` job runs the same assertion against a base install; this
    duplicate keeps it local.
    """
    import rfdf  # noqa: F401

    banned = {
        "uhd",
        "pyadi_iio",
        "SoapySDR",
        "torch",
        "torchsig",
        "torchvision",
        "onnx",
        "onnxruntime",
        "runpod",
        "modal",
        "vastai",
        "sky",
        "fastapi",
        "uvicorn",
        "skrf",
        "PyNEC",
        "necpp",
        "ai_orchestrator_client",
    }
    leaked = banned & set(sys.modules.keys())
    assert not leaked, f"Domain libs leaked into rfdf import surface: {leaked!r}"

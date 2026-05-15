"""Smoke test for the shipped examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_doa_on_mock_array_demo_runs() -> None:
    """examples/01-doa-on-mock-array/demo.py runs end-to-end and reports PASS."""
    demo = _REPO_ROOT / "examples" / "01-doa-on-mock-array" / "demo.py"
    result = subprocess.run(
        [sys.executable, str(demo)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "demo: DOA pipeline PASS" in result.stdout

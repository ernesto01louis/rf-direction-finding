"""Smoke test for the shipped examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_doa_on_mock_array_demo_runs() -> None:
    """examples/01-doa-on-mock-array/demo.py runs end-to-end and reports PASS.

    This example is torch-free and must keep running in the torch-less
    test-base / coverage CI jobs — it does NOT importorskip torch.
    """
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


def test_train_modulation_classifier_demo_runs() -> None:
    """examples/02-train-modulation-classifier/demo.py runs end-to-end and reports PASS.

    Needs the [ml] / [ml-onnx] extras; skipped cleanly when torch is absent so
    the torch-free 01 test above keeps running in the base CI jobs.
    """
    pytest.importorskip("torch")
    demo = _REPO_ROOT / "examples" / "02-train-modulation-classifier" / "demo.py"
    result = subprocess.run(
        [sys.executable, str(demo)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "demo: ML pipeline PASS" in result.stdout


def test_rent_gpu_and_train_demo_runs() -> None:
    """examples/04-rent-gpu-and-train/demo.py runs end-to-end and reports PASS.

    Needs the [ml] / [ml-onnx] extras; skipped cleanly when torch is absent.
    The demo runs the cost-estimate + local-baseline path — it submits no real
    cloud job.
    """
    pytest.importorskip("torch")
    demo = _REPO_ROOT / "examples" / "04-rent-gpu-and-train" / "demo.py"
    result = subprocess.run(
        [sys.executable, str(demo)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "demo: GPU-rental walkthrough PASS" in result.stdout

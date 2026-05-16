"""Verify that importing the torch-free ML modules does not pull in torch.

Each check spawns a fresh interpreter process to avoid contamination from
any other test that imports torch.  The test module itself does NOT
``importorskip("torch")`` — it must run and pass on a base install that
has no torch at all.
"""

from __future__ import annotations

import subprocess
import sys


def _clean_import_check(modules: list[str], banned: list[str]) -> str:
    """Run a fresh interpreter that imports *modules* and checks for *banned*.

    Args:
        modules: Module names to import in order.
        banned: Module-name prefixes that must not appear in ``sys.modules``.

    Returns:
        Empty string if clean; a diagnostic string if any banned module leaks.
    """
    import_stmts = "; ".join(f"import {m}" for m in modules)
    check_stmts = (
        f"leaked = {banned!r}\n"
        "found = [b for b in leaked if b in __import__('sys').modules]\n"
        "if found:\n"
        f"    raise AssertionError(f'banned modules in sys.modules: {{found!r}}')\n"
        "print('OK')"
    )
    code = f"{import_stmts}; {check_stmts}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip()
    return ""


def test_import_rfdf_no_torch() -> None:
    """``import rfdf`` must not load torch, torchsig, or torchvision."""
    err = _clean_import_check(["rfdf"], ["torch", "torchsig", "torchvision"])
    assert not err, f"Lazy-import violated for 'rfdf': {err}"


def test_import_rfdf_ml_no_torch() -> None:
    """``import rfdf.ml`` must not load torch, torchsig, or torchvision."""
    err = _clean_import_check(["rfdf.ml"], ["torch", "torchsig", "torchvision"])
    assert not err, f"Lazy-import violated for 'rfdf.ml': {err}"


def test_import_rfdf_ml_datasets_no_torch() -> None:
    """``import rfdf.ml.datasets`` must not load torch, torchsig, or torchvision."""
    err = _clean_import_check(["rfdf.ml.datasets"], ["torch", "torchsig", "torchvision"])
    assert not err, f"Lazy-import violated for 'rfdf.ml.datasets': {err}"


def test_import_rfdf_ml_datasets_augmentation_no_torch() -> None:
    """``import rfdf.ml.datasets.augmentation`` must not load torch or torchsig."""
    err = _clean_import_check(
        ["rfdf.ml.datasets.augmentation"], ["torch", "torchsig", "torchvision"]
    )
    assert not err, f"Lazy-import violated for 'rfdf.ml.datasets.augmentation': {err}"


def test_import_rfdf_ml_errors_no_torch() -> None:
    """``import rfdf.ml.errors`` must not load torch or torchsig."""
    err = _clean_import_check(["rfdf.ml.errors"], ["torch", "torchsig", "torchvision"])
    assert not err, f"Lazy-import violated for 'rfdf.ml.errors': {err}"


def test_import_rfdf_ml_datasets_synthetic_no_torch() -> None:
    """Importing the synthetic module (not calling a function) must not load torch."""
    err = _clean_import_check(["rfdf.ml.datasets.synthetic"], ["torch", "torchsig", "torchvision"])
    assert not err, f"Lazy-import violated for 'rfdf.ml.datasets.synthetic': {err}"


def test_import_rfdf_ml_datasets_radioml_no_torch() -> None:
    """Importing the radioml module (not calling a function) must not load torch."""
    err = _clean_import_check(["rfdf.ml.datasets.radioml"], ["torch", "torchsig", "torchvision"])
    assert not err, f"Lazy-import violated for 'rfdf.ml.datasets.radioml': {err}"


def test_import_rfdf_ml_datasets_captured_no_torch() -> None:
    """Importing the captured module (not calling a function) must not load torch."""
    err = _clean_import_check(["rfdf.ml.datasets.captured"], ["torch", "torchsig", "torchvision"])
    assert not err, f"Lazy-import violated for 'rfdf.ml.datasets.captured': {err}"


def test_import_rfdf_ml_datasets_torchsig_compat_no_torch() -> None:
    """Importing _torchsig_compat must not eagerly load torch or torchsig."""
    err = _clean_import_check(
        ["rfdf.ml.datasets._torchsig_compat"], ["torch", "torchsig", "torchvision"]
    )
    assert not err, f"Lazy-import violated for 'rfdf.ml.datasets._torchsig_compat': {err}"

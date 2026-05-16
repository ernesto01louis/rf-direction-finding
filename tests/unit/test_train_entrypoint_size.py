"""Assert that train_entrypoint.py stays under 300 lines.

The training logic lives in rfdf.ml.training; the entrypoint is intentionally
thin.  This test is the enforcement mechanism.
"""

from __future__ import annotations

from pathlib import Path


def test_train_entrypoint_line_count() -> None:
    """train_entrypoint.py must have fewer than 300 lines."""
    entrypoint = Path(__file__).parent.parent.parent / "src" / "rfdf" / "ml" / "train_entrypoint.py"
    assert entrypoint.exists(), f"train_entrypoint.py not found at {entrypoint}"
    lines = entrypoint.read_text().splitlines()
    count = len(lines)
    assert count < 300, (
        f"train_entrypoint.py has {count} lines — must be < 300.  "
        "Move logic into rfdf.ml.training instead."
    )

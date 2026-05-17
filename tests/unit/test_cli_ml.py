"""Unit tests for the ``rfdf ml`` CLI sub-commands."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rfdf.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_ml_help_lists_subcommands() -> None:
    """``rfdf ml --help`` advertises train, export, and registry."""
    result = runner.invoke(app, ["ml", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("train", "export", "registry"):
        assert sub in result.output


def test_top_level_help_lists_ml() -> None:
    """``rfdf --help`` lists the ml sub-app."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "ml" in result.output


def test_ml_registry_help_lists_subcommands() -> None:
    """``rfdf ml registry --help`` advertises list, show, export, import, delete."""
    result = runner.invoke(app, ["ml", "registry", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("list", "show", "export", "import", "delete"):
        assert sub in result.output


# ---------------------------------------------------------------------------
# registry list (empty registry via RFDF_MODEL_REGISTRY env override)
# ---------------------------------------------------------------------------


def test_ml_registry_list_empty(tmp_path: Path) -> None:
    """``rfdf ml registry list`` with an empty registry prints a helpful message."""
    env = {**os.environ, "RFDF_MODEL_REGISTRY": str(tmp_path)}
    result = runner.invoke(app, ["ml", "registry", "list"], env=env)
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower() or "No" in result.output


def test_ml_registry_show_missing_model(tmp_path: Path) -> None:
    """``rfdf ml registry show`` for a non-existent model exits non-zero."""
    env = {**os.environ, "RFDF_MODEL_REGISTRY": str(tmp_path)}
    result = runner.invoke(app, ["ml", "registry", "show", "no-such-model"], env=env)
    assert result.exit_code != 0
    assert "error" in result.output.lower() or "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# train — missing recipe exits non-zero
# ---------------------------------------------------------------------------


def test_ml_train_missing_recipe(tmp_path: Path) -> None:
    """``rfdf ml train`` with a missing recipe path exits non-zero."""
    missing = tmp_path / "no-such-recipe.toml"
    result = runner.invoke(app, ["ml", "train", "--recipe", str(missing), "--yes"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "recipe" in result.output.lower()


# ---------------------------------------------------------------------------
# train — cost-confirmation guardrail
# ---------------------------------------------------------------------------


def _write_minimal_recipe(path: Path) -> None:
    """Write a minimal valid TOML training recipe to *path*."""
    path.write_text(
        """
name = "test-recipe"

[dataset]
kind = "modulation"
num_signals = 4
num_samples_per_signal = 16
num_iq_samples = 64

[model]
architecture = "resnet1d"
num_classes = 4
input_shape = [2, 64]

[training]
epochs = 1
batch_size = 4

[compute]
backend = "local"
gpu_count = 0
""",
        encoding="utf-8",
    )


def test_ml_train_requires_confirmation_without_yes(tmp_path: Path) -> None:
    """``rfdf ml train`` must NOT auto-submit when --yes is absent.

    The CliRunner receives no stdin input, so ``typer.confirm`` sees EOF and
    treats it as a 'no' — the job must NOT be submitted.
    """
    recipe = tmp_path / "recipe.toml"
    _write_minimal_recipe(recipe)

    # mix_stderr=False so we can inspect stdout independently
    result = runner.invoke(
        app,
        ["ml", "train", "--recipe", str(recipe)],
        input="n\n",  # explicit 'n' to the confirmation prompt
    )
    # Should exit 0 (user declined gracefully) or exit with cancelled message.
    # The critical assertion: "Submit" was asked but job was NOT submitted.
    assert "cancelled" in result.output.lower() or result.exit_code == 0
    # Verify the job was NOT submitted (no "Job submitted" in output).
    assert "job submitted" not in result.output.lower()


def test_ml_train_yes_flag_bypasses_confirmation(tmp_path: Path) -> None:
    """``rfdf ml train --yes`` skips confirmation and submits to local backend."""
    recipe = tmp_path / "recipe.toml"
    _write_minimal_recipe(recipe)

    result = runner.invoke(
        app,
        ["ml", "train", "--recipe", str(recipe), "--yes"],
    )
    # Local backend runs synchronously; expect success or a training error
    # (torch may be available in the test environment).
    # The key assertion: no crash due to the confirmation guardrail blocking it.
    assert result.exit_code in (0, 1)
    # Must NOT say "cancelled"
    assert "cancelled" not in result.output.lower()


def test_ml_train_require_cost_confirmation_false_skips_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``require_cost_confirmation=False`` in config, no prompt is shown."""
    recipe = tmp_path / "recipe.toml"
    _write_minimal_recipe(recipe)

    # Patch load_config to return require_cost_confirmation=False.
    from rfdf.config import ComputeSection

    class _FakeConfig:
        compute = ComputeSection(require_cost_confirmation=False)

    monkeypatch.setattr(
        "rfdf.cli.ml.load_config",
        lambda: _FakeConfig(),
        raising=False,
    )

    result = runner.invoke(
        app,
        ["ml", "train", "--recipe", str(recipe)],
        # No input — if a prompt appears it would get EOF and fail.
        input="",
    )
    # Should proceed to submit (not cancel), regardless of exit code from training.
    assert "cancelled" not in result.output.lower()


# ---------------------------------------------------------------------------
# export — unknown format exits non-zero
# ---------------------------------------------------------------------------


def test_ml_export_unknown_format(tmp_path: Path) -> None:
    """``rfdf ml export`` with an unknown --format exits non-zero."""
    result = runner.invoke(
        app,
        ["ml", "export", "some-model", "--format", "pickle", "--output", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "pickle" in result.output.lower() or "unknown" in result.output.lower()


def test_ml_export_missing_model_exits_nonzero(tmp_path: Path) -> None:
    """``rfdf ml export`` for a non-existent model exits non-zero."""
    env = {**os.environ, "RFDF_MODEL_REGISTRY": str(tmp_path)}
    result = runner.invoke(
        app,
        ["ml", "export", "no-such-model", "--format", "onnx", "--output", str(tmp_path)],
        env=env,
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# train — --gpu-count override
# ---------------------------------------------------------------------------


def test_ml_train_gpu_count_override(tmp_path: Path) -> None:
    """``rfdf ml train --gpu-count N`` overrides the recipe's [compute] gpu_count.

    The shipped recipes target a GPU (``gpu_count = 1``); ``--gpu-count 0`` is
    what lets ``--compute local`` run on a CPU-only host. The recipe summary
    line echoes the *effective* (post-override) GPU count, and it is printed
    before any torch import, so this assertion holds in the torch-free
    ``test-base`` job too.
    """
    recipe = tmp_path / "gpu-recipe.toml"
    recipe.write_text(
        """
name = "gpu-recipe"

[dataset]
kind = "modulation"
num_signals = 4
num_samples_per_signal = 16
num_iq_samples = 64

[model]
architecture = "resnet1d"
num_classes = 4
input_shape = [2, 64]

[training]
epochs = 1
batch_size = 4

[compute]
backend = "local"
gpu_count = 1
""",
        encoding="utf-8",
    )

    # Without the override, the summary echoes the recipe's own gpu_count (1).
    base = runner.invoke(
        app, ["ml", "train", "--recipe", str(recipe), "--compute", "local", "--yes"]
    )
    assert re.search(r"GPUs:\s*1", base.output), base.output

    # With --gpu-count 0 the summary echoes the overridden count (0) — the path
    # that lets `--compute local` run on a CPU-only host.
    overridden = runner.invoke(
        app,
        ["ml", "train", "--recipe", str(recipe), "--compute", "local", "--gpu-count", "0", "--yes"],
    )
    assert re.search(r"GPUs:\s*0", overridden.output), overridden.output

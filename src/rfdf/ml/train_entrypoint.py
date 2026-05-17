"""Training entrypoint for ``rfdf.ml``.

Invoked as::

    python -m rfdf.ml.train_entrypoint <recipe.toml>

or, when submitted via a :class:`~rfdf.hal.compute.ComputeJob` shim, the recipe
is read from the ``$RFDF_RECIPE`` environment variable (JSON-serialised
:class:`~rfdf.ml.recipes.TrainingRecipe`).

This file is intentionally thin — all logic lives in
:mod:`rfdf.ml.training`.  The 300-line ceiling is enforced by
``tests/unit/test_train_entrypoint_size.py``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_recipe() -> TrainingRecipe:  # type: ignore[name-defined]  # noqa: F821
    """Load the recipe from CLI arg or ``$RFDF_RECIPE`` env var.

    Returns:
        A validated :class:`~rfdf.ml.recipes.TrainingRecipe`.

    Raises:
        SystemExit: If neither a CLI argument nor ``$RFDF_RECIPE`` is available.
    """
    from rfdf.ml.recipes import TrainingRecipe, load_recipe

    if len(sys.argv) > 1:
        recipe_path = Path(sys.argv[1])
        logger.info("Loading recipe from file: %s", recipe_path)
        return load_recipe(recipe_path)

    env_json = os.environ.get("RFDF_RECIPE", "")
    if env_json:
        logger.info("Loading recipe from $RFDF_RECIPE env var.")
        data = json.loads(env_json)
        return TrainingRecipe.model_validate(data)

    print(
        "Usage: python -m rfdf.ml.train_entrypoint <recipe.toml>",
        file=sys.stderr,
    )
    print(
        "Or set $RFDF_RECIPE to a JSON-serialised TrainingRecipe.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    """Entry point: load recipe, build datasets + model, run training, report.

    Raises:
        SystemExit: On invalid recipe or missing dependencies.
    """
    from rfdf.ml.datasets import build_datasets
    from rfdf.ml.training import train

    recipe = _resolve_recipe()

    logger.info("Recipe: %s  device=auto", recipe.name)

    # Build datasets.
    train_ds, val_ds, _test_ds = build_datasets(recipe.dataset)

    # Choose output directory.
    output_dir = Path(f"outputs/{recipe.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run training.
    result = train(
        recipe=recipe,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=output_dir,
        device=None,  # auto-detect
    )

    # Print a summary line for CI / operator inspection.
    print(
        f"{recipe.name}: trained best_val_acc={result.best_val_accuracy:.4f}"
        f" steps={result.final_step}"
        f" wall={result.wall_clock_s:.1f}s"
        f" PASS"
    )


if __name__ == "__main__":
    main()

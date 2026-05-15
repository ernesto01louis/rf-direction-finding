"""rfdf command-line interface.

The ``rfdf`` entry point exposes a small Typer app. Subcommands land per stage:

* Stage 2: ``rfdf hw list-backends``, ``rfdf hw selftest``, ``rfdf config show/validate``
* Stage 3: ``rfdf doa run/calibrate/benchmark/morph-capture``
* Stage 4: ``rfdf ml train/registry/export``, ``rfdf compute *``
* Stage 5: ``rfdf hw udev install``, ``rfdf hw geometry``, ``rfdf hw rotator``
* Stage 7: ``rfdf orchestrator status/register/hindsight/vault/planner``

At Stage 1 only ``--version`` is wired.
"""

"""Concrete backend implementations for the rfdf HAL Protocols.

Backends are organized by HAL group: ``sdr``, ``rotator``, ``geometry``, ``compute``.
Each backend module exposes a ``create(**kwargs) -> BackendInstance`` factory
registered via the ``rfdf.backends.<group>`` entry-point in ``pyproject.toml``.

Nothing here is imported until a backend is explicitly requested via
:func:`rfdf.hal.discovery.load_backend`. This preserves the audit-lesson
``zero-domain-deps`` guarantee — installing the base ``rfdf`` package never
pulls in hardware SDKs or remote compute providers.
"""

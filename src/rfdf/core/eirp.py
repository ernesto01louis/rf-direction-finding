"""EIRP cap enforcement.

The platform enforces a configurable EIRP cap on every TX-initiating call so that
operators must take an explicit, auditable step to exceed the EU SRD 25 mW default.
See :doc:`/SECURITY.md` §3 for the regulatory context.

Public surface:

* :class:`EirpCapExceededError` — raised when a power request exceeds the cap.
* :func:`enforce_eirp_cap` — imperative check; pass the requested power.
* :func:`requires_eirp_check` — decorator for async methods with a ``power_dbm``
  kwarg; raises before the wrapped body runs.

The cap is read from :class:`rfdf.config.RfdfConfig`. The override gate
(``eirp.override_explicit = true``) makes the operator decision visible in config
diffs; without the override, ``max_eirp_dbm`` is treated as a hard ceiling.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from rfdf.config import RfdfConfig, load_config

#: Default EIRP cap when no configuration is supplied (EU SRD 25 mW = 14 dBm).
DEFAULT_EIRP_CAP_DBM: float = 14.0

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


class EirpCapExceededError(RuntimeError):
    """Raised when a transmit request would exceed the configured EIRP cap.

    Attributes:
        power_dbm: The requested EIRP in dBm.
        cap_dbm: The configured cap in dBm.
    """

    def __init__(self, power_dbm: float, cap_dbm: float) -> None:
        """Initialise with the requested + capped power values."""
        super().__init__(
            f"EIRP cap exceeded: requested {power_dbm:.2f} dBm, "
            f"cap {cap_dbm:.2f} dBm. Set eirp.override_explicit=true "
            f"and raise eirp.max_eirp_dbm in config to authorize."
        )
        self.power_dbm = power_dbm
        self.cap_dbm = cap_dbm


def enforce_eirp_cap(power_dbm: float, *, config: RfdfConfig | None = None) -> None:
    """Validate ``power_dbm`` against the configured cap.

    Args:
        power_dbm: Requested effective isotropic radiated power, in dBm.
        config: Optional pre-resolved :class:`RfdfConfig`. When ``None``, the
            module loads the default config (env + TOML + builtins).

    Raises:
        EirpCapExceededError: If ``power_dbm > config.eirp.max_eirp_dbm`` and
            ``config.eirp.override_explicit`` is ``False``.
    """
    cfg = config if config is not None else load_config()
    if power_dbm > cfg.eirp.max_eirp_dbm and not cfg.eirp.override_explicit:
        raise EirpCapExceededError(power_dbm, cfg.eirp.max_eirp_dbm)


def requires_eirp_check(func: F) -> F:
    """Decorate an async method whose ``power_dbm`` kwarg triggers EIRP enforcement.

    The decorator extracts ``power_dbm`` from the keyword arguments and calls
    :func:`enforce_eirp_cap` before the wrapped function runs. If the kwarg is
    absent, the wrapped function runs unchecked (use plain ``enforce_eirp_cap``
    when ``power_dbm`` is positional or computed dynamically).

    Args:
        func: Async callable accepting a ``power_dbm`` keyword argument.

    Returns:
        Wrapped function that performs the EIRP check before forwarding.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        power_dbm = kwargs.get("power_dbm")
        if power_dbm is not None:
            config = kwargs.get("config")
            enforce_eirp_cap(float(power_dbm), config=config)
        return await func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


__all__ = [
    "DEFAULT_EIRP_CAP_DBM",
    "EirpCapExceededError",
    "enforce_eirp_cap",
    "requires_eirp_check",
]

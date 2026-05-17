"""Pydantic-settings configuration for rfdf.

The precedence rule (**CLI > env > TOML > defaults**) is documented in *exactly one
place*: :doc:`/ARCHITECTURE.md` §4. This module is the implementation; do not
restate the rule here. Keep documentation drift impossible.

Layout mirrors the TOML example in ``docs/configuration.md``:

.. code-block:: toml

    [default]
    log_level = "info"
    data_dir = "~/.local/share/rfdf"

    [sdr]
    backend = "mock"
    center_freq_hz = 868e6
    sample_rate_hz = 2e6
    rx_gain_db = 30.0

    [rotator]
    backend = "mock"

    [geometry]
    backend = "static"
    antennas = [[0.0, 0.0, 0.0], [0.17, 0.0, 0.0], ...]

    [compute]
    backend = "local"

    [eirp]
    max_eirp_dbm = 14
    override_explicit = false

Environment variables use ``RFDF_`` as the top-level prefix and ``__`` as the
nested delimiter — ``RFDF_SDR__CENTER_FREQ_HZ=2400e6`` overrides ``sdr.center_freq_hz``.

CLI overrides arrive via the explicit ``cli_overrides`` kwarg on
:func:`load_config`; the CLI layer (``src/rfdf/cli/``) is responsible for parsing
flags and threading them through.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal

import platformdirs
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class DefaultSection(BaseModel):
    """Top-level defaults applied across the platform."""

    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    data_dir: Path = Field(default_factory=lambda: Path(platformdirs.user_data_path("rfdf")))


class SdrSection(BaseModel):
    """SDR backend selection + default capture parameters."""

    backend: str = "mock"
    center_freq_hz: float = 868e6
    sample_rate_hz: float = 2e6
    bandwidth_hz: float | None = None
    rx_gain_db: float = 30.0
    antenna: str | None = None
    channels: list[int] = Field(default_factory=lambda: [0])
    coherent: bool = False
    reference_clock: Literal["internal", "external", "gpsdo"] = "internal"
    timing_source: Literal["internal", "external", "gpsdo", "pps"] = "internal"


class RotatorSection(BaseModel):
    """Rotator backend selection."""

    backend: str = "mock"


class GeometrySection(BaseModel):
    """Geometry backend selection + reference antenna layout.

    The default ``antennas`` list is the pentagonal 5-antenna LPDA cluster
    at λ/2 spacing for 868 MHz used by the demo-no-hardware smoke test.
    """

    backend: str = "static"
    antennas: list[list[float]] = Field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.17, 0.0, 0.0],
            [0.0, 0.17, 0.0],
            [-0.17, 0.0, 0.0],
            [0.0, -0.17, 0.0],
        ]
    )


class ComputeSection(BaseModel):
    """Compute backend selection and non-secret routing defaults.

    Attributes:
        backend: Default compute backend name (e.g. ``"local"``, ``"runpod"``).
        default_gpu_model: Optional GPU model preference applied when a training
            recipe does not specify one (e.g. ``"A6000"``).  ``None`` means no
            preference — the backend selects any available GPU.
        default_gpu_min_vram_gb: Minimum per-GPU VRAM in GB when a recipe does
            not specify a minimum.  Default 16 GB suits most 1-D signal
            classification models.
        require_cost_confirmation: When ``True`` (default), ``rfdf ml train``
            prints the cost estimate and requires explicit operator confirmation
            (or the ``--yes`` flag) before submitting a job.  Set to ``False``
            only in fully automated CI pipelines where a human is never present.
            **Never auto-submit cloud jobs without understanding the cost.**
    """

    backend: str = "local"
    default_gpu_model: str | None = None
    default_gpu_min_vram_gb: float = 16.0
    require_cost_confirmation: bool = True


class EirpSection(BaseModel):
    """EIRP cap policy.

    ``max_eirp_dbm`` defaults to 14 dBm (25 mW EIRP — the EU SRD general limit).
    Operators raise the cap by setting ``override_explicit = true`` and adjusting
    ``max_eirp_dbm``; the override gate makes the decision visible in config diffs.
    """

    max_eirp_dbm: float = 14.0
    override_explicit: bool = False


def _default_config_path() -> Path:
    """Return ``~/.config/rfdf/config.toml`` (platform-aware)."""
    return Path(platformdirs.user_config_path("rfdf")) / "config.toml"


class RfdfConfig(BaseSettings):
    """Aggregate rfdf configuration with documented source precedence.

    Sources, **in precedence order**:

    1. ``cli_overrides`` kwarg (highest — explicit CLI flag values).
    2. ``RFDF_*`` environment variables (nested via ``__``).
    3. TOML file at ``~/.config/rfdf/config.toml`` (or ``$RFDF_CONFIG`` if set).
    4. Built-in defaults (lowest).

    See :doc:`/ARCHITECTURE.md` §4 — this docstring intentionally points at the
    canonical source rather than restating the rule.
    """

    model_config = SettingsConfigDict(
        env_prefix="RFDF_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    default: DefaultSection = Field(default_factory=DefaultSection)
    sdr: SdrSection = Field(default_factory=SdrSection)
    rotator: RotatorSection = Field(default_factory=RotatorSection)
    geometry: GeometrySection = Field(default_factory=GeometrySection)
    compute: ComputeSection = Field(default_factory=ComputeSection)
    eirp: EirpSection = Field(default_factory=EirpSection)


class _TomlConfigSource(PydanticBaseSettingsSource):
    """Pydantic-settings source that reads the rfdf TOML config file."""

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        toml_path: Path | None,
    ) -> None:
        super().__init__(settings_cls)
        self._toml_path = toml_path
        self._cached: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._cached is not None:
            return self._cached
        if self._toml_path is None or not self._toml_path.is_file():
            self._cached = {}
            return self._cached
        with self._toml_path.open("rb") as fh:
            self._cached = tomllib.load(fh)
        return self._cached

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # We supply the whole mapping via __call__; per-field hooks aren't used.
        data = self._load()
        if field_name in data:
            return data[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._load()


class _ExplicitOverridesSource(PydanticBaseSettingsSource):
    """Highest-precedence source carrying explicit dict overrides (e.g. from CLI)."""

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        overrides: dict[str, Any] | None,
    ) -> None:
        super().__init__(settings_cls)
        self._overrides = overrides or {}

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        if field_name in self._overrides:
            return self._overrides[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._overrides)


def load_config(
    *,
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> RfdfConfig:
    """Resolve a :class:`RfdfConfig` honouring the precedence rule.

    Args:
        config_path: Override for the TOML location. Defaults to
            ``$RFDF_CONFIG`` if set, else ``platformdirs.user_config_path()``.
        cli_overrides: Optional top-level overrides (``{"sdr": {...}, "eirp":
            {...}}``) injected from the CLI; highest precedence after defaults.

    Returns:
        Fully-resolved :class:`RfdfConfig` instance.
    """
    if config_path is None:
        env_path = os.environ.get("RFDF_CONFIG")
        toml_path = Path(env_path) if env_path else _default_config_path()
    else:
        toml_path = config_path

    overrides_source = _ExplicitOverridesSource(RfdfConfig, cli_overrides)
    toml_source = _TomlConfigSource(RfdfConfig, toml_path)

    class _ConfiguredSettings(RfdfConfig):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            # Pydantic-settings calls sources left-to-right; first hit wins.
            # Precedence order: cli > env > toml > defaults (init_settings is
            # the kwargs-to-the-constructor path — empty here).
            return (
                overrides_source,
                env_settings,
                toml_source,
                init_settings,
                file_secret_settings,
            )

    return _ConfiguredSettings()


def resolve_value_sources(
    *,
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Annotate where each top-level config section was resolved from.

    Returns a flat ``{section: source}`` mapping where ``source`` is one of
    ``"cli"``, ``"env"``, ``"toml"``, or ``"default"``. Used by
    ``rfdf config show`` to surface origins.
    """
    if config_path is None:
        env_path = os.environ.get("RFDF_CONFIG")
        toml_path = Path(env_path) if env_path else _default_config_path()
    else:
        toml_path = config_path

    toml_data: dict[str, Any] = {}
    if toml_path.is_file():
        with toml_path.open("rb") as fh:
            toml_data = tomllib.load(fh)

    sections = ("default", "sdr", "rotator", "geometry", "compute", "eirp")
    sources: dict[str, str] = {}
    overrides = cli_overrides or {}
    for section in sections:
        if section in overrides:
            sources[section] = "cli"
            continue
        prefix = f"RFDF_{section.upper()}__"
        if any(key.startswith(prefix) for key in os.environ):
            sources[section] = "env"
            continue
        if section in toml_data:
            sources[section] = "toml"
            continue
        sources[section] = "default"
    return sources


__all__ = [
    "ComputeSection",
    "DefaultSection",
    "EirpSection",
    "GeometrySection",
    "RfdfConfig",
    "RotatorSection",
    "SdrSection",
    "load_config",
    "resolve_value_sources",
]


# Sentinel so coverage tooling doesn't complain about unreachable mypy guard imports.
if "pytest" in sys.modules:  # pragma: no cover
    _ = (DefaultSection, SdrSection, RotatorSection)

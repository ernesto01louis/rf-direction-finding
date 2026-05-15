"""Unit tests for ``rfdf.config``.

Covers the four-tier precedence (cli > env > toml > defaults) and the
``resolve_value_sources`` annotation map used by ``rfdf config show``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfdf.config import RfdfConfig, load_config, resolve_value_sources


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content)


def test_defaults_when_no_toml_no_env_no_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no inputs, RfdfConfig falls back to the built-in defaults."""
    monkeypatch.delenv("RFDF_CONFIG", raising=False)
    for key in list(__import__("os").environ.keys()):
        if key.startswith("RFDF_"):
            monkeypatch.delenv(key, raising=False)

    cfg = load_config(config_path=tmp_path / "missing.toml")
    assert isinstance(cfg, RfdfConfig)
    assert cfg.sdr.backend == "mock"
    assert cfg.sdr.center_freq_hz == pytest.approx(868e6)
    assert cfg.eirp.max_eirp_dbm == pytest.approx(14.0)
    assert cfg.eirp.override_explicit is False


def test_toml_overrides_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A TOML file replaces matching defaults."""
    for key in list(__import__("os").environ.keys()):
        if key.startswith("RFDF_"):
            monkeypatch.delenv(key, raising=False)

    toml = tmp_path / "config.toml"
    _write_toml(
        toml,
        """
[sdr]
backend = "file-replay"
center_freq_hz = 2400e6
""",
    )
    cfg = load_config(config_path=toml)
    assert cfg.sdr.backend == "file-replay"
    assert cfg.sdr.center_freq_hz == pytest.approx(2400e6)
    # Untouched defaults preserved.
    assert cfg.eirp.max_eirp_dbm == pytest.approx(14.0)


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables (RFDF_SDR__*) outrank the TOML file."""
    toml = tmp_path / "config.toml"
    _write_toml(
        toml,
        """
[sdr]
backend = "file-replay"
""",
    )
    monkeypatch.setenv("RFDF_SDR__BACKEND", "mock")
    cfg = load_config(config_path=toml)
    assert cfg.sdr.backend == "mock"


def test_cli_overrides_outrank_env_and_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit cli_overrides win over env and TOML."""
    toml = tmp_path / "config.toml"
    _write_toml(toml, '[sdr]\nbackend = "toml-pick"\n')
    monkeypatch.setenv("RFDF_SDR__BACKEND", "env-pick")

    cfg = load_config(
        config_path=toml,
        cli_overrides={"sdr": {"backend": "cli-pick"}},
    )
    assert cfg.sdr.backend == "cli-pick"


def test_resolve_value_sources_labels_each_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source annotation: one of cli / env / toml / default per section."""
    for key in list(__import__("os").environ.keys()):
        if key.startswith("RFDF_"):
            monkeypatch.delenv(key, raising=False)

    toml = tmp_path / "config.toml"
    _write_toml(
        toml,
        """
[rotator]
backend = "mock"

[compute]
backend = "local"
""",
    )
    monkeypatch.setenv("RFDF_GEOMETRY__BACKEND", "static")

    sources = resolve_value_sources(
        config_path=toml,
        cli_overrides={"eirp": {"max_eirp_dbm": 20.0}},
    )
    assert sources["eirp"] == "cli"
    assert sources["geometry"] == "env"
    assert sources["rotator"] == "toml"
    assert sources["compute"] == "toml"
    assert sources["default"] == "default"
    assert sources["sdr"] == "default"


def test_geometry_default_is_pentagonal_5_antenna_array() -> None:
    """The built-in geometry default is the demo-no-hardware reference array."""
    cfg = RfdfConfig()
    antennas = cfg.geometry.antennas
    assert len(antennas) == 5
    # Origin antenna at (0, 0, 0).
    assert antennas[0] == [0.0, 0.0, 0.0]
    # Pentagonal at ~λ/2 spacing for 868 MHz (~0.17 m).
    for ant in antennas[1:]:
        radius = (ant[0] ** 2 + ant[1] ** 2) ** 0.5
        assert 0.16 <= radius <= 0.18

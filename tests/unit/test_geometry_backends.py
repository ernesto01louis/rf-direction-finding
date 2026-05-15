"""Unit tests for the static and mock-morph geometry backends.

Verifies the contract-conformant pieces that benefit from focused tests; the
broader Hypothesis-driven property tests land in tests/contracts/ (commit 11).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from rfdf.backends.geometry.mock_morph import create as create_mock_morph
from rfdf.backends.geometry.static import create as create_static
from rfdf.hal import GeometryController


def test_static_defaults_to_pentagonal_5_antenna_array() -> None:
    """Calling create() with no args yields the demo-no-hardware reference geometry."""
    geo = create_static()
    assert isinstance(geo, GeometryController)
    assert geo.num_antennas == 5
    assert geo.is_morphable is False
    assert geo.positioning_repeatability_mm == 0.0


def test_static_positions_shape_and_finite() -> None:
    """positions() returns a (N, 3) finite float array."""
    geo = create_static()
    arr = asyncio.run(geo.positions())
    assert arr.shape == (5, 3)
    assert np.all(np.isfinite(arr))


def test_static_morphing_methods_raise() -> None:
    """goto_preset / goto_positions / save_preset all raise NotImplementedError."""
    geo = create_static()
    with pytest.raises(NotImplementedError):
        asyncio.run(geo.goto_preset("any"))
    with pytest.raises(NotImplementedError):
        asyncio.run(geo.goto_positions(np.zeros((5, 3))))
    with pytest.raises(NotImplementedError):
        asyncio.run(geo.save_preset("any", np.zeros((5, 3))))
    assert asyncio.run(geo.list_presets()) == []


def test_static_rejects_bad_shape() -> None:
    """Construction with non-(N, 3) input raises ValueError."""
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        create_static(antennas=[[0.0, 0.0]])
    with pytest.raises(ValueError, match=r"got shape"):
        create_static(antennas=[])
    with pytest.raises(ValueError, match="at least one"):
        create_static(antennas=np.empty((0, 3)))


def test_static_calibrate_is_noop_ok() -> None:
    """calibrate() returns ok=True with a no-op message."""
    geo = create_static()
    report = asyncio.run(geo.calibrate())
    assert report.ok is True
    assert report.backend == "static"


def test_mock_morph_round_trips_commanded_positions(tmp_path: Path) -> None:
    """goto_positions(target) → positions() returns target ± repeatability noise."""
    geo = create_mock_morph(
        repeatability_mm=0.0,  # zero noise → exact round-trip
        presets_path=tmp_path / "presets.toml",
        seed=42,
    )
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.10, 0.0],
            [-0.10, 0.0, 0.0],
            [0.0, -0.10, 0.0],
        ]
    )
    asyncio.run(geo.goto_positions(target))
    actual = asyncio.run(geo.positions())
    np.testing.assert_allclose(actual, target, atol=1e-9)


def test_mock_morph_applies_repeatability_noise(tmp_path: Path) -> None:
    """Non-zero repeatability produces small but measurable offsets."""
    geo = create_mock_morph(
        repeatability_mm=1.0,  # 1 mm 1-sigma noise
        presets_path=tmp_path / "presets.toml",
        seed=1,
    )
    target = np.zeros((5, 3))
    asyncio.run(geo.goto_positions(target))
    actual = asyncio.run(geo.positions())
    # Noise is in meters; 1 mm = 0.001 m. With 5x3=15 samples and seed=1,
    # the actual should not equal target but stays well under 5 mm.
    assert not np.allclose(actual, target)
    assert np.max(np.abs(actual - target)) < 0.005


def test_mock_morph_persists_presets_across_instances(tmp_path: Path) -> None:
    """Presets saved by one instance load in a new instance pointing at the same path."""
    presets_path = tmp_path / "presets.toml"
    geo_a = create_mock_morph(presets_path=presets_path, seed=0)
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, -0.05, 0.0],
        ]
    )
    asyncio.run(geo_a.save_preset("tight", target))
    assert "tight" in asyncio.run(geo_a.list_presets())

    geo_b = create_mock_morph(presets_path=presets_path, seed=0)
    assert "tight" in asyncio.run(geo_b.list_presets())
    asyncio.run(geo_b.goto_preset("tight"))
    actual = asyncio.run(geo_b.positions())
    # repeatability noise is 0.5 mm default — expect close-to-target.
    np.testing.assert_allclose(actual, target, atol=0.005)


def test_mock_morph_goto_positions_shape_mismatch_raises(tmp_path: Path) -> None:
    """A target with the wrong shape raises ValueError before applying."""
    geo = create_mock_morph(presets_path=tmp_path / "presets.toml")
    with pytest.raises(ValueError, match="shape"):
        asyncio.run(geo.goto_positions(np.zeros((3, 3))))


def test_mock_morph_unknown_preset_raises(tmp_path: Path) -> None:
    """goto_preset on an unknown name raises KeyError."""
    geo = create_mock_morph(presets_path=tmp_path / "presets.toml")
    with pytest.raises(KeyError, match="missing"):
        asyncio.run(geo.goto_preset("missing"))


def test_mock_morph_calibrate_snaps_actual_to_commanded(tmp_path: Path) -> None:
    """calibrate() makes actual == commanded (i.e. clears positioning error)."""
    geo = create_mock_morph(
        repeatability_mm=2.0,
        presets_path=tmp_path / "presets.toml",
        seed=7,
    )
    target = np.zeros((5, 3))
    asyncio.run(geo.goto_positions(target))
    asyncio.run(geo.calibrate())
    actual = asyncio.run(geo.positions())
    np.testing.assert_array_equal(actual, target)

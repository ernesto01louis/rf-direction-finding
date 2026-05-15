"""Morphable mock geometry.

Simulates a motorized linear-rail array that can move antennas to arbitrary
commanded positions, with realistic positioning error and TOML-backed presets.
Use this to develop morphing-array algorithms (Stage 3+) without burning rail
hours on the physical rig.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import platformdirs
import structlog

from rfdf.hal.types import CalibrationReport

_LOG = structlog.get_logger(__name__)


class MockMorphGeometry:
    """Implements :class:`rfdf.hal.GeometryController` with morphable positions.

    Positioning error is sampled per ``goto_positions`` call from a Gaussian with
    standard deviation ``repeatability_mm / 1000`` (meters). A fixed
    ``np.random.Generator`` seeded by ``seed`` keeps tests reproducible.

    Args:
        initial_positions: Starting antenna positions, ``(N, 3)``.
        repeatability_mm: Positioning repeatability (1-sigma noise in mm).
        presets_path: Where to persist named presets (TOML). Defaults to
            ``platformdirs.user_data_path("rfdf") / "presets.toml"``.
        seed: RNG seed for deterministic positioning error.
    """

    def __init__(
        self,
        initial_positions: Sequence[Sequence[float]],
        *,
        repeatability_mm: float = 0.5,
        presets_path: Path | None = None,
        seed: int = 0,
    ) -> None:
        """Cache initial positions, RNG, and preset-store location."""
        arr = np.asarray(initial_positions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"mock-morph geometry: expected (N, 3) positions, got shape {arr.shape}"
            )
        if arr.shape[0] == 0:
            raise ValueError("mock-morph geometry: at least one antenna position required")
        self._commanded = arr
        self._actual = arr.copy()
        self._repeatability_mm = float(repeatability_mm)
        self._presets_path = presets_path or (
            Path(platformdirs.user_data_path("rfdf")) / "presets.toml"
        )
        self._rng = np.random.default_rng(seed)
        self._presets: dict[str, np.ndarray] = {}
        self._load_presets()

    @property
    def num_antennas(self) -> int:
        """Number of antennas in the array."""
        return int(self._commanded.shape[0])

    @property
    def is_morphable(self) -> bool:
        """Always ``True`` — that's the point."""
        return True

    @property
    def positioning_repeatability_mm(self) -> float:
        """Configured positioning repeatability."""
        return self._repeatability_mm

    async def positions(self) -> np.ndarray:
        """Return the current actual (noisy) positions."""
        return self._actual.copy()

    async def goto_preset(self, preset_name: str) -> None:
        """Move to the antennas stored under ``preset_name``."""
        if preset_name not in self._presets:
            raise KeyError(f"mock-morph geometry: no preset named {preset_name!r}")
        await self.goto_positions(self._presets[preset_name])

    async def goto_positions(self, positions: np.ndarray) -> None:
        """Command the array to ``positions``; applies simulated positioning error."""
        target = np.asarray(positions, dtype=np.float64)
        if target.shape != self._commanded.shape:
            raise ValueError(
                f"mock-morph geometry: position shape {target.shape} does not match "
                f"array shape {self._commanded.shape}"
            )
        self._commanded = target.copy()
        sigma_m = self._repeatability_mm / 1000.0
        noise = self._rng.normal(loc=0.0, scale=sigma_m, size=target.shape)
        self._actual = target + noise

    async def list_presets(self) -> list[str]:
        """List preset names sorted alphabetically."""
        return sorted(self._presets)

    async def save_preset(self, name: str, positions: np.ndarray) -> None:
        """Persist ``positions`` under ``name`` in the preset TOML."""
        arr = np.asarray(positions, dtype=np.float64)
        if arr.shape != self._commanded.shape:
            raise ValueError(
                f"mock-morph geometry: preset shape {arr.shape} does not match "
                f"array shape {self._commanded.shape}"
            )
        self._presets[name] = arr.copy()
        self._write_presets()

    async def calibrate(self) -> CalibrationReport:
        """Simulate a rail-end-stop home routine."""
        # Resets commanded position to the initial configuration; actual snaps to commanded.
        self._actual = self._commanded.copy()
        return CalibrationReport(
            ok=True,
            message=f"mock-morph geometry: homed {self.num_antennas} axes",
            backend="mock-morph",
            details={"repeatability_mm": self._repeatability_mm},
        )

    # ------------------------------------------------------------------
    # Preset persistence (TOML)
    # ------------------------------------------------------------------

    def _load_presets(self) -> None:
        if not self._presets_path.is_file():
            return
        try:
            import tomllib

            with self._presets_path.open("rb") as fh:
                data = tomllib.load(fh)
        except OSError as exc:
            _LOG.warning(
                "mock_morph_presets_unreadable", path=str(self._presets_path), error=str(exc)
            )
            return
        loaded: dict[str, np.ndarray] = {}
        for name, positions in data.get("presets", {}).items():
            try:
                arr = np.asarray(positions, dtype=np.float64)
                if arr.shape == self._commanded.shape:
                    loaded[name] = arr
            except (TypeError, ValueError):
                _LOG.warning("mock_morph_preset_malformed", name=name)
        self._presets = loaded

    def _write_presets(self) -> None:
        self._presets_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["[presets]\n"]
        for name in sorted(self._presets):
            arr = self._presets[name]
            rows = ", ".join("[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in arr.tolist())
            lines.append(f"{name} = [{rows}]\n")
        self._presets_path.write_text("".join(lines))


def create(
    initial_positions: Sequence[Sequence[float]] | None = None,
    *,
    repeatability_mm: float = 0.5,
    presets_path: Path | None = None,
    seed: int = 0,
    **_: Any,
) -> MockMorphGeometry:
    """Factory wired into the ``rfdf.backends.geometry`` entry-point.

    Args:
        initial_positions: Starting antenna positions; defaults to the
            pentagonal 5-antenna reference array if ``None``.
        repeatability_mm: Positioning repeatability noise (1-sigma in mm).
        presets_path: Override for the preset TOML location.
        seed: RNG seed.

    Returns:
        Configured :class:`MockMorphGeometry`.
    """
    if initial_positions is None:
        initial_positions = [
            (0.0, 0.0, 0.0),
            (0.17, 0.0, 0.0),
            (0.0, 0.17, 0.0),
            (-0.17, 0.0, 0.0),
            (0.0, -0.17, 0.0),
        ]
    return MockMorphGeometry(
        initial_positions=initial_positions,
        repeatability_mm=repeatability_mm,
        presets_path=presets_path,
        seed=seed,
    )

"""GRBL linear-rail geometry backend — the morphing array made real.

Each antenna rides a motorized linear rail driven by a stepper + closed-loop
driver behind a GRBL_ESP32 controller. Rails may sit at arbitrary angles in 3D
space; each rail's axis is fixed at construction by a :class:`RailConfig`. This
backend implements the Stage-2 ``GeometryController`` HAL contract — the same
contract the ``static`` and ``mock-morph`` geometries pass.

``goto_positions`` projects each requested 3-D antenna position onto its rail
axis and commands the corresponding GRBL axis; a position that does not lie on
its configured rail axis is rejected.

The GRBL HTTP transport (``rfdf.backends._grbl``) lazy-imports ``httpx``;
``import rfdf`` and base discovery work without the ``[geometry-grbl]`` extra.

Registered as the ``grbl-linear`` geometry backend.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import platformdirs
import structlog
from pydantic import BaseModel, field_validator

from rfdf.backends._grbl import GrblConnectionError, GrblHttpClient
from rfdf.hal.types import CalibrationReport

_LOG = structlog.get_logger(__name__)

#: GRBL axis letters in machine-position report order.
_AXIS_ORDER = ("X", "Y", "Z", "A", "B", "C")

#: A requested position this far (metres) off its rail axis is rejected.
_ON_AXIS_TOLERANCE_M = 1e-4


class RailConfig(BaseModel):
    """Fixed configuration of one linear rail carrying one antenna.

    Attributes:
        antenna_id: Index of the antenna this rail carries.
        origin: Rail-start position ``(x, y, z)`` in metres (offset 0).
        direction: Unit vector along the rail; normalised on validation.
        travel_m: Maximum travel from ``origin`` along ``direction``.
        grbl_axis: GRBL axis letter driving this rail (``X``/``Y``/``Z``/``A``/``B``).
    """

    antenna_id: int
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    travel_m: float
    grbl_axis: str

    @field_validator("direction")
    @classmethod
    def _normalise_direction(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        """Reject a zero vector; normalise to unit length."""
        norm = math.sqrt(sum(c * c for c in value))
        if norm == 0.0:
            raise ValueError("rail direction must be a non-zero vector")
        return (value[0] / norm, value[1] / norm, value[2] / norm)

    @field_validator("grbl_axis")
    @classmethod
    def _check_axis(cls, value: str) -> str:
        """Reject an unknown GRBL axis letter."""
        axis = value.strip().upper()
        if axis not in _AXIS_ORDER:
            raise ValueError(f"grbl_axis must be one of {_AXIS_ORDER}, got {value!r}")
        return axis


class PositionRepeatabilityReport(BaseModel):
    """Result of a position-error-budget commissioning run.

    Attributes:
        iterations: Number of move repetitions per preset.
        max_deviation_mm: Worst single-axis deviation from commanded, mm.
        rms_deviation_mm: RMS deviation across every measurement, mm.
        per_rail_max_mm: Worst deviation per antenna, mm.
        within_budget: True when ``max_deviation_mm`` < ``budget_mm``.
        budget_mm: The acceptance budget the run was judged against.
    """

    iterations: int
    max_deviation_mm: float
    rms_deviation_mm: float
    per_rail_max_mm: list[float]
    within_budget: bool
    budget_mm: float = 1.0


def _solve_rail_offset(target: np.ndarray, rail: RailConfig) -> float:
    """Project ``target`` onto a rail axis and return the offset from origin.

    Args:
        target: Desired antenna position ``(x, y, z)`` in metres.
        rail: The rail configuration.

    Returns:
        The offset along ``rail.direction`` from ``rail.origin``, in metres.

    Raises:
        ValueError: If ``target`` does not lie on the rail axis, or the required
            offset is outside ``[0, travel_m]``.
    """
    origin = np.asarray(rail.origin, dtype=np.float64)
    direction = np.asarray(rail.direction, dtype=np.float64)
    delta = target - origin
    offset = float(delta @ direction)
    # Closest point on the (infinite) rail line to the target.
    on_axis = origin + offset * direction
    residual = float(np.linalg.norm(target - on_axis))
    if residual > _ON_AXIS_TOLERANCE_M:
        raise ValueError(
            f"antenna {rail.antenna_id}: target {tuple(target)} lies "
            f"{residual * 1e3:.2f} mm off rail axis {rail.grbl_axis} — the GRBL "
            f"linear-rail backend can only place antennas on their configured rails."
        )
    if offset < -_ON_AXIS_TOLERANCE_M or offset > rail.travel_m + _ON_AXIS_TOLERANCE_M:
        raise ValueError(
            f"antenna {rail.antenna_id}: rail offset {offset:.4f} m is outside the "
            f"reachable travel [0, {rail.travel_m}] m."
        )
    return offset


def _offset_to_position(rail: RailConfig, offset_m: float) -> np.ndarray:
    """Return the 3-D position of an antenna at ``offset_m`` along its rail."""
    origin = np.asarray(rail.origin, dtype=np.float64)
    direction = np.asarray(rail.direction, dtype=np.float64)
    return origin + offset_m * direction


class GrblLinearGeometry:
    """N motorized linear rails, each carrying one antenna.

    Args:
        rails: One :class:`RailConfig` per antenna. ``antenna_id`` values must
            form ``0..N-1``; each rail must drive a distinct GRBL axis.
        controller_host: GRBL controller IP/hostname.
        controller_port: GRBL controller HTTP port.
        positioning_repeatability_mm: Measured rail repeatability (1-sigma, mm).
        presets_path: TOML preset store; defaults to
            ``~/.config/rfdf/geometry-presets.toml``.
        command_path: GRBL_ESP32 command-endpoint template.
    """

    def __init__(
        self,
        rails: Sequence[RailConfig],
        *,
        controller_host: str,
        controller_port: int = 80,
        positioning_repeatability_mm: float = 0.05,
        presets_path: Path | None = None,
        command_path: str = "/command?commandText={cmd}",
    ) -> None:
        """Validate the rail set; defer all network I/O to the first command."""
        if not rails:
            raise ValueError("GrblLinearGeometry requires at least one rail")
        if not controller_host:
            raise ValueError("GrblLinearGeometry requires a controller_host")
        ids = sorted(r.antenna_id for r in rails)
        if ids != list(range(len(rails))):
            raise ValueError(f"rail antenna_id values must be 0..{len(rails) - 1}, got {ids}")
        axes = [r.grbl_axis for r in rails]
        if len(set(axes)) != len(axes):
            raise ValueError(f"each rail must drive a distinct GRBL axis, got {axes}")
        self._rails = sorted(rails, key=lambda r: r.antenna_id)
        self._grbl = GrblHttpClient(
            host=controller_host, port=controller_port, command_path=command_path
        )
        self._repeatability_mm = float(positioning_repeatability_mm)
        self._presets_path = presets_path or (
            Path(platformdirs.user_config_path("rfdf")) / "geometry-presets.toml"
        )
        # Commanded offsets along each rail; start at the origin end.
        self._offsets: list[float] = [0.0] * len(self._rails)
        self._presets: dict[str, np.ndarray] = {}
        self._load_presets()

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def num_antennas(self) -> int:
        """Number of antennas — one per rail."""
        return len(self._rails)

    @property
    def is_morphable(self) -> bool:
        """Motorized rails — positions are commandable at runtime."""
        return True

    @property
    def positioning_repeatability_mm(self) -> float:
        """Measured rail positioning repeatability (1-sigma, mm)."""
        return self._repeatability_mm

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    async def positions(self) -> np.ndarray:
        """Return current antenna positions ``(N, 3)`` in metres.

        Reads the GRBL machine position and maps each rail's axis coordinate to
        a 3-D position; falls back to the last commanded offsets if the
        controller is unreachable.
        """
        offsets = self._offsets
        try:
            status = await self._grbl.query_status()
            offsets = [self._axis_value(status.machine_pos, rail) for rail in self._rails]
        except GrblConnectionError as exc:
            _LOG.warning("grbl_linear_positions_offline", error=str(exc))
        rows = [
            _offset_to_position(rail, off) for rail, off in zip(self._rails, offsets, strict=True)
        ]
        return np.asarray(rows, dtype=np.float64)

    async def goto_positions(self, positions: np.ndarray) -> None:
        """Command the array to ``positions``.

        Raises:
            ValueError: If the shape is wrong, or any position does not lie on
                its rail axis / is outside the rail travel.
        """
        target = np.asarray(positions, dtype=np.float64)
        if target.shape != (self.num_antennas, 3):
            raise ValueError(
                f"grbl-linear geometry: expected ({self.num_antennas}, 3) positions, "
                f"got shape {target.shape}"
            )
        offsets = [_solve_rail_offset(target[i], rail) for i, rail in enumerate(self._rails)]
        # One absolute move: GRBL axis = rail offset in millimetres.
        words = " ".join(
            f"{rail.grbl_axis}{off * 1e3:.4f}"
            for rail, off in zip(self._rails, offsets, strict=True)
        )
        await self._grbl.send(f"G90 G0 {words}")
        self._offsets = offsets

    async def goto_preset(self, preset_name: str) -> None:
        """Move to the antenna positions stored under ``preset_name``."""
        if preset_name not in self._presets:
            raise KeyError(f"grbl-linear geometry: no preset named {preset_name!r}")
        await self.goto_positions(self._presets[preset_name])

    async def list_presets(self) -> list[str]:
        """List preset names sorted alphabetically."""
        return sorted(self._presets)

    def preset_positions(self, name: str) -> np.ndarray:
        """Return the commanded positions stored under preset ``name``.

        Raises:
            KeyError: If no preset of that name exists.
        """
        if name not in self._presets:
            raise KeyError(f"grbl-linear geometry: no preset named {name!r}")
        return self._presets[name].copy()

    async def save_preset(self, name: str, positions: np.ndarray) -> None:
        """Persist ``positions`` under ``name`` in the preset TOML."""
        arr = np.asarray(positions, dtype=np.float64)
        if arr.shape != (self.num_antennas, 3):
            raise ValueError(
                f"grbl-linear geometry: preset shape {arr.shape} does not match "
                f"({self.num_antennas}, 3)"
            )
        # Validate every preset position is reachable before persisting it.
        for i, rail in enumerate(self._rails):
            _solve_rail_offset(arr[i], rail)
        self._presets[name] = arr.copy()
        self._write_presets()

    async def calibrate(self) -> CalibrationReport:
        """Home every rail to its origin-end limit switch and zero the encoders."""
        try:
            await self._grbl.home()
        except GrblConnectionError as exc:
            return CalibrationReport(
                ok=False,
                message=f"grbl-linear geometry: homing failed — {exc}",
                backend="grbl-linear",
            )
        self._offsets = [0.0] * len(self._rails)
        return CalibrationReport(
            ok=True,
            message=f"grbl-linear geometry: homed {self.num_antennas} rail(s)",
            backend="grbl-linear",
            details={
                "repeatability_mm": self._repeatability_mm,
                "axes": [rail.grbl_axis for rail in self._rails],
            },
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._grbl.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_value(machine_pos: tuple[float, ...], rail: RailConfig) -> float:
        """Return ``rail``'s axis coordinate (metres) from a GRBL MPos tuple."""
        index = _AXIS_ORDER.index(rail.grbl_axis)
        if index >= len(machine_pos):
            return 0.0
        return machine_pos[index] / 1e3  # GRBL reports millimetres

    def _load_presets(self) -> None:
        """Load presets from the TOML store, skipping malformed entries."""
        if not self._presets_path.is_file():
            return
        try:
            with self._presets_path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            _LOG.warning("grbl_linear_presets_unreadable", error=str(exc))
            return
        loaded: dict[str, np.ndarray] = {}
        for name, body in data.get("preset", {}).items():
            positions = body.get("positions") if isinstance(body, dict) else None
            if positions is None:
                continue
            try:
                arr = np.asarray(positions, dtype=np.float64)
            except (TypeError, ValueError):
                _LOG.warning("grbl_linear_preset_malformed", name=name)
                continue
            if arr.shape == (self.num_antennas, 3):
                loaded[name] = arr
        self._presets = loaded

    def _write_presets(self) -> None:
        """Persist presets to the TOML store in the documented schema."""
        self._presets_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for name in sorted(self._presets):
            rows = self._presets[name].tolist()
            body = ",\n    ".join("[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in rows)
            lines.append(f"[preset.{name}]\npositions = [\n    {body},\n]\n")
        self._presets_path.write_text("\n".join(lines))


async def measure_position_repeatability(
    geometry: GrblLinearGeometry,
    *,
    iterations: int = 50,
    budget_mm: float = 1.0,
) -> PositionRepeatabilityReport:
    """Move to every preset ``iterations`` times and report position deviation.

    A commissioning routine — run once after the mechanical build, not in CI.
    For each preset, the array is commanded ``iterations`` times and the actual
    position read back; the report captures the max + RMS deviation from the
    commanded position. The Stage-5 acceptance budget is 1 mm (sufficient for
    synthetic aperture at 5.8 GHz).

    Named ``measure_*`` rather than the Stage-5 PDF's ``test_position_*`` so
    pytest never collects it as a test (see STAGE-5-OUTPUTS deviations).

    Args:
        geometry: A homed :class:`GrblLinearGeometry`.
        iterations: Move repetitions per preset.
        budget_mm: Acceptance budget the run is judged against.

    Returns:
        The :class:`PositionRepeatabilityReport`.
    """
    presets = await geometry.list_presets()
    n = geometry.num_antennas
    per_rail_max = np.zeros(n, dtype=np.float64)
    squares: list[float] = []
    worst = 0.0
    for preset in presets:
        commanded = geometry.preset_positions(preset)
        for _ in range(iterations):
            await geometry.goto_preset(preset)
            actual = await geometry.positions()
            deviation_mm = np.abs(actual - commanded) * 1e3
            per_rail = deviation_mm.max(axis=1)
            per_rail_max = np.maximum(per_rail_max, per_rail)
            worst = max(worst, float(deviation_mm.max()))
            squares.extend((deviation_mm.ravel() ** 2).tolist())
    rms = float(math.sqrt(sum(squares) / len(squares))) if squares else 0.0
    return PositionRepeatabilityReport(
        iterations=iterations,
        max_deviation_mm=worst,
        rms_deviation_mm=rms,
        per_rail_max_mm=per_rail_max.tolist(),
        within_budget=worst < budget_mm,
        budget_mm=budget_mm,
    )


def create(
    *,
    rails: Sequence[RailConfig | dict[str, Any]] | None = None,
    controller_host: str | None = None,
    controller_port: int = 80,
    positioning_repeatability_mm: float = 0.05,
    presets_path: Path | None = None,
    command_path: str = "/command?commandText={cmd}",
    **_: Any,
) -> GrblLinearGeometry:
    """Factory wired into the ``rfdf.backends.geometry`` ``grbl-linear`` entry-point.

    Args:
        rails: One :class:`RailConfig` (or equivalent dict) per antenna.
        controller_host: GRBL controller IP/hostname. Required — site-specific,
            kept in the operator's config, never committed.
        controller_port: GRBL controller HTTP port.
        positioning_repeatability_mm: Measured rail repeatability.
        presets_path: Override for the preset TOML location.
        command_path: GRBL_ESP32 command-endpoint template.

    Returns:
        A configured :class:`GrblLinearGeometry`.
    """
    if not rails:
        raise ValueError(
            "grbl-linear: rails is required — one RailConfig per antenna, "
            "configured from the operator's mechanical build."
        )
    if not controller_host:
        raise ValueError(
            "grbl-linear: controller_host is required — set it in ~/.config/rfdf/config.toml."
        )
    parsed = [r if isinstance(r, RailConfig) else RailConfig(**r) for r in rails]
    return GrblLinearGeometry(
        parsed,
        controller_host=controller_host,
        controller_port=controller_port,
        positioning_repeatability_mm=positioning_repeatability_mm,
        presets_path=presets_path,
        command_path=command_path,
    )


__all__ = [
    "GrblLinearGeometry",
    "PositionRepeatabilityReport",
    "RailConfig",
    "create",
    "measure_position_repeatability",
]

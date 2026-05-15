"""Array calibration framework.

A :class:`Calibration` bundles the per-channel complex gain corrections and the
mutual-coupling matrix that turn raw array IQ into the clean signal a DOA estimator
expects, plus provenance recording how it was produced. Three procedures build one:

* :func:`load_simulated` — an ideal identity calibration for mock development.
* :func:`calibrate_pilot_tone` — estimates per-channel gain and phase from a received
  CW pilot (the mock's :class:`~rfdf.backends.sdr.mock.PilotTone` at zenith).
* :func:`calibrate_mutual_coupling` — builds the coupling matrix from measured array
  S-parameters.

Calibrations persist as a NumPy ``.npz`` (the matrices) plus a TOML sidecar (the
metadata) under ``platformdirs`` user data. ``scikit-rf`` — needed only to read a
Touchstone file in :func:`load_s_parameters` — is imported lazily so the base
``import rfdf`` stays free of domain dependencies.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import platformdirs
from numpy.typing import ArrayLike, NDArray

from rfdf.dsp.errors import CalibrationError
from rfdf.dsp.geometry_presets import validate_positions
from rfdf.hal.sdr import SdrSource


def geometry_hash(positions: ArrayLike) -> str:
    """Return a short stable hash of an antenna geometry.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.

    Returns:
        The first 16 hex characters of the SHA-256 of the rounded positions.
    """
    pos = validate_positions(positions)
    rounded = np.round(pos, 9)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def calibrations_dir() -> Path:
    """Return the default directory calibrations are saved to and loaded from."""
    return platformdirs.user_data_path("rfdf") / "calibrations"


@dataclass(frozen=True)
class CalibrationProvenance:
    """Records how, when, and against what a calibration was produced.

    Attributes:
        procedure: The procedure name (``simulated`` / ``pilot_tone`` / ``mutual_coupling``).
        backend: The SDR source class name, or empty when not applicable.
        geometry_hash: :func:`geometry_hash` of the array the calibration is valid for.
        timestamp: ISO-8601 UTC time the calibration was produced.
        operator: Free-text operator identifier.
    """

    procedure: str
    backend: str
    geometry_hash: str
    timestamp: str
    operator: str


@dataclass(frozen=True, eq=False)
class Calibration:
    """Per-channel gain plus mutual-coupling corrections for one frequency.

    The forward array model is ``x = gain ⊙ (C @ s_clean)``; :meth:`apply` inverts it
    as ``s_est = C⁻¹ (x ⊙ channel_gains)``, where ``channel_gains`` are correction
    multipliers (the reciprocals of the channel errors).

    Attributes:
        frequency_hz: RF frequency the calibration is valid at.
        channel_gains: ``(M,)`` complex per-channel correction multipliers.
        coupling: ``(M, M)`` complex mutual-coupling matrix ``C``.
        provenance: How the calibration was produced.
    """

    frequency_hz: float
    channel_gains: NDArray[np.complex128]
    coupling: NDArray[np.complex128]
    provenance: CalibrationProvenance

    @property
    def num_channels(self) -> int:
        """Number of antenna channels the calibration covers."""
        return int(self.channel_gains.shape[0])

    def apply(self, iq: ArrayLike) -> NDArray[np.complex128]:
        """Correct an IQ block for per-channel gain and mutual coupling.

        Args:
            iq: Raw IQ, shape ``(M, N)``.

        Returns:
            The corrected ``(M, N)`` complex128 IQ.

        Raises:
            CalibrationError: If ``iq`` is not 2-D or its channel count is wrong.
        """
        data = np.asarray(iq, dtype=np.complex128)
        if data.ndim != 2:
            raise CalibrationError(f"expected (M, N) IQ, got ndim {data.ndim}")
        if data.shape[0] != self.num_channels:
            raise CalibrationError(
                f"IQ has {data.shape[0]} channels, calibration covers {self.num_channels}"
            )
        gain_corrected = data * self.channel_gains[:, None]
        decoupled: NDArray[np.complex128] = np.linalg.solve(self.coupling, gain_corrected).astype(
            np.complex128
        )
        return decoupled

    def matches_geometry(self, positions: ArrayLike) -> bool:
        """Return whether the calibration was built for the given geometry.

        Args:
            positions: Antenna positions, shape ``(M, 3)`` in metres.

        Returns:
            ``True`` if the geometry hash matches the calibration's provenance.
        """
        return geometry_hash(positions) == self.provenance.geometry_hash

    def save(self, name: str, *, directory: Path | None = None) -> Path:
        """Persist the calibration as an ``.npz`` plus a ``.toml`` sidecar.

        Args:
            name: Base filename (no extension).
            directory: Destination directory; defaults to :func:`calibrations_dir`.

        Returns:
            The path of the written TOML metadata file.
        """
        base = directory if directory is not None else calibrations_dir()
        base.mkdir(parents=True, exist_ok=True)
        np.savez(
            base / f"{name}.npz",
            channel_gains=self.channel_gains,
            coupling=self.coupling,
        )
        toml_path = base / f"{name}.toml"
        _write_flat_toml(
            toml_path,
            {
                "frequency_hz": self.frequency_hz,
                "procedure": self.provenance.procedure,
                "backend": self.provenance.backend,
                "geometry_hash": self.provenance.geometry_hash,
                "timestamp": self.provenance.timestamp,
                "operator": self.provenance.operator,
            },
        )
        return toml_path

    @classmethod
    def load(cls, name: str, *, directory: Path | None = None) -> Calibration:
        """Load a calibration previously written by :meth:`save`.

        Args:
            name: Base filename used at save time.
            directory: Directory to read from; defaults to :func:`calibrations_dir`.

        Returns:
            The reconstructed :class:`Calibration`.

        Raises:
            CalibrationError: If either the ``.npz`` or the ``.toml`` file is missing.
        """
        base = directory if directory is not None else calibrations_dir()
        npz_path = base / f"{name}.npz"
        toml_path = base / f"{name}.toml"
        if not npz_path.exists() or not toml_path.exists():
            raise CalibrationError(f"calibration {name!r} not found in {base}")
        with np.load(npz_path) as data:
            channel_gains = data["channel_gains"].astype(np.complex128)
            coupling = data["coupling"].astype(np.complex128)
        meta = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        provenance = CalibrationProvenance(
            procedure=str(meta["procedure"]),
            backend=str(meta["backend"]),
            geometry_hash=str(meta["geometry_hash"]),
            timestamp=str(meta["timestamp"]),
            operator=str(meta["operator"]),
        )
        return cls(
            frequency_hz=float(meta["frequency_hz"]),
            channel_gains=channel_gains,
            coupling=coupling,
            provenance=provenance,
        )


def _write_flat_toml(path: Path, data: dict[str, str | float]) -> None:
    """Write a flat string/number mapping as a minimal TOML file."""
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        else:
            lines.append(f"{key} = {value!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def load_simulated(positions: ArrayLike, freq_hz: float, *, operator: str = "") -> Calibration:
    """Build an ideal identity calibration for development against the mock.

    Args:
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        freq_hz: RF frequency the calibration is tagged with.
        operator: Optional operator identifier for provenance.

    Returns:
        A :class:`Calibration` with unit gains and identity coupling.
    """
    pos = validate_positions(positions)
    num_channels = pos.shape[0]
    provenance = CalibrationProvenance(
        procedure="simulated",
        backend="",
        geometry_hash=geometry_hash(pos),
        timestamp=_now_iso(),
        operator=operator,
    )
    return Calibration(
        frequency_hz=freq_hz,
        channel_gains=np.ones(num_channels, dtype=np.complex128),
        coupling=np.eye(num_channels, dtype=np.complex128),
        provenance=provenance,
    )


async def calibrate_pilot_tone(
    sdr: SdrSource,
    *,
    freq_hz: float,
    duration_s: float,
    power_dbm: float = 0.0,
    positions: ArrayLike | None = None,
    operator: str = "",
) -> Calibration:
    """Estimate per-channel gain and phase corrections from a received CW pilot.

    Requests a pilot tone, streams ``duration_s`` of IQ, and aligns every channel to
    channel 0. Assumes a planar array — the zenith pilot then has a flat wavefront, so
    any per-channel amplitude or phase difference is a channel error, not geometry.

    Args:
        sdr: A configured SDR source.
        freq_hz: Pilot frequency in hertz.
        duration_s: How much IQ to integrate over.
        power_dbm: Pilot power; must respect the configured EIRP cap.
        positions: Optional array geometry, recorded in the provenance hash.
        operator: Optional operator identifier for provenance.

    Returns:
        A :class:`Calibration` with identity coupling and per-channel gain corrections.

    Raises:
        CalibrationError: If no IQ is received.
    """
    await sdr.calibration_pilot(freq_hz, power_dbm)
    await sdr.start()
    blocks: list[NDArray[np.complex128]] = []
    collected = 0
    target: int | None = None
    try:
        async for block in sdr.stream():
            blocks.append(block.iq.astype(np.complex128))
            collected += block.iq.shape[1]
            if target is None:
                target = max(1, int(duration_s * block.sample_rate_hz))
            if collected >= target:
                break
    finally:
        await sdr.stop()
    if not blocks:
        raise CalibrationError("pilot-tone calibration received no IQ")

    iq = np.concatenate(blocks, axis=1)
    coefficients = np.mean(iq, axis=1)
    channel_gains: NDArray[np.complex128] = (coefficients[0] / coefficients).astype(np.complex128)
    num_channels = iq.shape[0]
    provenance = CalibrationProvenance(
        procedure="pilot_tone",
        backend=type(sdr).__name__,
        geometry_hash=geometry_hash(positions) if positions is not None else "",
        timestamp=_now_iso(),
        operator=operator,
    )
    return Calibration(
        frequency_hz=freq_hz,
        channel_gains=channel_gains,
        coupling=np.eye(num_channels, dtype=np.complex128),
        provenance=provenance,
    )


def calibrate_mutual_coupling(
    s_parameters: ArrayLike, *, freq_hz: float, operator: str = ""
) -> Calibration:
    """Build a mutual-coupling calibration from an array S-parameter matrix.

    Models the coupling matrix as ``C = I + (S - diag(S))`` — unit diagonal plus the
    measured off-diagonal port coupling. This is the standard first-order coupling
    model; full electromagnetic decoupling is out of scope.

    Args:
        s_parameters: The ``(M, M)`` complex array S-matrix at ``freq_hz``.
        freq_hz: RF frequency the S-parameters were measured at.
        operator: Optional operator identifier for provenance.

    Returns:
        A :class:`Calibration` with unit gains and the derived coupling matrix.

    Raises:
        CalibrationError: If ``s_parameters`` is not a square matrix.
    """
    s_matrix = np.asarray(s_parameters, dtype=np.complex128)
    if s_matrix.ndim != 2 or s_matrix.shape[0] != s_matrix.shape[1]:
        raise CalibrationError(
            f"S-parameter matrix must be square (M, M), got shape {s_matrix.shape}"
        )
    num_channels = s_matrix.shape[0]
    off_diagonal = s_matrix - np.diag(np.diag(s_matrix))
    coupling: NDArray[np.complex128] = np.eye(num_channels, dtype=np.complex128) + off_diagonal
    provenance = CalibrationProvenance(
        procedure="mutual_coupling",
        backend="",
        geometry_hash="",
        timestamp=_now_iso(),
        operator=operator,
    )
    return Calibration(
        frequency_hz=freq_hz,
        channel_gains=np.ones(num_channels, dtype=np.complex128),
        coupling=coupling,
        provenance=provenance,
    )


def load_s_parameters(touchstone_path: Path | str, freq_hz: float) -> NDArray[np.complex128]:
    """Read an array S-matrix from a Touchstone file at the nearest frequency.

    ``scikit-rf`` is imported lazily here so the base package never depends on it.

    Args:
        touchstone_path: Path to a Touchstone (``.sNp``) file.
        freq_hz: Frequency of interest in hertz.

    Returns:
        The ``(M, M)`` complex128 S-matrix at the nearest available frequency.

    Raises:
        CalibrationError: If ``scikit-rf`` is not installed or the file is missing.
    """
    try:
        import skrf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise CalibrationError(
            "scikit-rf is required to read Touchstone files; install rfdf[antenna]"
        ) from exc
    path = Path(touchstone_path)
    if not path.exists():
        raise CalibrationError(f"Touchstone file not found: {path}")
    network = skrf.Network(str(path))
    index = int(np.argmin(np.abs(np.asarray(network.f, dtype=np.float64) - freq_hz)))
    s_matrix: NDArray[np.complex128] = np.asarray(network.s[index], dtype=np.complex128)
    return s_matrix

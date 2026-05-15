"""Unit tests for rfdf.dsp.calibration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from rfdf.backends.sdr.mock import create as create_mock_sdr
from rfdf.dsp.calibration import (
    Calibration,
    calibrate_mutual_coupling,
    calibrate_pilot_tone,
    calibrations_dir,
    geometry_hash,
    load_s_parameters,
    load_simulated,
)
from rfdf.dsp.errors import CalibrationError
from rfdf.dsp.geometry_presets import planar_cross, ula
from rfdf.hal import SdrConfig


async def _grab_block(sdr: object) -> np.ndarray:  # type: ignore[type-arg]
    """Start, capture one StreamBlock, stop."""
    await sdr.start()  # type: ignore[attr-defined]
    async for block in sdr.stream():  # type: ignore[attr-defined]
        await sdr.stop()  # type: ignore[attr-defined]
        return block.iq.astype(np.complex128)
    raise AssertionError("no block")


def test_load_simulated_is_an_identity_calibration() -> None:
    """load_simulated yields unit gains and identity coupling."""
    cal = load_simulated(planar_cross(0.17), freq_hz=868e6)
    assert cal.num_channels == 5
    assert cal.provenance.procedure == "simulated"
    np.testing.assert_allclose(cal.channel_gains, 1.0)
    np.testing.assert_allclose(cal.coupling, np.eye(5))


def test_apply_identity_calibration_returns_input_unchanged() -> None:
    """Applying an identity calibration is a no-op."""
    cal = load_simulated(ula(4, 0.1), freq_hz=868e6)
    rng = np.random.default_rng(0)
    iq = rng.standard_normal((4, 64)) + 1j * rng.standard_normal((4, 64))
    np.testing.assert_allclose(cal.apply(iq), iq, atol=1e-12)


def test_apply_rejects_channel_count_mismatch() -> None:
    """Applying a 5-channel calibration to 3-channel IQ raises."""
    cal = load_simulated(planar_cross(0.17), freq_hz=868e6)
    with pytest.raises(CalibrationError, match="channels"):
        cal.apply(np.zeros((3, 32), dtype=np.complex128))


def test_geometry_hash_is_deterministic_and_distinguishing() -> None:
    """The geometry hash is stable for a geometry and differs across geometries."""
    cross = planar_cross(0.17)
    assert geometry_hash(cross) == geometry_hash(cross.copy())
    assert geometry_hash(cross) != geometry_hash(ula(5, 0.17))


def test_calibration_matches_its_own_geometry() -> None:
    """matches_geometry is true for the geometry the calibration was built for."""
    cross = planar_cross(0.17)
    cal = load_simulated(cross, freq_hz=868e6)
    assert cal.matches_geometry(cross) is True
    assert cal.matches_geometry(ula(5, 0.17)) is False


def test_calibrate_pilot_tone_equalizes_channels() -> None:
    """Pilot-tone calibration removes per-channel gain and phase errors."""
    sdr = create_mock_sdr(
        gain_errors_db=[3.0, -2.0, 1.5, 4.0, -1.0],
        mock_b210_behavior=True,
        seed=11,
    )
    asyncio.run(
        sdr.configure(SdrConfig(center_freq_hz=868e6, sample_rate_hz=2e6, channels=list(range(5))))
    )
    cal = asyncio.run(calibrate_pilot_tone(sdr, freq_hz=868e6, duration_s=0.05))
    assert cal.num_channels == 5
    assert cal.provenance.procedure == "pilot_tone"

    verify_iq = asyncio.run(_grab_block(sdr))
    coeffs = np.mean(cal.apply(verify_iq), axis=1)
    np.testing.assert_allclose(coeffs / coeffs[0], 1.0, atol=0.02)


def test_calibrate_mutual_coupling_builds_coupling_from_s_parameters() -> None:
    """The coupling matrix is unit-diagonal plus the off-diagonal S-parameters."""
    s_matrix = np.zeros((5, 5), dtype=np.complex128)
    s_matrix[0, 1] = s_matrix[1, 0] = 0.12
    s_matrix[3, 4] = s_matrix[4, 3] = 0.07
    cal = calibrate_mutual_coupling(s_matrix, freq_hz=868e6)
    np.testing.assert_allclose(cal.coupling, np.eye(5) + s_matrix)
    assert cal.provenance.procedure == "mutual_coupling"


def test_calibrate_mutual_coupling_apply_recovers_signal() -> None:
    """Applying a coupling calibration inverts a known coupling matrix."""
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((5, 200)) + 1j * rng.standard_normal((5, 200))
    s_matrix = np.zeros((5, 5), dtype=np.complex128)
    s_matrix[0, 1] = s_matrix[1, 0] = 0.12
    s_matrix[3, 4] = s_matrix[4, 3] = 0.07
    coupling = np.eye(5) + s_matrix
    cal = calibrate_mutual_coupling(s_matrix, freq_hz=868e6)
    np.testing.assert_allclose(cal.apply(coupling @ clean), clean, atol=1e-9)


def test_calibrate_mutual_coupling_rejects_nonsquare_s() -> None:
    """A non-square S-parameter matrix is rejected."""
    with pytest.raises(CalibrationError, match="square"):
        calibrate_mutual_coupling(np.zeros((3, 5), dtype=np.complex128), freq_hz=868e6)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """A calibration round-trips through disk unchanged."""
    s_matrix = np.zeros((5, 5), dtype=np.complex128)
    s_matrix[0, 1] = s_matrix[1, 0] = 0.2
    cal = calibrate_mutual_coupling(s_matrix, freq_hz=868e6, operator="louis")
    cal.save("array_a", directory=tmp_path)
    loaded = Calibration.load("array_a", directory=tmp_path)
    assert loaded.frequency_hz == cal.frequency_hz
    assert loaded.provenance == cal.provenance
    np.testing.assert_allclose(loaded.channel_gains, cal.channel_gains)
    np.testing.assert_allclose(loaded.coupling, cal.coupling)


def test_load_missing_calibration_raises(tmp_path: Path) -> None:
    """Loading a calibration that was never saved raises a clear error."""
    with pytest.raises(CalibrationError, match="not found"):
        Calibration.load("nonexistent", directory=tmp_path)


def test_load_s_parameters_reads_a_touchstone_file(tmp_path: Path) -> None:
    """load_s_parameters extracts the S-matrix at the requested frequency."""
    pytest.importorskip("skrf")
    content = (
        "# HZ S RI R 50\n"
        "800000000 0.0 0.0 0.1 0.0 0.1 0.0 0.0 0.0\n"
        "868000000 0.0 0.0 0.2 0.0 0.2 0.0 0.0 0.0\n"
    )
    path = tmp_path / "coupling.s2p"
    path.write_text(content)
    s_matrix = load_s_parameters(path, freq_hz=868e6)
    assert s_matrix.shape == (2, 2)
    assert abs(s_matrix[1, 0]) == pytest.approx(0.2, abs=1e-6)


def test_apply_rejects_non_2d_iq() -> None:
    """Applying a calibration to a 1-D array raises a clear error."""
    cal = load_simulated(ula(4, 0.1), freq_hz=868e6)
    with pytest.raises(CalibrationError, match="ndim"):
        cal.apply(np.zeros(16, dtype=np.complex128))


def test_calibrations_dir_is_named_calibrations() -> None:
    """The default calibration directory lives under the rfdf user-data path."""
    assert calibrations_dir().name == "calibrations"


def test_load_s_parameters_missing_file_raises(tmp_path: Path) -> None:
    """A missing Touchstone file raises a clear error."""
    pytest.importorskip("skrf")
    with pytest.raises(CalibrationError, match="not found"):
        load_s_parameters(tmp_path / "absent.s2p", freq_hz=868e6)

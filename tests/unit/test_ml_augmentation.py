"""Unit tests for rfdf.ml.datasets.augmentation (pure NumPy, torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from rfdf.ml.datasets.augmentation import AugmentationConfig, MultipathConfig, apply_augmentations
from rfdf.ml.errors import AugmentationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cw(n: int = 1024, freq_hz: float = 1e3, fs_hz: float = 2e6) -> np.ndarray:
    """Return a unit-amplitude CW tone as complex64."""
    t = np.arange(n, dtype=np.float64) / fs_hz
    return np.exp(2j * np.pi * freq_hz * t).astype(np.complex64)


FS = 2e6  # sample rate used throughout


# ---------------------------------------------------------------------------
# No-op: all None config returns input unchanged
# ---------------------------------------------------------------------------


def test_noop_config() -> None:
    """All-None config returns array identical to input."""
    cfg = AugmentationConfig()
    rng = np.random.default_rng(0)
    iq = _make_cw()
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    np.testing.assert_array_equal(out, iq)


# ---------------------------------------------------------------------------
# Shape / dtype contract
# ---------------------------------------------------------------------------


def test_output_shape_matches_input() -> None:
    """Output length equals input length for every enabled impairment."""
    cfg = AugmentationConfig(
        add_awgn=(10.0, 15.0),
        frequency_shift=(-1e3, 1e3),
        gain_variation=(-3.0, 3.0),
        iq_imbalance=(-5.0, 5.0),
        multipath=MultipathConfig(num_taps=3, max_delay_samples=10, gain_spread_db=6.0),
        impulsive_noise=(0.01, 0.1),
        sample_rate_jitter=(-100.0, 100.0),
    )
    rng = np.random.default_rng(7)
    iq = _make_cw(512)
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    assert out.shape == iq.shape, "Output shape must match input shape"


def test_output_dtype_preserved_complex64() -> None:
    """Output dtype matches input dtype for complex64 input."""
    cfg = AugmentationConfig(add_awgn=(5.0, 10.0))
    rng = np.random.default_rng(1)
    iq = _make_cw().astype(np.complex64)
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    assert out.dtype == np.complex64


def test_output_dtype_preserved_complex128() -> None:
    """Output dtype matches input dtype for complex128 input."""
    cfg = AugmentationConfig(gain_variation=(-1.0, 1.0))
    rng = np.random.default_rng(2)
    iq = _make_cw().astype(np.complex128)
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    assert out.dtype == np.complex128


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_2d_input_raises() -> None:
    """2-D input raises AugmentationError."""
    cfg = AugmentationConfig()
    rng = np.random.default_rng(0)
    iq_2d = np.ones((4, 256), dtype=np.complex64)
    with pytest.raises(AugmentationError, match="1-D"):
        apply_augmentations(iq_2d, cfg, rng, sample_rate_hz=FS)


def test_zero_sample_rate_raises() -> None:
    """Non-positive sample rate raises AugmentationError."""
    cfg = AugmentationConfig()
    rng = np.random.default_rng(0)
    iq = _make_cw()
    with pytest.raises(AugmentationError, match="sample_rate_hz"):
        apply_augmentations(iq, cfg, rng, sample_rate_hz=0.0)


# ---------------------------------------------------------------------------
# AWGN: noise power measurably changes SNR
# ---------------------------------------------------------------------------


def test_awgn_reduces_snr() -> None:
    """Adding AWGN at a low SNR makes the output noisier than the input."""
    cfg = AugmentationConfig(add_awgn=(0.0, 0.0))  # exactly 0 dB SNR
    rng = np.random.default_rng(42)
    iq = _make_cw(8192) * 10.0  # high amplitude input
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    # With 0 dB SNR the noise power equals signal power → SNR collapses.
    input_power = float(np.mean(np.abs(iq) ** 2))
    noise_power = float(np.mean(np.abs(out - iq) ** 2))
    assert noise_power > 0.0, "AWGN should add non-zero noise"
    measured_snr = 10.0 * np.log10(input_power / noise_power)
    assert abs(measured_snr) < 3.0, f"Measured SNR {measured_snr:.1f} dB far from 0 dB"


# ---------------------------------------------------------------------------
# Frequency shift: spectral centroid shifts
# ---------------------------------------------------------------------------


def test_frequency_shift_moves_centroid() -> None:
    """A +500 Hz shift moves the spectral centroid positively."""
    cfg = AugmentationConfig(frequency_shift=(500.0, 500.0))
    rng = np.random.default_rng(0)
    iq_in = _make_cw(4096, freq_hz=0.0)  # DC tone
    out = apply_augmentations(iq_in, cfg, rng, sample_rate_hz=FS)
    # FFT centroid of output should be positive (shifted by +500 Hz).
    freqs = np.fft.fftfreq(len(out), d=1.0 / FS)
    power = np.abs(np.fft.fft(out)) ** 2
    centroid = float(np.sum(freqs * power) / (np.sum(power) + 1e-12))
    assert centroid > 0.0, f"Expected positive centroid, got {centroid:.1f} Hz"


# ---------------------------------------------------------------------------
# Gain variation: RMS power changes
# ---------------------------------------------------------------------------


def test_gain_variation_changes_amplitude() -> None:
    """Gain variation of +6 dB approximately doubles the RMS amplitude."""
    cfg = AugmentationConfig(gain_variation=(6.0, 6.0))
    rng = np.random.default_rng(0)
    iq = _make_cw()
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    # Compare mean magnitudes — a CW has near-constant |iq|, so std≈0.
    ratio = float(np.mean(np.abs(out))) / float(np.mean(np.abs(iq)) + 1e-12)
    assert abs(ratio - 2.0) < 0.05, f"Expected x2 amplitude, got ratio {ratio:.3f}"


# ---------------------------------------------------------------------------
# IQ imbalance: Q branch rotates
# ---------------------------------------------------------------------------


def test_iq_imbalance_changes_q() -> None:
    """A non-zero phase imbalance measurably changes the Q branch."""
    cfg = AugmentationConfig(iq_imbalance=(10.0, 10.0))  # 10-degree Q rotation
    rng = np.random.default_rng(0)
    iq = _make_cw()
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    # The imaginary part should differ between input and output.
    assert not np.allclose(np.imag(iq), np.imag(out)), "IQ imbalance had no effect"


# ---------------------------------------------------------------------------
# Multipath: output differs from input
# ---------------------------------------------------------------------------


def test_multipath_changes_iq() -> None:
    """Multipath convolution measurably changes the IQ."""
    cfg = AugmentationConfig(
        multipath=MultipathConfig(num_taps=4, max_delay_samples=8, gain_spread_db=6.0)
    )
    rng = np.random.default_rng(99)
    iq = _make_cw(512)
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    assert not np.allclose(iq, out, atol=1e-4), "Multipath had no effect on IQ"


# ---------------------------------------------------------------------------
# Impulsive noise: some samples differ substantially
# ---------------------------------------------------------------------------


def test_impulsive_noise_injects_spikes() -> None:
    """With probability=0.1 and amplitude=10, some spikes appear."""
    cfg = AugmentationConfig(impulsive_noise=(0.1, 10.0))
    rng = np.random.default_rng(5)
    iq = _make_cw(4096)  # amplitude ≈ 1
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    # Spikes have amplitude ~10, normal samples have amplitude ~1.
    high_amp = np.sum(np.abs(out) > 5.0)
    assert high_amp > 0, "Expected at least one impulse spike"


# ---------------------------------------------------------------------------
# Sample-rate jitter: output differs from input
# ---------------------------------------------------------------------------


def test_sample_rate_jitter_changes_iq() -> None:
    """A ±1000 ppm jitter measurably changes the IQ array."""
    cfg = AugmentationConfig(sample_rate_jitter=(1000.0, 1000.0))
    rng = np.random.default_rng(3)
    iq = _make_cw(512)
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    assert not np.allclose(iq, out, atol=1e-4), "Jitter had no effect"


# ---------------------------------------------------------------------------
# Determinism: same seed → same result
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    """Same seed + config produces identical output on two calls."""
    cfg = AugmentationConfig(
        add_awgn=(5.0, 15.0),
        frequency_shift=(-500.0, 500.0),
        multipath=MultipathConfig(),
    )
    iq = _make_cw(512)
    out1 = apply_augmentations(iq.copy(), cfg, np.random.default_rng(42), sample_rate_hz=FS)
    out2 = apply_augmentations(iq.copy(), cfg, np.random.default_rng(42), sample_rate_hz=FS)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# MultipathConfig validation
# ---------------------------------------------------------------------------


def test_multipath_config_defaults() -> None:
    """MultipathConfig has sensible physical defaults."""
    cfg = MultipathConfig()
    assert cfg.num_taps >= 1
    assert cfg.max_delay_samples >= 1
    assert cfg.gain_spread_db >= 0.0


def test_multipath_single_tap_is_passthrough() -> None:
    """A single-tap multipath (direct path only) is a near-identity."""
    cfg = AugmentationConfig(
        multipath=MultipathConfig(num_taps=1, max_delay_samples=1, gain_spread_db=0.0)
    )
    rng = np.random.default_rng(0)
    iq = _make_cw(256)
    out = apply_augmentations(iq, cfg, rng, sample_rate_hz=FS)
    np.testing.assert_allclose(np.abs(out), np.abs(iq), rtol=1e-5)


# ---------------------------------------------------------------------------
# AugmentationConfig Pydantic validation
# ---------------------------------------------------------------------------


def test_augmentation_config_accepts_none() -> None:
    """A fully-None AugmentationConfig is valid and constructable."""
    cfg = AugmentationConfig()
    assert cfg.add_awgn is None
    assert cfg.multipath is None


def test_augmentation_config_accepts_valid_ranges() -> None:
    """Valid range tuples are accepted without error."""
    cfg = AugmentationConfig(
        add_awgn=(-10.0, 30.0),
        frequency_shift=(-1e6, 1e6),
        gain_variation=(-10.0, 10.0),
        iq_imbalance=(-15.0, 15.0),
        multipath=MultipathConfig(num_taps=8, max_delay_samples=50, gain_spread_db=12.0),
        impulsive_noise=(0.001, 5.0),
        sample_rate_jitter=(-500.0, 500.0),
    )
    assert cfg.add_awgn == (-10.0, 30.0)

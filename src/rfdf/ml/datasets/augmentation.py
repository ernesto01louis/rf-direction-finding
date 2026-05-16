"""Pure-NumPy IQ augmentation for RF signal classification.

All operations work on complex IQ arrays (``np.ndarray`` of dtype
``complex64`` or ``complex128``) and are intentionally torch-free so this
module can be imported without the ``[ml]`` extra.

Augmentation is applied per-sample during dataset construction. Pass an
:class:`AugmentationConfig` to a dataset loader; it will call
:func:`apply_augmentations` for each item using a seeded RNG derived from
the item index so the augmented dataset is deterministic.

Example::

    import numpy as np
    from rfdf.ml.datasets.augmentation import AugmentationConfig, apply_augmentations

    cfg = AugmentationConfig(add_awgn=(-10.0, 20.0), frequency_shift=(-1e3, 1e3))
    rng = np.random.default_rng(42)
    iq = np.ones(1024, dtype=np.complex64)
    augmented = apply_augmentations(iq, cfg, rng, sample_rate_hz=2e6)
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Type alias used throughout: a 1-D NumPy array of any complex dtype.
IQArray = np.ndarray  # shape (N,), dtype complex64 or complex128


class MultipathConfig(BaseModel):
    """Configuration for a synthetic multipath channel.

    Attributes:
        num_taps: Number of multipath taps (including the direct path at tap 0).
        max_delay_samples: Maximum additional delay spread in samples. Tap
            delays are drawn uniformly from ``[1, max_delay_samples]``.
        gain_spread_db: Standard deviation (dB) of per-tap gain variations
            relative to the direct path.  A value of 0.0 produces equal-gain
            combining.
    """

    model_config = ConfigDict(frozen=True)

    num_taps: int = Field(default=4, ge=1, le=64)
    max_delay_samples: int = Field(default=20, ge=1, le=512)
    gain_spread_db: float = Field(default=6.0, ge=0.0, le=40.0)


class AugmentationConfig(BaseModel):
    """Specification of IQ impairments to apply during dataset loading.

    Each field is either ``None`` (disabled) or a ``(min, max)`` tuple from
    which a value is drawn uniformly at runtime.  All None is a valid "no-op"
    configuration.

    Attributes:
        add_awgn: SNR range in dB. The impairment adds white Gaussian noise
            to achieve the sampled SNR.  Negative SNR means the noise power
            exceeds the signal power.
        frequency_shift: Carrier frequency offset range in Hz.  The sampled
            CFO is applied as a complex exponential ``exp(j*2pi*f*t/fs)``.
        gain_variation: Per-sample amplitude gain range in dB.  Applied as a
            linear multiplier to the whole IQ vector.
        iq_imbalance: Phase imbalance range in degrees.  Rotates the Q
            branch relative to the I branch by the sampled angle.
        multipath: When not ``None``, convolves the IQ with a randomly
            generated finite impulse response using :class:`MultipathConfig`.
        impulsive_noise: ``(probability, amplitude)`` tuple controlling
            random impulse injection.  Each sample is replaced with a
            complex impulse of the given amplitude with the given probability.
        sample_rate_jitter: Sample-rate error range in parts-per-million
            (ppm).  Implemented as a resampling of the IQ vector by
            ``1 + ppm/1e6``, then clipping or zero-padding back to the
            original length.
    """

    model_config = ConfigDict(frozen=True)

    add_awgn: tuple[float, float] | None = None
    frequency_shift: tuple[float, float] | None = None
    gain_variation: tuple[float, float] | None = None
    iq_imbalance: tuple[float, float] | None = None
    multipath: MultipathConfig | None = None
    impulsive_noise: tuple[float, float] | None = None
    sample_rate_jitter: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Private impairment helpers
# ---------------------------------------------------------------------------


def _awgn(
    iq: IQArray,
    snr_db: float,
    rng: np.random.Generator,
) -> IQArray:
    """Add AWGN to achieve the target SNR.

    Args:
        iq: Input IQ array (any complex dtype).
        snr_db: Desired signal-to-noise ratio in dB.
        rng: Seeded NumPy random Generator.

    Returns:
        Noisy IQ array in the same dtype as the input.
    """
    original_dtype = iq.dtype
    iq_f = iq.astype(np.complex128)
    signal_power = np.mean(np.abs(iq_f) ** 2)
    if signal_power == 0.0:
        return iq
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (rng.standard_normal(iq_f.shape) + 1j * rng.standard_normal(iq_f.shape))
    return (iq_f + noise).astype(original_dtype)  # type: ignore[no-any-return]


def _frequency_shift(
    iq: IQArray,
    cfo_hz: float,
    sample_rate_hz: float,
) -> IQArray:
    """Apply a carrier frequency offset.

    Args:
        iq: Input IQ array.
        cfo_hz: Carrier frequency offset in Hz.
        sample_rate_hz: Sample rate of the IQ in Hz.

    Returns:
        Frequency-shifted IQ in the same dtype as the input.
    """
    original_dtype = iq.dtype
    n = np.arange(len(iq), dtype=np.float64)
    shift = np.exp(1j * 2.0 * np.pi * cfo_hz * n / sample_rate_hz)
    return (iq.astype(np.complex128) * shift).astype(original_dtype)  # type: ignore[no-any-return]


def _gain_variation(
    iq: IQArray,
    gain_db: float,
) -> IQArray:
    """Multiply IQ by a linear gain factor.

    Args:
        iq: Input IQ array.
        gain_db: Gain in dB (positive = amplification, negative = attenuation).

    Returns:
        Gain-adjusted IQ in the same dtype as the input.
    """
    linear = 10.0 ** (gain_db / 20.0)
    return (iq * linear).astype(iq.dtype)  # type: ignore[no-any-return]


def _iq_imbalance(
    iq: IQArray,
    phase_deg: float,
) -> IQArray:
    """Rotate the Q branch by a phase imbalance angle.

    Args:
        iq: Input IQ array.
        phase_deg: Phase imbalance in degrees.  Positive rotates Q forward.

    Returns:
        Imbalanced IQ in the same dtype as the input.
    """
    original_dtype = iq.dtype
    phase_rad = np.deg2rad(phase_deg)
    i = np.real(iq).astype(np.float64)
    q = np.imag(iq).astype(np.float64)
    q_shifted = q * np.cos(phase_rad) + i * np.sin(phase_rad)
    return (i + 1j * q_shifted).astype(original_dtype)  # type: ignore[no-any-return]


def _multipath(
    iq: IQArray,
    cfg: MultipathConfig,
    rng: np.random.Generator,
) -> IQArray:
    """Convolve IQ with a random multipath FIR channel.

    The direct path (tap 0) always has unity gain.  Additional taps have
    random gains (log-uniform in the requested spread range) and delays
    uniformly drawn from ``[1, max_delay_samples]``.

    Args:
        iq: Input IQ array.
        cfg: Multipath configuration.
        rng: Seeded NumPy random Generator.

    Returns:
        Channel-distorted IQ, clipped to the original length.
    """
    original_dtype = iq.dtype
    n_taps = cfg.num_taps
    max_delay = cfg.max_delay_samples
    gain_spread_db = cfg.gain_spread_db

    # Build the FIR impulse response on a time axis of length max_delay + 1.
    h = np.zeros(max_delay + 1, dtype=np.complex128)
    h[0] = 1.0  # direct path

    if n_taps > 1:
        delays = rng.integers(1, max_delay + 1, size=n_taps - 1)
        gains_db = rng.uniform(-gain_spread_db, 0.0, size=n_taps - 1)
        gains = 10.0 ** (gains_db / 20.0)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_taps - 1)
        for delay, gain, phase in zip(delays, gains, phases, strict=True):
            h[int(delay)] += gain * np.exp(1j * phase)

    result: np.ndarray = np.convolve(iq.astype(np.complex128), h)
    return result[: len(iq)].astype(original_dtype)


def _impulsive_noise(
    iq: IQArray,
    probability: float,
    amplitude: float,
    rng: np.random.Generator,
) -> IQArray:
    """Inject random complex impulses.

    Args:
        iq: Input IQ array.
        probability: Fraction of samples that receive an impulse in ``[0, 1)``.
        amplitude: Amplitude of each injected impulse.
        rng: Seeded NumPy random Generator.

    Returns:
        IQ with injected impulses in the same dtype as the input.
    """
    original_dtype = iq.dtype
    out = iq.astype(np.complex128).copy()
    mask = rng.random(len(out)) < probability
    if np.any(mask):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(mask)))
        out[mask] = amplitude * np.exp(1j * phases)
    return out.astype(original_dtype)


def _sample_rate_jitter(
    iq: IQArray,
    jitter_ppm: float,
    rng: np.random.Generator,
) -> IQArray:
    """Simulate sample-rate error by resampling and length-matching.

    Args:
        iq: Input IQ array.
        jitter_ppm: Sample-rate error in parts-per-million.  Positive means the
            ADC runs faster than nominal (stretches the signal in time, so we
            drop tail samples).  Negative means it runs slow (the waveform
            is shorter; we zero-pad).
        rng: Seeded NumPy random Generator (unused but kept for API uniformity).

    Returns:
        Resampled IQ, clipped or zero-padded to the original length.
    """
    _ = rng  # intentionally unused; kept for uniform API
    original_dtype = iq.dtype
    n = len(iq)
    if n == 0:
        return iq
    factor = 1.0 + jitter_ppm / 1.0e6
    # Resampled length (round to nearest integer).
    resampled_n = max(1, round(n * factor))
    # Linear interpolation of the real and imaginary parts.
    orig_indices = np.linspace(0.0, n - 1.0, num=n)
    new_indices = np.linspace(0.0, n - 1.0, num=resampled_n)
    i_interp = np.interp(new_indices, orig_indices, np.real(iq.astype(np.complex128)))
    q_interp = np.interp(new_indices, orig_indices, np.imag(iq.astype(np.complex128)))
    resampled = i_interp + 1j * q_interp
    # Clip or zero-pad back to the original length.
    out = np.zeros(n, dtype=np.complex128)
    copy_len = min(n, resampled_n)
    out[:copy_len] = resampled[:copy_len]
    return out.astype(original_dtype)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_augmentations(
    iq: IQArray,
    config: AugmentationConfig,
    rng: np.random.Generator,
    *,
    sample_rate_hz: float,
) -> IQArray:
    """Apply the enabled impairments from *config* to *iq* in a fixed order.

    The order of impairments is: multipath -> gain variation -> IQ imbalance ->
    frequency shift -> AWGN -> impulsive noise -> sample-rate jitter.  This
    matches a physically plausible receive chain (channel distortion first,
    receiver hardware impairments second, noise last).

    Args:
        iq: Complex IQ array of shape ``(N,)`` with dtype ``complex64`` or
            ``complex128``.  1-D only; use ``iq.flatten()`` if needed.
        config: Augmentation configuration.  A fully default
            :class:`AugmentationConfig` (all ``None``) is a no-op and
            returns the input unchanged.
        rng: A seeded :class:`numpy.random.Generator`.  Pass a deterministic
            generator (e.g. ``np.random.default_rng(seed ^ index)``) for
            reproducible augmentation.
        sample_rate_hz: Sample rate of the IQ in Hz.  Required for the
            frequency-shift and sample-rate-jitter impairments.

    Returns:
        Augmented IQ array in the same dtype as the input.

    Raises:
        AugmentationError: If ``iq`` is not 1-D or ``sample_rate_hz <= 0``.
    """
    from rfdf.ml.errors import AugmentationError

    if iq.ndim != 1:
        raise AugmentationError(f"apply_augmentations expects 1-D IQ, got shape {iq.shape}")
    if sample_rate_hz <= 0.0:
        raise AugmentationError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

    out = iq

    # 1. Multipath channel (earliest -- propagation distortion before the receiver).
    if config.multipath is not None:
        out = _multipath(out, config.multipath, rng)

    # 2. Gain variation (front-end AGC error).
    if config.gain_variation is not None:
        lo, hi = config.gain_variation
        gain_db = float(rng.uniform(lo, hi))
        out = _gain_variation(out, gain_db)

    # 3. IQ imbalance (ADC/mixer quadrature error).
    if config.iq_imbalance is not None:
        lo, hi = config.iq_imbalance
        phase_deg = float(rng.uniform(lo, hi))
        out = _iq_imbalance(out, phase_deg)

    # 4. Frequency shift (local oscillator offset / Doppler).
    if config.frequency_shift is not None:
        lo, hi = config.frequency_shift
        cfo_hz = float(rng.uniform(lo, hi))
        out = _frequency_shift(out, cfo_hz, sample_rate_hz)

    # 5. AWGN (thermal noise floor, last wideband impairment).
    if config.add_awgn is not None:
        lo, hi = config.add_awgn
        snr_db = float(rng.uniform(lo, hi))
        out = _awgn(out, snr_db, rng)

    # 6. Impulsive noise (man-made interference / ADC spikes).
    if config.impulsive_noise is not None:
        prob, amp = config.impulsive_noise
        out = _impulsive_noise(out, prob, amp, rng)

    # 7. Sample-rate jitter (crystal oscillator tolerance).
    if config.sample_rate_jitter is not None:
        lo, hi = config.sample_rate_jitter
        jitter_ppm = float(rng.uniform(lo, hi))
        out = _sample_rate_jitter(out, jitter_ppm, rng)

    return out

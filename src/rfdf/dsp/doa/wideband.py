"""Wideband DOA: incoherent MUSIC and the Coherent Signal-Subspace Method."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfdf.dsp.doa._common import peak_pick_1d, signal_noise_subspaces
from rfdf.dsp.doa.music import music
from rfdf.dsp.doa.result import DoaEstimate
from rfdf.dsp.errors import InvalidCovarianceError
from rfdf.dsp.steering import build_manifold, steering_vector

#: Floor applied before a log so an empty bin never produces ``-inf`` dB.
_FLOOR: float = 1e-300


def _subband_covariances(
    iq: ArrayLike, sample_rate_hz: float, center_freq_hz: float, num_bins: int
) -> list[tuple[float, np.ndarray]]:
    """FFT an IQ block and group its bins into sub-bands.

    Returns a ``(centre_frequency_hz, covariance)`` pair per usable sub-band.
    """
    samples = np.asarray(iq, dtype=np.complex128)
    if samples.ndim != 2:
        raise InvalidCovarianceError(f"expected (M, N) IQ, got ndim {samples.ndim}")
    num_channels, num_samples = samples.shape
    spectrum = np.fft.fftshift(np.fft.fft(samples, axis=1), axes=1)
    bin_offsets = np.fft.fftshift(np.fft.fftfreq(num_samples, 1.0 / sample_rate_hz))
    edges = np.linspace(0, num_samples, num_bins + 1, dtype=int)
    subbands: list[tuple[float, np.ndarray]] = []
    for index in range(num_bins):
        low, high = int(edges[index]), int(edges[index + 1])
        if high - low < num_channels:
            continue
        band = spectrum[:, low:high]
        covariance = (band @ band.conj().T) / band.shape[1]
        frequency = center_freq_hz + float(np.mean(bin_offsets[low:high]))
        subbands.append((frequency, covariance))
    return subbands


def incoherent_wideband_music(
    iq: ArrayLike,
    *,
    positions: ArrayLike,
    num_signals: int,
    sample_rate_hz: float,
    center_freq_hz: float,
    az_grid_deg: ArrayLike,
    num_bins: int = 16,
) -> DoaEstimate:
    """Estimate DOA by incoherent wideband MUSIC.

    Splits the band into sub-bands, runs narrowband MUSIC in each with that sub-band's
    own steering manifold, and averages the pseudospectra in the log domain (Wax, Shan
    & Kailath 1984). Simple and robust, works at any geometry; no coherent gain.

    Args:
        iq: Wideband IQ, shape ``(M, N)``.
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        num_signals: Number of sources ``K``.
        sample_rate_hz: Complex sample rate of the IQ.
        center_freq_hz: RF centre frequency of the IQ.
        az_grid_deg: 1-D azimuth grid in degrees.
        num_bins: Number of sub-bands to split the spectrum into.

    Returns:
        A :class:`DoaEstimate` carrying the averaged pseudospectrum and peak bearings.

    Raises:
        InvalidCovarianceError: If the IQ is not 2-D or yields no usable sub-bands.
    """
    subbands = _subband_covariances(iq, sample_rate_hz, center_freq_hz, num_bins)
    if not subbands:
        raise InvalidCovarianceError(
            "no usable sub-bands; lengthen the IQ block or reduce num_bins"
        )
    grid = np.asarray(az_grid_deg, dtype=np.float64)
    accumulated_db = np.zeros(grid.size, dtype=np.float64)
    for frequency, covariance in subbands:
        manifold = build_manifold(positions, grid, np.array([0.0]), frequency)
        _, noise = signal_noise_subspaces(covariance, num_signals)
        projection = manifold.matrix @ noise.conj()
        null = np.real(np.sum(np.abs(projection) ** 2, axis=1))
        accumulated_db += -10.0 * np.log10(np.maximum(null, _FLOOR))
    spectrum = 10.0 ** (accumulated_db / len(subbands) / 10.0)
    indices, azimuths, strengths, spectrum_db = peak_pick_1d(spectrum, grid, num_signals)
    return DoaEstimate(
        algorithm="incoherent_wideband_music",
        num_signals=num_signals,
        azimuth_deg=azimuths,
        elevation_deg=[0.0] * len(azimuths),
        pseudospectrum_db=spectrum_db,
        peak_indices=indices,
        peak_strengths_db=strengths,
    )


def cssm(
    iq: ArrayLike,
    *,
    positions: ArrayLike,
    num_signals: int,
    sample_rate_hz: float,
    center_freq_hz: float,
    az_grid_deg: ArrayLike,
    num_bins: int = 16,
) -> DoaEstimate:
    """Estimate DOA by the Coherent Signal-Subspace Method.

    Bootstraps rough angles with an incoherent pass, builds a unitary focusing matrix
    per sub-band (the Procrustes solution that maps each sub-band's manifold onto the
    centre-frequency reference), forms one focused covariance, and runs a single MUSIC
    (Wang & Kaveh 1985). CSSM combines sub-bands coherently — better resolution than the
    incoherent method, at the cost of needing the bootstrap angles.

    Args:
        iq: Wideband IQ, shape ``(M, N)``.
        positions: Antenna positions, shape ``(M, 3)`` in metres.
        num_signals: Number of sources ``K``.
        sample_rate_hz: Complex sample rate of the IQ.
        center_freq_hz: RF centre frequency (also the focusing frequency).
        az_grid_deg: 1-D azimuth grid in degrees.
        num_bins: Number of sub-bands to split the spectrum into.

    Returns:
        A :class:`DoaEstimate` from MUSIC on the focused covariance.

    Raises:
        InvalidCovarianceError: If the IQ is not 2-D or yields no usable sub-bands.
    """
    bootstrap = incoherent_wideband_music(
        iq,
        positions=positions,
        num_signals=num_signals,
        sample_rate_hz=sample_rate_hz,
        center_freq_hz=center_freq_hz,
        az_grid_deg=az_grid_deg,
        num_bins=num_bins,
    )
    rough_azimuths = bootstrap.azimuth_deg
    subbands = _subband_covariances(iq, sample_rate_hz, center_freq_hz, num_bins)
    pos = np.asarray(positions, dtype=np.float64)
    num_channels = pos.shape[0]
    reference = np.stack(
        [
            steering_vector(pos, az_deg=az, el_deg=0.0, freq_hz=center_freq_hz)
            for az in rough_azimuths
        ],
        axis=1,
    )
    focused = np.zeros((num_channels, num_channels), dtype=np.complex128)
    for frequency, covariance in subbands:
        band = np.stack(
            [
                steering_vector(pos, az_deg=az, el_deg=0.0, freq_hz=frequency)
                for az in rough_azimuths
            ],
            axis=1,
        )
        left, _, right = np.linalg.svd(reference @ band.conj().T)
        transform = left @ right
        focused += transform @ covariance @ transform.conj().T
    focused /= len(subbands)
    manifold = build_manifold(
        pos, np.asarray(az_grid_deg, dtype=np.float64), np.array([0.0]), center_freq_hz
    )
    estimate = music(focused, manifold, num_signals)
    return estimate.model_copy(update={"algorithm": "cssm"})

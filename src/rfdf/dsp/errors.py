"""Exception taxonomy for the DSP layer.

Every DOA / calibration / CRLB failure mode raises a specific :class:`DspError`
subclass with an actionable message, so a caller can distinguish "you passed a
non-ULA geometry to ESPRIT" from "your covariance matrix is the wrong shape"
without parsing strings.
"""

from __future__ import annotations


class DspError(Exception):
    """Base class for every error raised by the ``rfdf.dsp`` package."""


class InvalidCovarianceError(DspError):
    """A covariance matrix is the wrong shape, non-finite, or not Hermitian."""


class InvalidGeometryError(DspError):
    """An antenna-position array is malformed, empty, or degenerate."""


class SourceCountError(DspError):
    """The requested number of signals is invalid for the array or covariance.

    Raised when ``num_signals`` is non-positive, or is too large for the array
    (a rank-``M`` covariance cannot carry ``M`` or more resolvable sources).
    """


class NotULAError(DspError):
    """An algorithm requires a uniform linear array but received another geometry.

    Root-MUSIC, ESPRIT, and forward spatial smoothing all rely on the shift-invariant
    Vandermonde structure of a ULA. They raise this rather than returning silently
    wrong bearings on an arbitrary array.
    """


class CalibrationError(DspError):
    """A calibration could not be produced, applied, loaded, or saved.

    Covers malformed S-parameter matrices, an IQ block whose channel count does not
    match the calibration, a missing calibration file, and a missing ``scikit-rf``
    install when reading a Touchstone file.
    """

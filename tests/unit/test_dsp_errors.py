"""Unit tests for the rfdf.dsp.errors exception taxonomy."""

from __future__ import annotations

import pytest

from rfdf.dsp.errors import (
    DspError,
    InvalidCovarianceError,
    InvalidGeometryError,
    NotULAError,
    SourceCountError,
)


@pytest.mark.parametrize(
    "exc",
    [InvalidCovarianceError, InvalidGeometryError, NotULAError, SourceCountError],
)
def test_dsp_errors_are_dsperror_subclasses(exc: type[Exception]) -> None:
    """Every concrete DSP error derives from the DspError base."""
    assert issubclass(exc, DspError)


def test_dsp_error_is_an_exception() -> None:
    """DspError is a plain Exception subclass, catchable generically."""
    assert issubclass(DspError, Exception)


def test_dsp_error_carries_a_message() -> None:
    """A raised DSP error preserves its message for the caller."""
    with pytest.raises(InvalidCovarianceError, match="bad shape"):
        raise InvalidCovarianceError("bad shape")

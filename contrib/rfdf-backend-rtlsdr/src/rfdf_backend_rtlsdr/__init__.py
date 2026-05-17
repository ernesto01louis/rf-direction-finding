"""RTL-SDR contrib backend for rfdf.

A separate, pip-installable package that registers an ``rtlsdr`` SDR backend
via the ``rfdf.backends.sdr`` entry-point group. It is NOT a dependency of core
rfdf — it is the canonical worked example of "how to write a contrib backend".
"""

from rfdf_backend_rtlsdr.source import RtlSdrSource, create

__version__ = "0.1.0"
__all__ = ["RtlSdrSource", "__version__", "create"]

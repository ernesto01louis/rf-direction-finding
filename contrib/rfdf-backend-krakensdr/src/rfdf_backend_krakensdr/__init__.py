"""KrakenSDR contrib backend for rfdf.

A separate, pip-installable package registering a ``krakensdr`` SDR backend via
the ``rfdf.backends.sdr`` entry-point group. It is NOT a dependency of core
rfdf — it is the coherent-multi-channel counterpart to the RTL-SDR contrib
example.
"""

from rfdf_backend_krakensdr.heimdall import HeimdallInterface, HeimdallShmInterface
from rfdf_backend_krakensdr.source import KrakenSdrSource, create

__version__ = "0.1.0"
__all__ = [
    "HeimdallInterface",
    "HeimdallShmInterface",
    "KrakenSdrSource",
    "__version__",
    "create",
]

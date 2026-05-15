"""SDR backends implementing :class:`rfdf.hal.SdrSource`.

Stage 2 ships:

* :mod:`rfdf.backends.sdr.mock` — synthetic emitter scenarios using a
  physically-faithful array-factor signal model.
* :mod:`rfdf.backends.sdr.file_replay` — SigMF file playback.

Real hardware backends (UHD-based B210, SoapySDR, pyadi-iio) land in Stage 5
under the ``[sdr-uhd]`` / ``[sdr-soapy]`` / ``[sdr-pluto]`` optional extras.
"""

# rfdf-backend-rtlsdr

An **RTL-SDR** `SdrSource` backend for [rfdf](https://github.com/ernesto01louis/rf-direction-finding).

This is a **contrib backend** — a separate, pip-installable package that is *not*
a dependency of core rfdf. It is the canonical worked example of "how to write a
contrib backend": copy this directory, swap `pyrtlsdr` for your SDR's driver,
and you have a new backend in a weekend.

## What it is

A single-channel RTL2832U + R820T2 dongle (24 MHz – 1.766 GHz). Cheap,
ubiquitous, perfect for demos and learning. Not phase-coherent — `supports_coherent`
is `False`.

## Install

```sh
pip install -e contrib/rfdf-backend-rtlsdr/      # from an rfdf checkout
# or, once published:  pip install rfdf-backend-rtlsdr
```

You also need the system `librtlsdr` driver (`apt install rtl-sdr`, `brew install
librtlsdr`, …) and a udev rule so a non-root process can open the dongle —
`rfdf hw udev install` ships one.

## Use

Installing the package registers an `rtlsdr` entry point under the
`rfdf.backends.sdr` group. Core rfdf discovers it with **no code change**:

```sh
rfdf hw list-backends     # `rtlsdr` now appears alongside `mock` / `file-replay`
```

```python
from rfdf.hal import load_backend, SdrConfig

sdr = load_backend("rfdf.backends.sdr", "rtlsdr", device_index=0)
await sdr.configure(SdrConfig(center_freq_hz=100e6, sample_rate_hz=2.4e6))
await sdr.start()
async for block in sdr.stream():
    ...  # block.iq is (1, N) complex64
```

## How a contrib backend works

1. **Own package, own `pyproject.toml`.** It depends on `rfdf` (for the HAL
   types) and its device driver — never on rfdf internals.
2. **Implement the `SdrSource` Protocol.** See `src/rfdf_backend_rtlsdr/source.py`
   — `configure` / `start` / `stop` / `stream` / `capture` / `status` /
   `calibration_pilot` / `close` plus the capability properties.
3. **Register an entry point.** The `[project.entry-points."rfdf.backends.sdr"]`
   table in `pyproject.toml` is the entire integration surface.
4. **Lazy-import the driver** so the module imports (and the backend is
   discoverable) even when the driver is absent.

That is the whole contract. See `docs/hardware/contrib-backends.md` in the main
repo for the full guide.

## License

Apache-2.0.

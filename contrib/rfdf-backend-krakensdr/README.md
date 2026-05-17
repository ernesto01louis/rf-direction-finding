# rfdf-backend-krakensdr

A **KrakenSDR** coherent `SdrSource` backend for
[rfdf](https://github.com/ernesto01louis/rf-direction-finding).

This is a **contrib backend** — a separate, pip-installable package, *not* a
dependency of core rfdf. It is the coherent-multi-channel counterpart to the
RTL-SDR contrib example.

## What it is

The KrakenSDR is five phase-coherent RTL-SDR receivers with a built-in
noise-source calibration network — coherent up to ~1.8 GHz. It is the cheapest
real coherent array for direction finding.

## Heimdall DAQ — a hard prerequisite

Coherent capture across five tuners is **not** something this backend
re-implements. It is the job of the **Heimdall DAQ daemon**
(`heimdall_daq_fw`) — the same daemon `krakensdr_doa` uses upstream. Heimdall
disciplines the tuners, runs the noise-source calibration, and publishes
aligned IQ frames over a shared-memory ring.

This backend talks to a **running Heimdall instance** — option (a) of the
Stage-5 design — rather than reinventing what already works. You must install
and run Heimdall separately:

```
KrakenSDR ──USB──▶ Heimdall DAQ daemon ──shared memory──▶ rfdf-backend-krakensdr
```

## Install

```sh
pip install -e contrib/rfdf-backend-krakensdr/      # from an rfdf checkout
# or, once published:  pip install rfdf-backend-krakensdr
```

Then install Heimdall (`heimdall_daq_fw`) per the KrakenSDR project docs and
start it against your unit.

## Use

```sh
rfdf hw list-backends     # `krakensdr` appears once the package is installed
```

```python
from rfdf.hal import load_backend, SdrConfig

sdr = load_backend("rfdf.backends.sdr", "krakensdr")
await sdr.configure(SdrConfig(center_freq_hz=433e6, sample_rate_hz=2.4e6))
await sdr.start()
async for block in sdr.stream():
    ...  # block.iq is (5, N) complex64 — phase-coherent
```

## Architecture — the Heimdall seam

`heimdall.py` defines a `HeimdallInterface` Protocol — the seam between this
backend and the daemon:

- `HeimdallShmInterface` — the real adapter; attaches to Heimdall's
  shared-memory ring. Requires a live Heimdall daemon (verified on a hardware
  runner).
- Tests inject an in-memory fake, so the full backend — `configure` / `stream`
  / `capture` / `status` — is exercised without a KrakenSDR or a daemon.

This is the recommended pattern for any backend that fronts an external
service: depend on a Protocol, ship a real adapter + a test fake.

## License

Apache-2.0.

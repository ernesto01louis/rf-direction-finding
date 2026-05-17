# Contrib backends

A **contrib backend** is a backend that lives in `contrib/` as its own
pip-installable package. It is *not* a dependency of core rfdf — it is installed
separately and discovered through the entry-point system. Contrib backends are
how the platform stays hardware-agnostic: anyone with a HackRF, LimeSDR, Pluto,
or any future SDR can add support in a weekend by copying a template.

## Shipped examples

| Package | Device | Notes |
|---|---|---|
| [`contrib/rfdf-backend-rtlsdr/`](../../contrib/rfdf-backend-rtlsdr/) | RTL-SDR (RTL2832U + R820T2) | Single-channel, 24 MHz–1.766 GHz. The simplest worked example. |
| [`contrib/rfdf-backend-krakensdr/`](../../contrib/rfdf-backend-krakensdr/) | KrakenSDR | 5 coherent channels via the Heimdall DAQ daemon. |

Both are **canonical examples** for community contributors — the RTL-SDR package
is the one to copy.

## Installing a contrib backend

Each contrib package is independent. It can be:

```sh
# Installed editable from an rfdf checkout:
pip install -e contrib/rfdf-backend-rtlsdr/

# Published to PyPI and installed by name:
pip install rfdf-backend-rtlsdr

# Forked and maintained externally — it only depends on the rfdf HAL contract.
```

Once installed, `rfdf hw list-backends` discovers it automatically — **core
rfdf needs no change**.

## How to write one

1. **Create the package layout** — copy `contrib/rfdf-backend-rtlsdr/`:
   ```
   contrib/rfdf-backend-<name>/
   ├── pyproject.toml          # own package; depends on rfdf + the device driver
   ├── README.md
   ├── src/rfdf_backend_<name>/
   │   ├── __init__.py
   │   └── source.py           # the backend implementation
   └── tests/
   ```
2. **Implement the HAL Protocol.** An SDR backend implements `SdrSource`
   (`rfdf.hal.SdrSource`) — `configure` / `start` / `stop` / `stream` /
   `capture` / `status` / `calibration_pilot` / `close` plus the capability
   properties. A rotator implements `RotatorController`; a geometry implements
   `GeometryController`.
3. **Register an entry point** — this is the entire integration surface:
   ```toml
   [project.entry-points."rfdf.backends.sdr"]
   <name> = "rfdf_backend_<name>.source:create"
   ```
4. **Lazy-import the device driver** inside the factory / methods, never at
   module load — the module must import (and the backend must be discoverable)
   even when the driver is absent. Raise a clear "install X" error on first use.
5. **Test without hardware** — exercise the capability properties, the Protocol
   conformance (`isinstance(backend, SdrSource)`), config validation, and the
   driver-absent path. Real-device behaviour goes behind a `hardware` marker.

That is the whole contract. The RTL-SDR package is ~200 lines — most of any new
backend is just translating the device's API to the eight Protocol methods.

## Not in the core CI gate

Contrib packages have their own `tests/` and are tested by installing the
package and running its `pytest`. They are deliberately outside the core
`rfdf` test suite — a broken contrib backend must never block a core release.

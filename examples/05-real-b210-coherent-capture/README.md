# 05 — Real B210 coherent capture + DOA

**Hardware required.** Unlike examples 01–04 this demo needs physical hardware
and is *not* part of the demo-no-hardware CI gate.

## What you need

- One or more Ettus USRP B210s. Put their serials in `RFDF_B210_SERIALS`
  (comma-separated).
- An OctoClock-G (or equivalent) distributing 10 MHz + 1 PPS, if you run more
  than one B210 — see [`docs/hardware/sdr-b210.md`](../../docs/hardware/sdr-b210.md).
- The `[sdr-uhd]` extra plus the system UHD driver: `pip install rfdf[sdr-uhd]`.

## Run

```sh
RFDF_B210_SERIALS=31D5A3B,31D5A40 python examples/05-real-b210-coherent-capture/demo.py
```

With no `RFDF_B210_SERIALS` set the script prints its hardware banner and exits
0 — so it is safe to invoke anywhere.

## What it does

1. Configures the B210 backend with the configured units (mandatory pilot-tone
   recalibration runs automatically).
2. Verifies reference-clock + time lock on every board — a lock failure is
   fatal, never a silent fall-back to non-coherent capture.
3. Tunes to 868 MHz and captures 5 s of coherent IQ to a SigMF pair.
4. Runs a MUSIC DOA estimate on a real ambient signal — **receive only, no
   transmit**. Point it at a LoRa gateway, a weather-satellite downlink, or any
   steady in-band emitter.
5. Reports the recovered bearing.

To compare phase coherence with and without the OctoClock external reference,
run it once with `clock_source="external"` and once with `"internal"` and watch
the pilot-tone phase repeatability in `rfdf hw selftest`.

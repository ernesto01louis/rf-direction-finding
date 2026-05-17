# B210 SDR backend — setup & operation

The `b210` backend (`rfdf.backends.sdr.b210`) drives one or more Ettus USRP
B210 software-defined radios over UHD for **multi-device coherent capture**. It
implements the Stage-2 `SdrSource` HAL contract — the same contract the `mock`
and `file-replay` backends pass.

> **Install:** `pip install rfdf[sdr-uhd]` plus the system UHD driver
> (`uhd-host` / `libuhd`). `import rfdf` never imports `uhd`; the SDK is loaded
> lazily the first time a `B210Source` opens a device session.

## Why coherence needs calibration

Each B210 carries one AD9361 transceiver with **two RX channels behind a shared
LO**. The AD9361 PLL is fractional-N: **every retune randomizes the per-channel
phase**. Two channels on one board, or channels across several boards, are
therefore *not* phase-aligned after a tune — even with a shared 10 MHz / 1 PPS
reference. Coherent DOA on uncalibrated channels produces confident, wrong
bearings.

The fix is **pilot-tone calibration after every retune**. `B210Source.configure()`
always ends with a pilot recalibration unless you pass `recalibrate=False`. Do
not disable it for a coherent capture.

## Multi-device coherent clocking

To extend beyond two coherent channels, share a reference across boards:

```
  OctoClock-G
   ├── 10 MHz ──┬── B210 #1  REF IN
   │            └── B210 #2  REF IN
   └── 1 PPS  ──┬── B210 #1  PPS IN
                └── B210 #2  PPS IN
```

- Use **equal-length** reference cables to every board.
- Configure the backend with `clock_source="external"`, `time_source="external"`
  (or `"gpsdo"` if the OctoClock has a GPSDO and you need absolute time).
- The backend verifies `ref_locked` (and `gps_locked` for GPSDO) on every board
  and **fails fatally** if any board does not lock — it never silently falls
  back to non-coherent capture.
- t = 0 is latched on a common 1 PPS edge across every board; tuning is issued
  as a UHD timed command so all boards retune on the same sample.

## USB topology — the #1 instability cause

Each B210 should sit on its **own USB-3.0 root controller**. Two B210s sharing
one controller is the most common "works, but coherence is unstable" failure.
With `usb_topology_check=True` (default) the backend parses `lsusb -t` at
open-time and logs a **critical** warning if the controllers are insufficient.
Check yourself:

```sh
lsusb -t        # each B210 should appear under a different 5000M/10000M bus
```

Add a discrete PCIe USB-3 card if your host has only one root controller.

## Pilot-tone source

The B210 backend is **RX-only** in this stage — it does not transmit the pilot.
The pilot tone is a known CW signal radiated by separate hardware near the
array. Two documented options:

1. **DDS module (development).** A cheap AD9851 DDS board driven by an Arduino
   on a fixed frequency, radiated from a small antenna a short distance from the
   array. Simple, cheap, good enough to develop against.
2. **Coupled B210 TX path (production).** Drive the pilot from a B210's own TX
   channel through a calibration coupler, so the pilot phase is itself locked to
   the shared reference. More work, better long-term stability.

Either way, the pilot must be on the configured centre frequency before
`configure()` runs. Calibration estimates per-channel gain + phase corrections
(channel 0 is the reference) and stores them as the active `Calibration`.
Failure (no pilot detected, a dead channel) raises `B210CalibrationError` —
calibration never silently succeeds with bad data.

## Data-rate envelope

| Limit | Value |
|---|---|
| Sustained per-channel rate | 25 MS/s |
| Aggregate USB throughput | 240 MB/s (e.g. 6 ch × 10 MS/s × 4 B `sc16`) |

A request that exceeds either limit raises `B210RateError` with a suggestion to
cut the channel count or the sample rate.

## Configuration

Site-specific serials are **never committed**. Put them in
`~/.config/rfdf/config.toml`:

```toml
[sdr]
backend = "b210"
serial_numbers = ["31D5A3B", "31D5A40"]
clock_source = "external"
time_source = "external"
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `B210LockError` on `ref_locked` | OctoClock 10 MHz not wired / not powered; loose SMA |
| `B210LockError` on `gps_locked` | GPSDO has no sky view or has not warmed up |
| Critical USB-topology warning | two B210s on one root controller — add a PCIe USB-3 card |
| `B210RateError` | requested rate/channels exceed the envelope — reduce them |
| `B210CalibrationError` "no pilot energy" | pilot source off, mistuned, or an antenna disconnected |
| Bearings wander between captures | recalibration disabled, or pilot not stable |

See also [`docs/troubleshooting.md`](../troubleshooting.md) and the
`rfdf hw selftest` command, which runs the HAL contract plus B210-specific smoke
checks (UHD probe, clock/time lock, coherent-retune phase noise) against the
attached hardware.

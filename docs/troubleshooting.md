# Hardware troubleshooting

Common hardware issues and how to diagnose them. Start with `rfdf hw selftest`
— it runs the HAL contract plus device-specific smoke checks against whatever
is configured and prints a colour-coded report.

## "My SDR doesn't show up"

The #1 issue. A fresh USB SDR is root-owned; a non-root process cannot open it.

```sh
rfdf hw udev install      # installs /etc/udev/rules.d/70-rfdf.rules (needs sudo)
```

Then re-plug the device. Confirm with `lsusb` that the IDs match a rule in
`rfdf hw udev list`. See [`udev-rules.md`](hardware/udev-rules.md).

## B210

| Symptom | Diagnosis |
|---|---|
| `B210LockError` (`ref_locked`) | OctoClock 10 MHz not wired/powered; loose SMA. Fatal — coherent operation is impossible. |
| `B210LockError` (`gps_locked`) | GPSDO has no sky view or has not warmed up. |
| Critical USB-topology warning | Two B210s share one USB-3 root controller — the #1 "works but unstable" cause. Add a PCIe USB-3 card. |
| `B210RateError` | The requested rate/channel count exceeds the 25 MS/s per-channel or 240 MB/s aggregate envelope. |
| `B210CalibrationError` "no pilot energy" | Pilot source off, mistuned, or an antenna disconnected. |
| Bearings drift between captures | Pilot recalibration disabled, or the pilot source is not stable. Coherent DOA needs pilot-tone calibration after *every* retune. |

Raw-UHD sanity check before suspecting rfdf: `uhd_usrp_probe`, then the bundled
`rx_multi_samples` example. See [`sdr-b210.md`](hardware/sdr-b210.md).

## AntRunner rotator

| Symptom | Diagnosis |
|---|---|
| `AntRunnerError` "Alarm state" | Hit a limit switch / lost position — re-home with `rfdf hw rotator home`. |
| `RotatorNotHomedError` | Home before the first move. |
| `RotatorPositionError` | Lost steps, mechanical slip, or an encoder fault. |
| `GrblConnectionError` | Wrong host/port, controller off, or a firmware-specific `command_path`. |
| Slew never settles | Feed rate too low, or a binding mechanical axis. |

See [`rotator-antrunner.md`](hardware/rotator-antrunner.md).

## GRBL linear rails

| Symptom | Diagnosis |
|---|---|
| `ValueError` "off rail axis" | The requested position is not on a rail — the array can only place antennas where a rail carries them. |
| Position deviation grows over a run | Thermal drift in long lead screws. |
| Random per-move scatter | Loose lead-screw nuts; backlash on a non-closed-loop axis. |
| One rail consistently off | Mis-set steps/mm, or a slipping coupler. |
| Repeatability budget exceeded | Run `measure_position_repeatability` — the report names the problem rails. |

See [`geometry-grbl-rails.md`](hardware/geometry-grbl-rails.md).

## RTL-SDR / KrakenSDR (contrib)

| Symptom | Diagnosis |
|---|---|
| `RtlSdrNotInstalledError` | Install `pyrtlsdr` + the system `librtlsdr`. |
| RTL-SDR not openable | udev rule missing — `rfdf hw udev install`. |
| `HeimdallError` "control path not found" | The Heimdall DAQ daemon is not running — start `heimdall_daq_fw` against the KrakenSDR first. |

## Nothing works and selftest is clean

If `rfdf hw selftest` is green but a campaign still fails, the issue is above
the HAL — check the DOA configuration, the geometry preset, and the calibration
that is active for the current frequency.

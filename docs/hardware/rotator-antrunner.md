# AntRunner rotator backend — setup & operation

The `antrunner` backend (`rfdf.backends.rotator.antrunner`) drives a wuxx/AntRunner
AZ/EL rotator. The AntRunner runs the GRBL_ESP32 firmware and exposes a CNC
G-code dialect over HTTP; this backend wraps that in the Stage-2
`RotatorController` HAL contract — the same contract the `mock` rotator passes.

> **Install:** `pip install rfdf[rotator-antrunner]` (adds the `httpx` HTTP
> client). `import rfdf` never imports `httpx`; it loads lazily on first use.

## Axis mapping

| HAL axis | GRBL axis |
|---|---|
| Azimuth (degrees) | X |
| Elevation (degrees) | Y |

Moves are absolute: `goto(az, el)` sends `G90 G0 X<az> Y<el>`.

## GRBL command set

| Purpose | Command |
|---|---|
| Status query | `?` → `<Idle\|MPos:az,el,…>` |
| Absolute move | `G90 G0 X<az> Y<el>` |
| Home | `$H` |
| Soft-limit set | `$130=<az_max>`, `$131=<el_max>`, `$20=1` (enable) |
| Feed-hold | `!` |
| Resume | `~` |
| Soft reset | `0x18` (Ctrl-X) |
| Settings dump | `$$` |

The backend uses the **HTTP one-shot** command API (simple, stateless). The
firmware also exposes a WebSocket for high-rate position streaming — not used
here.

## Homing & park

- `calibrate()` runs the GRBL homing cycle (`$H`) — both axes drive to their
  limit switches and the encoders zero — then writes the cable-management soft
  limits to `$130`/`$131` and enables soft limits (`$20=1`).
- With `homing_required_on_startup=True` (default), `goto` raises
  `RotatorNotHomedError` until `calibrate()` has run.
- `goto(0, 0)` after homing brings the array to the documented **home**
  position (north, horizontal).
- `park()` commands the safe storage position — elevation 90° (straight up),
  azimuth 0°.
- `stop()` issues a GRBL feed-hold (`!`) then a soft reset.

## Closed-loop validation

The user's closed-loop stepper drivers expose encoder readback, so the GRBL
`MPos` field reflects the *actual* shaft position. With `encoder_validation=True`
(default) the backend reads `MPos` back after every `goto` and raises
`RotatorPositionError` if it disagrees with the commanded position by more than
`positioning_accuracy_deg` — a mechanical slip, a lost step, or an encoder
fault is caught immediately rather than silently corrupting a DOA sweep.

## Cable management

Coax and control cables run up through the rotator. Unbounded azimuth rotation
wraps and eventually severs them. The soft limits are deliberately conservative:

```python
soft_limits_az = (-180.0, 180.0)   # +/- one cable turn from home
soft_limits_el = (0.0, 180.0)
```

`goto` rejects an out-of-range target in software *before* any motion; homing
additionally pushes the limits into the firmware (`$130`/`$131`, `$20=1`) so the
controller itself refuses a cable-wrapping move.

**Rotation tracking.** Azimuth is mechanically continuous but logically bounded
to ±180° from home. The backend does not track multi-turn windup — operate
within ±180°. If a cable looks twisted, drive to azimuth 0° and visually confirm
the cables hang straight; that is the "untwisted" reference. After any manual
re-cabling, re-run `calibrate()`.

## Hamlib / Gpredict

`rfdf hw rotator-server` exposes an optional `rotctld` TCP server (port 4533 by
default) so Gpredict and other amateur-satellite trackers can point the rotator
through the standard Hamlib network protocol.

## Configuration

The controller host is **never committed**. Put it in `~/.config/rfdf/config.toml`:

```toml
[rotator]
backend = "antrunner"
host = "rotator.local"
port = 80
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `AntRunnerError` "Alarm state" | hit a limit switch / lost position — re-home with `calibrate()` |
| `RotatorNotHomedError` | call `calibrate()` before the first `goto` |
| `RotatorPositionError` | lost steps, mechanical slip, or an encoder fault |
| `GrblConnectionError` | wrong host/port, controller off, or a firmware-specific `command_path` |
| Slew never settles | feed rate too low, or a binding mechanical axis |

If your GRBL_ESP32 fork uses a different command endpoint, pass `command_path`
to `create()` (the `{cmd}` placeholder receives the URL-encoded command).

# GRBL linear-rail geometry backend — design & commissioning

The `grbl-linear` backend (`rfdf.backends.geometry.grbl_linear`) is the morphing
array made real: each antenna rides a motorized linear rail driven by a
stepper + closed-loop driver behind a GRBL_ESP32 controller. It implements the
Stage-2 `GeometryController` HAL contract — the same contract the `static` and
`mock-morph` geometries pass.

> **Install:** `pip install rfdf[geometry-grbl]` (adds the `httpx` HTTP client,
> shared with the AntRunner backend). `import rfdf` never imports `httpx`.

## Rail model

Each rail is fixed at construction by a `RailConfig`:

| Field | Meaning |
|---|---|
| `antenna_id` | Antenna index (the set must form `0..N-1`) |
| `origin` | Rail-start `(x, y, z)` in metres — offset 0 |
| `direction` | Unit vector along the rail (normalised on validation) |
| `travel_m` | Maximum travel from `origin` along `direction` |
| `grbl_axis` | GRBL axis letter driving the rail (`X`/`Y`/`Z`/`A`/`B`) |

Rails may sit at **arbitrary angles** in 3D space — they need not be coplanar.
For the reference 5-LPDA cluster, five rails radiate outward from the rotator
base at different angles and elevations, each with ~1 m of travel.

`goto_positions()` projects each requested 3-D antenna position onto its rail
axis: `offset = (target − origin) · direction`. A position that does **not** lie
on its rail axis (residual > 0.1 mm), or whose offset is outside `[0, travel_m]`,
is rejected with a clear `ValueError` — the backend can only place an antenna
where its rail can physically carry it.

## Presets

Named array configurations live in `~/.config/rfdf/geometry-presets.toml`:

```toml
[preset.uhf_compact]
positions = [
    [0.0, 0.0, 0.0],
    [0.17, 0.0, 0.0],
    [0.0, 0.17, 0.0],
    [-0.17, 0.0, 0.0],
    [0.0, -0.17, 0.0],
]
```

`save_preset()` validates every position is reachable before persisting it.
CLI: `rfdf hw geometry list-presets`, `rfdf hw geometry goto <preset>`.

## Position-error budget

`measure_position_repeatability(geometry, iterations=50)` is a commissioning
routine (not run in CI): it commands every preset `iterations` times, reads the
actual position back, and reports the **max + RMS deviation** from commanded.
The Stage-5 acceptance budget is **max deviation < 1 mm** — sufficient for
synthetic aperture at 5.8 GHz. The report is saved as a JSON artifact next to
the geometry preset.

> The Stage-5 PDF names this routine `test_position_repeatability`; it ships as
> `measure_position_repeatability` so pytest never collects it as a test.

## Linear-rail commissioning checklist

Run once after the mechanical build is complete:

1. **Home** every axis (`calibrate()` → `$H`) — establishes machine zero.
2. **Set + enable soft limits** to the measured mechanical travel.
3. **Set steps/mm** (`$100…`) per the lead-screw pitch + microstepping; verify a
   commanded 100 mm move measures 100 mm with calipers.
4. **Command 10 round-trips** per preset; record encoder readback vs commanded.
5. Run `measure_position_repeatability(geometry, iterations=50)`; require
   `within_budget` (max < 1 mm).
6. **Persist** the JSON report + any per-rail offset calibration next to the
   preset.

## Mechanical specs (operator-built)

The mechanical build is the operator's responsibility — the backend assumes the
rails exist and are configured. Recommended specs:

- **Linear rails** — ≥ 1 m travel, supported (not unsupported round rail) to
  keep antenna-tip sag below the budget.
- **Lead screws** — anti-backlash nuts; ball screws if the budget is tight.
- **Steppers + closed-loop drivers** — closed-loop is strongly recommended; an
  open-loop system that loses a step silently corrupts the array geometry.
- **Cable carriers** — drag chains sized for the coax bundle; anchor to the
  rotator base so rotation does not load the rails.

## Common position-error causes

| Symptom | Likely cause |
|---|---|
| Deviation grows over a long run | thermal drift in long lead screws |
| Random per-move scatter | loose lead-screw nuts; backlash on a non-closed-loop axis |
| One rail consistently off | mis-set steps/mm, or a slipping coupler |
| Sudden large jump | a lost step (open-loop) or a homing-switch bounce |

## Configuration

The controller host is **never committed**. Site config in
`~/.config/rfdf/config.toml`; rails are passed to the backend factory from the
operator's measured build.

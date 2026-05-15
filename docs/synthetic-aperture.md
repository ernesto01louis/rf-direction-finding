# Position-domain synthetic aperture

A small array on a motorised rail can visit several positions and have its captures
fused into one estimate over a much larger *virtual* aperture. This is the platform's
differentiator: most hobby DF setups cannot do it, and the commercial products that can
are five-figure instruments.

## The idea

`N` rail stations, each an `M`-element array, give an `M·N`-element virtual array. Since
angular resolution is set by aperture span — beamwidth ≈ `lambda / D` — spreading the
stations over a long rail buys resolution proportional to the rail length, not just to
the element count.

`rfdf.dsp.doa.synthetic_aperture` provides:

- `StationCapture` — one station's antenna positions, IQ block, and pilot-tone phase.
- `synthetic_aperture_doa(captures, *, calibration, freq_hz, az_grid_deg, algorithm,
  fusion, num_signals)` — fuses the captures and returns a `DoaEstimate`.

## Worked example

Five antennas per station, six stations, 5.8 GHz (`lambda ≈ 5.17 cm`):

- A single station's 5-element cross spans `D ≈ 0.34 m` → beamwidth ≈ `0.0517 / 0.34`
  ≈ **8.7 degrees**.
- Six stations evenly spread along a **1.5 m rail** give a virtual aperture
  `D ≈ 1.5 + 0.34 = 1.84 m` → beamwidth ≈ `0.0517 / 1.84` ≈ **1.6 degrees**.

That is a **~5.4x resolution gain**, set by the rail span — the 30 virtual elements
also improve SNR and sidelobes, but the resolution comes from the aperture.

| Rail span | Virtual aperture | Beamwidth | Gain |
|---|---|---|---|
| 1.0 m | 1.34 m | 2.2 deg | 3.9x |
| 1.5 m | 1.84 m | 1.6 deg | 5.4x |
| 2.0 m | 2.34 m | 1.3 deg | 6.9x |

## Fusion modes

- **`coherent`** — each station's IQ is phase-corrected with its pilot reference and
  stacked into one `MN`-channel block; one combined covariance, one estimate. Delivers
  the full aperture gain. Requires source coherence across the inter-station interval
  (CW pilots, locked carriers, GPS C/A after despread).
- **`incoherent`** — a per-station estimate, pseudospectra averaged in the log domain.
  Robust to bursty signals; no coherent aperture gain.
- **`block-diagonal`** — the per-station covariances are assembled into one
  block-diagonal virtual covariance and a single MUSIC is run over the virtual array.

**The coherent-fusion gotcha.** The cross-station phase correction must be applied
*before* the covariance is formed. The rail moves the array between captures and each
capture has its own SDR retune phase; stack the IQ without de-rotating each station by
its pilot phase first and the combined covariance is incoherent garbage. `coherent`
fusion does this de-rotation; the `pilot_phase_rad` on each `StationCapture` (or the
`pilot_phase_corrections` override) supplies the reference.

The `rfdf doa morph-capture` CLI command writes a per-station capture set ready for
fusion.

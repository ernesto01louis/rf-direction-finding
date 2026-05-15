# Calibration

`rfdf.dsp.calibration` corrects the per-channel gain/phase errors and mutual coupling
that turn an ideal array manifold into the messy thing real hardware presents to a DOA
estimator.

## The `Calibration` object

A `Calibration` bundles, for one frequency:

- `channel_gains` — `(M,)` complex per-channel correction multipliers;
- `coupling` — the `(M, M)` complex mutual-coupling matrix `C`;
- `provenance` — a `CalibrationProvenance` recording the procedure, the SDR backend, a
  geometry hash, an ISO-8601 timestamp, and an operator string.

`Calibration.apply(iq)` inverts the forward array model `x = gain ⊙ (C s_clean)`:

```
s_estimate = C⁻¹ (x ⊙ channel_gains)
```

`Calibration.matches_geometry(positions)` checks the stored `geometry_hash` against a
geometry — a guard against applying a calibration to the wrong array.

## Producing a calibration

| Procedure | What it does |
|---|---|
| `load_simulated(positions, freq_hz)` | An ideal identity calibration for mock development. |
| `calibrate_pilot_tone(sdr, ...)` | Estimates per-channel gain and phase from a received CW pilot. Async — it streams from the SDR. Assumes a planar array (a zenith pilot then has a flat wavefront, so any per-channel difference is a channel error). |
| `calibrate_mutual_coupling(s_parameters, ...)` | Builds `C = I + (S − diag S)` from a measured array S-matrix — unit diagonal plus the off-diagonal port coupling. |

`load_s_parameters(touchstone_path, freq_hz)` reads an S-matrix from a Touchstone file.
It imports `scikit-rf` lazily, so the base `import rfdf` stays free of domain
dependencies — install the `antenna` extra (`pip install rfdf[antenna]`) to use it.

## Persistence

`Calibration.save(name, directory=...)` writes two files: a NumPy `.npz` holding the
matrices and a TOML sidecar holding the metadata. `Calibration.load(name, directory=...)`
reads them back. The default directory is the `platformdirs` user-data path under
`rfdf/calibrations/`.

```python
from rfdf.dsp.calibration import calibrate_pilot_tone

calibration = await calibrate_pilot_tone(sdr, freq_hz=5.8e9, duration_s=0.05)
calibration.save("cband-array")
```

The `rfdf doa calibrate` CLI command runs the pilot-tone procedure against a mock SDR
and saves the result.

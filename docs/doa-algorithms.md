# DOA algorithms

The `rfdf.dsp` package implements the classical direction-of-arrival estimators. All of
it is pure NumPy/SciPy and synchronous; the I/O layer (capturing IQ) is the HAL.

## The steering convention

Every estimator builds its array manifold through one function — `rfdf.dsp.steering` —
so the phase convention is defined exactly once and matches the mock SDR's signal model:

```
a(theta) = exp(-j * (2*pi / lambda) * (P . theta_hat))
theta_hat = [cos(el) cos(az), cos(el) sin(az), sin(el)]
```

`P` is the `(M, 3)` antenna-position matrix in metres. The `exp(-j)` sign is the
standard far-field plane-wave convention. `build_manifold` precomputes the manifold over
an azimuth/elevation grid; evaluation is fully vectorised (a 360x90 grid for five
elements is a few megabytes — broadcasting beats a Python loop).

The sample covariance is `R = (1/N) X Xᴴ` (`rfdf.dsp.covariance.sample_covariance`).
`diagonal_load` regularises a snapshot-starved or rank-deficient covariance.

## Grid estimators

These take a precomputed `SteeringManifold` and scan it. Signature:
`estimator(covariance, steering, num_signals)`.

| Estimator | Pseudospectrum | Reference | Notes |
|---|---|---|---|
| `bartlett` | `P = aᴴ R a / (aᴴ a)` | Bartlett 1948; Van Trees 2002 | Unconditionally stable; aperture-limited resolution. The robustness floor. |
| `mvdr` | `P = 1 / (aᴴ R⁻¹ a)` | Capon 1969 | Better resolution; needs diagonal loading; sensitive to covariance error. |
| `music` | `P = 1 / ‖Uₙᴴ a‖²` | Schmidt 1986 | The workhorse; needs `num_signals` and full-rank (non-coherent) sources. |

`music` eigendecomposes `R` with `numpy.linalg.eigh` (ascending eigenvalues) and takes
the `M - K` smallest-eigenvalue eigenvectors as the noise subspace `Uₙ`.

Peak picking is parabola-refined, so a grid estimate is accurate to a fraction of a grid
cell rather than quantised to it.

## Parametric ULA estimators

These are closed-form and need no grid; they take the geometry directly:
`estimator(covariance, *, positions, freq_hz, num_signals)`. They require a **uniform
linear array** and raise `NotULAError` on any other geometry — the cone ambiguity means
they recover azimuth in `[0, 180]`.

| Estimator | Method | Reference |
|---|---|---|
| `root_music` | Polynomial rooting of the noise-subspace projector | Barabell 1983 |
| `esprit` | Total-least-squares rotational invariance of two ULA subarrays | Roy & Kailath 1989 |
| `unitary_esprit` | ESPRIT with forward-backward averaging (see below) | Haardt & Nossek 1995 |

`unitary_esprit` applies forward-backward averaging then the ESPRIT solve. FB averaging
is the substance of Haardt & Nossek's accuracy gain (it doubles the effective snapshot
count and partially decorrelates coherent sources); the real-valued unitary transform in
the original paper is an arithmetic optimisation that produces the same estimates.

## 2-D MUSIC

`music_2d(covariance, steering, num_signals)` runs the MUSIC null spectrum over a 2-D
`(azimuth, elevation)` grid and peak-picks the surface, returning a `Doa2DResult` with
the full pseudospectrum. It is the workhorse for an array with extent in more than one
axis. Note that the planar reference array has no vertical aperture — an in-plane source
is resolved as a cone angle, not as independent `(az, el)`.

## Wideband DOA

`rfdf.dsp.doa.wideband` covers sources whose bandwidth is not negligible:

- `incoherent_wideband_music` — splits the band into sub-bands, runs MUSIC per sub-band
  with that band's own manifold, and averages the pseudospectra in the log domain
  (Wax, Shan & Kailath 1984). Simple and robust; no coherent gain.
- `cssm` — the Coherent Signal-Subspace Method (Wang & Kaveh 1985). Bootstraps rough
  angles, builds a unitary focusing matrix per sub-band, forms one focused covariance,
  and runs a single MUSIC. Better resolution at the cost of the bootstrap.

## Coherent sources

Two fully-correlated sources (multipath) collapse the signal covariance to rank 1 and
MUSIC fails. `rfdf.dsp.coherent` restores the rank a downstream estimator needs:

- `forward_spatial_smoothing` — averages overlapping ULA subarrays (Shan/Wax/Kailath
  1985). Trades aperture for decorrelation.
- `forward_backward_smoothing` — adds the conjugate-reversed average (Pillai & Kwon
  1989); roughly doubles the decorrelation capacity.
- `toeplitz_rectify` — replaces the covariance with its nearest Hermitian-Toeplitz
  matrix (Williams et al. 1988); keeps the full aperture.

## Number of sources

When the source count is unknown, `rfdf.dsp.model_order` estimates it from the
covariance eigenvalues: `aic`, `mdl` (Wax & Kailath 1985), and `sorte` (He/Wang/Kong
2010). `estimate_num_signals` dispatches by name.

## The `Doa` orchestration class

`rfdf.dsp.doa.Doa` wires an `SdrSource`, a `GeometryController`, an optional
`Calibration`, and an algorithm choice into one `await doa.run()` that captures IQ,
builds the covariance, and returns a `DoaEstimate`. With `num_signals` left unset it
estimates the count via MDL. This is the API the `rfdf doa` CLI uses.

## Choosing an algorithm

- Unknown source count, any geometry, a quick look — **Bartlett** or **MVDR**.
- Known count, non-coherent sources, best 1-D accuracy — **MUSIC** (any geometry) or
  **Root-MUSIC / ESPRIT** (a ULA, closed-form).
- Azimuth *and* elevation — **2-D MUSIC**.
- Coherent (multipath) sources — pre-process with spatial smoothing, then MUSIC.
- A wide signal — **incoherent wideband MUSIC** or **CSSM**.

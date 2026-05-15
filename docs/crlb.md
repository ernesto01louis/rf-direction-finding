# The Cramer-Rao lower bound

The Cramer-Rao lower bound (CRLB) is the smallest variance any unbiased estimator of a
parameter can achieve. For DOA it is the floor on how precisely a bearing can be
estimated from a given array, SNR, and snapshot count. `rfdf.dsp.crlb` computes it, and
the CRLB-bounded tests hold every algorithm to it.

## Why it matters

DOA papers are full of plots captioned "our method approaches the CRLB". The
CRLB-bounded test gate proves the *implementation* actually does: each estimator's
empirical RMSE over a fixed-seed Monte-Carlo sweep must stay within `3 * sqrt(CRLB)`.
This is mathematical verification, not heuristic spot-checking — it is what separates a
research platform from a toy. A DOA function that returns a wrong answer with no test
failure is a silent regression waiting to happen.

## What is computed

The deterministic (conditional) CRB of Stoica & Nehorai 1989:

```
CRB = (sigma^2 / 2N) * { Re[ (Dᴴ Π_A^⊥ D) ⊙ R_s^T ] }⁻¹
```

`D` is the manifold derivative, `Π_A^⊥` the projector onto the noise subspace, `R_s` the
source covariance, `N` the snapshot count, `sigma^2` the noise power. The formula is
geometry-agnostic — it uses analytic steering derivatives, so it covers the ULA and the
planar cross alike.

| Function | Returns |
|---|---|
| `compute_crlb(positions, *, freq_hz, snr_db, snapshots, direction_deg)` | Single-source azimuth variance, degrees². |
| `crlb_azimuth(...)` | Per-source azimuth variance for several sources. |
| `crlb_ula_closed_form(...)` | The textbook closed form for a ULA — an independent cross-check on the geometry-agnostic path. |
| `crlb_joint_azimuth_elevation(...)` | Joint azimuth/elevation variance per source. |

The CRLB falls as `1 / (N * SNR)` and tightens with array size.

## The planar-array elevation caveat

The reference 5-element cross is planar — every element is at `z = 0`, so it has zero
vertical aperture. For a source *in the array plane* the elevation Fisher information is
zero: elevation is un-estimable, and `crlb_joint_azimuth_elevation` reports
`CRLB(elevation) = +inf` rather than a misleading finite number. A ULA cannot resolve
azimuth and elevation jointly at all — the two collapse onto one cone — so both bounds
are infinite there. Azimuth remains finite and well-bounded in both cases.

## Reading a CRLB-bounded test

Such a test fixes a scenario, runs the estimator over many seeded noise realisations,
computes the empirical RMSE, and asserts `RMSE < 3 * sqrt(CRLB)`. The `3x` factor
absorbs finite-sample bias and Monte-Carlo error; it is deliberately generous and can be
tightened as an algorithm proves itself. The seeds are fixed, so the test is
deterministic — no flakiness.

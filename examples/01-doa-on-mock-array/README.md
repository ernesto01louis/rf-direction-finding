# 01 — DOA on a mock array

The canonical "is the platform working?" demo. It needs no hardware: a mock SDR
synthesises the IQ a real array would receive.

```sh
python examples/01-doa-on-mock-array/demo.py
```

[`demo.py`](demo.py) configures an 8-element uniform linear array, places three CW
emitters at known bearings, captures synthetic IQ, and runs two direction-of-arrival
estimators:

- **MUSIC** — the subspace workhorse, over a 0.25-degree azimuth grid.
- **ESPRIT** — the closed-form parametric estimator.

It prints the recovered bearings, the Cramer-Rao lower bound for each source, and
exits with `demo: DOA pipeline PASS` when every estimate lands within 1 degree of
the truth. The whole run takes well under a second.

For the wider DOA surface — calibration, 2-D MUSIC, wideband estimation, coherent-
source smoothing, and the position-domain synthetic aperture — see
[`docs/doa-algorithms.md`](../../docs/doa-algorithms.md) and the `rfdf doa` CLI
(`rfdf doa run`, `benchmark`, `calibrate`, `morph-capture`).

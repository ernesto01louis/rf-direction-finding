# 04 — Rent a GPU and train

A walk through the cost-aware cloud-GPU-rental flow: discover a cloud compute
backend, get a cost estimate, see the confirmation gate, and run the training.

```sh
python examples/04-rent-gpu-and-train/demo.py
```

[`demo.py`](demo.py):

1. Discovers the `runpod` compute backend through the HAL entry-point catalog.
2. Builds a representative `TrainingRecipe` and compiles it to a `ComputeJob`.
3. Prints RunPod's `cost_estimate` for the job — the low / estimated / high USD
   range and its rationale.
4. Shows the cost-confirmation gate: the platform never auto-submits a paid
   cloud job; submission needs explicit operator confirmation.
5. Runs the training on the **`local`** backend as the compare-with-baseline
   step, exactly as a real workflow would do before committing cloud spend.

It exits `0` and prints `demo: GPU-rental walkthrough PASS`.

## What this demo does NOT do

It does **not** submit a real RunPod job. The cost estimate is pure
arithmetic — `cost_estimate` works without the RunPod SDK or any credentials —
so the demo always shows it. But a live submission needs both:

```sh
pip install rfdf[compute-runpod]    # the RunPod SDK
export RUNPOD_API_KEY=...           # your RunPod API key
```

With no credentials configured, this demo runs the **cost-estimate +
local-baseline** path instead. To launch a real cloud job once credentials are
in place:

```sh
rfdf ml train --recipe recipes/sig53-resnet1d-baseline.toml --compute runpod
```

`rfdf ml train` prints the estimate and requires confirmation before
submitting — the audit guardrail against a forgotten cloud instance running up
a bill.

See [`docs/ml/compute-backends.md`](../../docs/ml/compute-backends.md) for every
backend's auth, cost model, and container strategy, and
[`docs/ml/training.md`](../../docs/ml/training.md) for the recipe system.

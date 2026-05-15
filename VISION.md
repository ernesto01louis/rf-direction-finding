# VISION

## 1. Why this exists

The RF research workflow has many excellent tools — GNU Radio for DSP, SDRangel for
exploration, KrakenSDR DOA for low-end direction finding, TorchSig for signal
classification — but no integrating *platform*. There's no shared abstraction for
"the SDR I happen to have", no portable definition of "the array geometry I happen to
be running", no provenance trail tying a captured signal to the model that classified
it to the GPU rental that trained that model.

`rfdf` exists to be that platform:

- A **reproducible campaign abstraction** for capture, calibration, DOA, training, and
  inference.
- **Hardware-agnostic interfaces** that span everything from an RTL-SDR dongle to a
  coherent USRP B210 cluster.
- **Multi-cloud compute** that decouples model training from any specific provider.
- **Citation-grade evidence bundles** when integrated with `ai-orchestrator`, so any
  result the platform produces can be traced back to its inputs, code SHA, dataset
  hash, and (if any) LlmCall records.

## 2. Platform-not-hub principle

`rfdf` is a **platform**, not a hub for specific research projects. The acid test:

> Would a researcher with completely different hardware — a HackRF, a Pluto, a CN0566 —
> also benefit from this code?

If yes, it belongs in `rfdf`. If no, it belongs in a downstream consumer repo or in
`contrib/`.

Domain-flavored work (drone detection, aero RF, antenna manufacturing) lives in its
own repository and imports `rfdf` as a library. The platform itself stays generic.

## 3. Hardware-agnostic principle

No hardware in core. The HAL (`src/rfdf/hal/`) defines four Protocol classes:
`SdrSource`, `RotatorController`, `GeometryController`, `ComputeBackend`. Concrete
backends live in `src/rfdf/backends/` and are loaded via entry-points. The reference
backends (B210, AntRunner, GRBL linear rails) are the *first concrete validation* of
the abstractions, not the abstractions themselves.

`pip install rfdf` must complete with **zero RF or ML dependencies**. Domain-flavored
dependencies (`uhd`, `pyadi-iio`, `torch`, provider SDKs) live in named extras.

## 4. Demo-without-hardware principle

Every algorithm, every pipeline, every tutorial must work end-to-end on canned data
(synthetic emitters + SigMF replay). The CI matrix enforces this via a job named
`test-demo-no-hardware` that runs the full DOA + ML pipeline against simulated input
and asserts results meet CRLB-bounded accuracy.

If a feature breaks the no-hardware demo path, the feature is incorrectly designed.
Fix the design.

## 5. Compute-rented-before-owned principle

`ComputeBackend` is a first-class abstraction. Implementations cover RunPod, Vast.ai,
Modal, SkyPilot, and local. The same training recipe runs everywhere; users pick the
backend in config without code changes. No provider-specific code lives outside
`src/rfdf/backends/compute/`.

## 6. Standalone-first, orchestrator-optional

The orchestrator (`ai-orchestrator`) is a **consumer of `rfdf`, not a dependency**. The
standard install path produces a fully functional platform. Installing the
`[orchestrator]` extra adds optional integration: evidence bundles, Hindsight memory
writes, L5 vault notes, ntfy alerts, planner-dispatched flowgraphs.

A user who never installs the orchestrator gets a complete platform. A user running the
orchestrator gets the full integrated research environment. Both are first-class modes.

## 7. The "validate-before-buy" workflow

Stages 1–4 of the build (scaffold, HAL, DOA, ML) ship with **€0 hardware spend**. Stage
5 (real B210 + AntRunner + linear rails) is the gate: only after the math + ML are
verified on synthetic data is it rational to buy gear. If the CRLB tests fail at Stage
3, no amount of hardware fixes the problem.

## 8. The "rented-before-owned" compute workflow

Same pattern for GPUs. Stage 4 wires up RunPod / Vast.ai / Modal / SkyPilot before
anyone considers buying an RTX 3090. A typical sig53 modulation classifier trains in
~3 hours on a rented A4000 for under €2.

## 9. When in doubt — questions to ask

Before adding code, ask:

1. **Does this couple to specific hardware?** If yes, it goes in a backend module, not
   in core.
2. **Does this require an external compute provider?** If yes, it goes behind the
   `ComputeBackend` abstraction.
3. **Could this be tested with mock + file-replay backends?** If no, it's incorrectly
   designed. Fix the design.
4. **Does this depend on the orchestrator?** If yes, it goes in `src/rfdf/orchestrator/`
   and is lazy-imported with a clear error message if `ai-orchestrator-client` isn't
   installed.
5. **Is this domain-specific?** (drone-detection, aero, etc.) If yes, it goes in a
   downstream consumer repo or in `contrib/`, not in core.
6. **Have I documented the decision?** If no, write it in CLAUDE.md or ARCHITECTURE.md
   before merging.

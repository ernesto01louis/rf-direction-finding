# Standalone vs orchestrator

rfdf is **standalone-first**. Someone who has never heard of the
ai-orchestrator can `pip install rfdf` and run the entire DOA +
classification + capture pipeline. The orchestrator integration is a
bonus — it is never a baseline.

## The two modes

| | Standalone | Orchestrator-connected |
|---|---|---|
| Install | `pip install rfdf` | `pip install rfdf[orchestrator]` |
| DOA / classification / capture | ✅ full pipeline | ✅ identical |
| Result persistence | local JSON + SigMF | local JSON + SigMF |
| Evidence bundle | local only (`build_bundle`) | + pushed to the orchestrator |
| Hindsight memory | — | ✅ detections become memory |
| L5 vault notes | — | ✅ calibrations, model cards, … |
| ntfy alerts | — | ✅ three channels |
| Planner-dispatched flowgraphs | — | ✅ `FlowgraphBridge` |
| Capability discovery by the planner | — | ✅ after `rfdf orchestrator register` |

Both modes produce the **same answer**. The orchestrator mode adds
context, traceability, and integration — nothing else.

## Standalone is not "degraded"

A standalone run is *less context-rich*, not degraded. Its evidence
bundle is `quality: "citation-grade"`: the git SHA, canonicalised
inputs, and reproducibility hash are a complete provenance trail. A
bundle is only `degraded` when an orchestrator-bound write was
attempted mid-run and the orchestrator was unreachable — see
[evidence-bundles](orchestrator/evidence-bundles.md).

## The optionality is the architecture

The platform-not-hub principle: the orchestrator does not know about
RF; rfdf does not require the orchestrator. They compose cleanly via
the consumer pattern. Every orchestrator code path in rfdf is gated on
`rfdf.orchestrator.is_available()` and lazy-imports
`ai-orchestrator-client` — any code path that breaks without the
orchestrator is a bug.

Run [`examples/06-standalone-vs-orchestrator/`](../examples/06-standalone-vs-orchestrator/)
to see both modes side by side.

# 06 — Standalone vs orchestrator-connected

The single most important thing to understand about rfdf's orchestrator
integration: **it is optional, and it never changes the answer.**

```bash
python examples/06-standalone-vs-orchestrator/demo.py
```

The script runs one MUSIC DOA estimate on a mock 8-element ULA with two
emitters at 35° and 80°, twice:

| | Standalone mode | Orchestrator-connected mode |
|---|---|---|
| DOA result | ✅ on screen + local JSON | ✅ identical |
| Evidence bundle | — | ✅ citation-grade, reproducibility hash |
| Hindsight memory entry | — | ✅ (with a reachable orchestrator) |
| L5 vault note | — | ✅ (with a reachable orchestrator) |
| ntfy alert | — | ✅ (with a reachable orchestrator) |
| Capability discovery by the planner | — | ✅ after `rfdf orchestrator register` |

Both modes produce the **same** `azimuth_deg` estimate. Orchestrator
mode adds context, traceability, and memory — it is **not** a different
or better answer.

## Standalone is not "degraded"

A standalone run with no orchestrator is *less context-rich*, not
degraded. Its evidence bundle is still `quality: "citation-grade"`: the
git SHA, the canonicalised inputs, and the reproducibility hash are a
complete provenance trail without any LLM involvement. A bundle is only
`degraded` when an orchestrator-bound write was attempted mid-run and
the orchestrator was unreachable.

## Trying orchestrator mode

```bash
pip install rfdf[orchestrator]
export ORCHESTRATOR_URL=http://your-orchestrator:8000
export ORCHESTRATOR_TOKEN=...           # if the orchestrator enforces auth
rfdf orchestrator status                # check connectivity
rfdf orchestrator register --base-url http://this-host:8001 --callback-token ...
```

See [docs/orchestrator/](../../docs/orchestrator/) for the full
integration guide.

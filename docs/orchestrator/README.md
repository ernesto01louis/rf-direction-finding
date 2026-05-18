# Orchestrator integration

rfdf runs **standalone-first**. The optional `[orchestrator]` extra adds
integration with the [ai-orchestrator](https://github.com/ernesto01louis/ai-orchestrator):
consumer registration, evidence bundles, Hindsight memory writes, L5
vault notes, planner-dispatched GNU Radio flowgraphs, and ntfy alerts.

> The platform works fully without any of this. The orchestrator is a
> bonus, not a baseline — see [standalone-vs-orchestrator](../standalone-vs-orchestrator.md).

## Pages

- [installation.md](installation.md) — installing the optional dependency
- [consumer-pattern.md](consumer-pattern.md) — how rfdf registers as a consumer
- [evidence-bundles.md](evidence-bundles.md) — citation-grade evidence bundles
- [hindsight-and-vault.md](hindsight-and-vault.md) — memory + L5 vault writes
- [flowgraph-generation.md](flowgraph-generation.md) — planner-dispatched flowgraphs
- [ntfy-alerts.md](ntfy-alerts.md) — RF event alerting

## The lazy-import contract

`import rfdf.orchestrator` always succeeds. Accessing an integration
class without the extra raises `OrchestratorNotAvailableError` with an
install hint. `ai_orchestrator_client` is imported only inside
`rfdf.orchestrator._real` / its sibling modules — never at the top of a
`dsp/` or `ml/` module.

```python
import rfdf.orchestrator as orch

if orch.is_available():
    consumer = orch.RfdfConsumer()
```

## CLI

```
rfdf orchestrator status      # connection state + declared capabilities
rfdf orchestrator register    # register rfdf as a consumer
rfdf orchestrator hindsight   # manual Hindsight memory write
rfdf orchestrator vault       # manual L5 vault note
rfdf orchestrator planner     # request a flowgraph from the planner
```

Every subcommand degrades gracefully — with the extra absent it prints
an install hint and exits non-zero.

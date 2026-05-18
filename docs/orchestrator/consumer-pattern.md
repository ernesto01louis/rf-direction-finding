# Consumer pattern

rfdf registers with the orchestrator as a **consumer** — exactly the
pattern any research project (aero-research-platform, …) uses. A
consumer declares capabilities and exposes handlers the orchestrator's
planner can dispatch work to.

## RfdfConsumer

`rfdf.orchestrator.RfdfConsumer` subclasses the SDK's `Consumer` base
and declares its capabilities with the `@capability` decorator. Each
handler is a thin adapter over the standalone platform API:

| Capability | Delegates to |
|---|---|
| `rf.doa.run` | `rfdf.dsp.doa.Doa` on a mock SDR |
| `rf.classify` | `rfdf.ml.inference.Classifier` |
| `rf.geometry.morph` | `rfdf.dsp.geometry_presets` |

No DSP/ML business logic lives in the consumer — it is pure adapter.

## Registering

```python
from ai_orchestrator_client import OrchestratorClient
from rfdf.orchestrator import RfdfConsumer
from rfdf.orchestrator.consumer import DATA_PLANE_CAPABILITIES

consumer = RfdfConsumer()
with OrchestratorClient(base_url="http://orchestrator:8000") as client:
    consumer.register(
        client,
        base_url="http://this-host:8001",   # where the rfdf API is reachable
        callback_token="…",                  # presented by the orchestrator on dispatch
        extra_capabilities=DATA_PLANE_CAPABILITIES,
    )
```

`DATA_PLANE_CAPABILITIES` (`memory.write`, `vault.write`, `notify.send`,
`evidence.push`) opt the consumer into the orchestrator's push
endpoints — see [hindsight-and-vault.md](hindsight-and-vault.md).

Or from the CLI:

```bash
rfdf orchestrator register --base-url http://this-host:8001 --callback-token …
```

## Dispatch — the bidirectional half

Once registered, the orchestrator can call
`POST /capabilities/{capability}/invoke`; it proxies to the rfdf REST
API's `POST /capabilities/{capability}`, which runs
`RfdfConsumer.dispatch(...)`. Start that server with `rfdf api serve`
(needs the `[api]` extra).

`dispatch` is framework-agnostic — the SDK's `Consumer` never imports a
web framework; rfdf's own FastAPI app wires `dispatch` into an HTTP
route.

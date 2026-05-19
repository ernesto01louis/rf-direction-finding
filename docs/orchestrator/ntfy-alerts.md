# ntfy alerts

`rfdf.orchestrator.RfdfAlerts` pushes interesting RF events to the
orchestrator's notification service (Gotify primary, ntfy fallback)
over three logical channels.

```python
from ai_orchestrator_client import OrchestratorClient
from rfdf.orchestrator import RfdfAlerts

with OrchestratorClient(base_url="http://orchestrator:8000") as client:
    alerts = RfdfAlerts(client)
    alerts.alerts("calibration drift", "channel 3 phase drifted 12 deg")
    alerts.ops("capture started", "868 MHz sweep, 6 stations")
    alerts.research("novel pattern", "unclassified OFDM variant at 5.8 GHz")
```

## Channels

| Channel | Default priority | For |
|---|---|---|
| `alerts` | 4 (high) | unusual emitters, calibration drift, hardware errors |
| `ops` | 3 (normal) | capture/training lifecycle events |
| `research` | 2 (low) | scientific findings, novel signal patterns, trends |

`.alerts()`, `.ops()`, and `.research()` are convenience wrappers over
`alert(channel, …)`. The orchestrator's notify endpoint is a single
sink, so the channel is carried as a title prefix (`[rfdf-alerts] …`),
a tag, and the default priority. Pass `priority=` to override.

## Fail-tolerance

Every call is fail-tolerant — a notification to an unreachable
orchestrator returns `{"status": "failed", …}` and never raises. An
unknown channel name raises `ValueError` (a programming error, caught
at the call site, not a runtime condition).

## Capability requirement

Notifications go through the orchestrator's `/consumers/{id}/notify`
endpoint, gated on the `notify.send` capability. Register with
`DATA_PLANE_CAPABILITIES` to declare it — see
[consumer-pattern.md](consumer-pattern.md).

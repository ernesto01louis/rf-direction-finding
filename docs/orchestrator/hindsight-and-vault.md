# Hindsight memory + L5 vault

With the `[orchestrator]` extra, rfdf artifacts become orchestrator
**memory entries** (Hindsight) and wiki-linked **L5 vault notes**. Over
time the vault notes accumulate into the operator's RF research diary.

## RfdfRecorder

`rfdf.orchestrator.RfdfRecorder` binds once to an `OrchestratorClient`
and exposes domain-shaped helpers:

```python
from ai_orchestrator_client import OrchestratorClient
from rfdf.orchestrator import RfdfRecorder

with OrchestratorClient(base_url="http://orchestrator:8000") as client:
    rec = RfdfRecorder(client)
    rec.record_detection(
        bearing_deg=47.0, frequency_hz=5.8e9,
        modulation="OFDM", confidence=0.91,
    )
```

| Method | Writes |
|---|---|
| `record_detection` | Hindsight memory entry **+** an `rf-detection` note |
| `record_calibration` | `rf-calibration` note |
| `record_geometry_preset` | `rf-geometry-preset` note |
| `record_model_card` | `rf-model-card` note |
| `record_campaign` | `rf-campaign` note |

## Fail-tolerance

Every `record_*` method is **fail-tolerant** — a write to an
unreachable orchestrator logs a warning, returns a status dict
(`{"memory": "failed", ...}`), and **never raises**. A standalone
platform keeps working when the orchestrator is down; the writes are
bonus context, not a hard dependency.

## Thin clients

For ad-hoc writes the SDK's thin clients are exposed directly:

```python
from rfdf.orchestrator import Hindsight, Vault

Hindsight(client, "rfdf").write("Detected emitter at 47 deg, 5.8 GHz")
Vault(client, "rfdf").write_note("geometry-ula8", "8-element ULA", tags=["rf"])
```

The `rfdf orchestrator hindsight` / `rfdf orchestrator vault` CLI
subcommands wrap these for debugging.

## Capability requirement

The orchestrator gates each push endpoint on the consumer having
declared the matching generic capability (`memory.write`,
`vault.write`). `RfdfConsumer.to_registration(extra_capabilities=
DATA_PLANE_CAPABILITIES)` declares them — see
[consumer-pattern.md](consumer-pattern.md).

# Planner-dispatched GNU Radio flowgraphs

The orchestrator's coding planner can generate GNU Radio flowgraphs on
demand. `rfdf.orchestrator.FlowgraphBridge` turns the planner into an
RF DSL programmer: describe what you want, get a validated `.grc`.

```python
from ai_orchestrator_client import OrchestratorClient
from rfdf.orchestrator import FlowgraphBridge

with OrchestratorClient(base_url="http://orchestrator:8000") as client:
    bridge = FlowgraphBridge(client)
    fg = bridge.generate("detect AIS signals at 162 MHz and decode them")
    result = bridge.validate(fg)
    if result.ok:
        handle = bridge.deploy(fg, target_host="rfdf-tools")
```

Or from the CLI:

```bash
rfdf orchestrator planner --prompt "detect AIS at 162 MHz and decode"
```

## How it works

There is no SDK API for an ad-hoc planner prompt, so `generate()` wraps
the orchestrator's existing run pipeline:

1. `generate()` builds an `OrchestrateRequest` with a flowgraph-shaped
   prompt and dispatches it via `client.run`.
2. It waits for completion and reads the GRC XML back from the run
   result.
3. `validate()` compiles the `.grc` with `grcc --no-execute`.
4. `deploy()` `scp`s the validated `.grc` to the rfdf-tools host.

## grcc requirement

`grcc` ships with GNU Radio. It is **not** part of the rfdf dependency
tree — `validate()` / `deploy()` shell out to it. When `grcc` is
absent, `validate()` returns `ValidationResult(ok=False, output="grcc
not found …")` rather than raising. Install GNU Radio on the host that
runs the bridge (typically the rfdf-tools LXC).

## Self-correction

`validate()` returns the compiler output on failure. Feed it back to
the planner for a self-correction loop — generate, validate, and on
failure re-prompt with the `grcc` error until it compiles.

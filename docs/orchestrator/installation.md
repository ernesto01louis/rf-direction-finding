# Installing the orchestrator integration

The integration ships behind the optional `[orchestrator]` extra.

```bash
pip install rfdf                 # standalone — no orchestrator features
pip install rfdf[orchestrator]   # orchestrator integration available
```

The extra pulls in exactly one dependency —
[`ai-orchestrator-client`](https://pypi.org/project/ai-orchestrator-client/)
(`>=0.1.1,<0.2`) — and whatever that package pulls in. No heavy or
domain-flavoured libraries enter the dependency tree.

## Verifying

```bash
python -c "import rfdf.orchestrator; print('importable')"          # always works
python -c "import rfdf.orchestrator as o; print(o.is_available())" # True with the extra
rfdf orchestrator status
```

Without the extra, `rfdf.orchestrator` still imports — accessing an
integration class raises `OrchestratorNotAvailableError`:

```python
>>> import rfdf.orchestrator as orch
>>> orch.RfdfConsumer
OrchestratorNotAvailableError: rfdf.orchestrator.RfdfConsumer requires
the `ai-orchestrator-client` package. Install it with
`pip install rfdf[orchestrator]`.
```

## Connecting to an orchestrator

The CLI and `rfdf.orchestrator` helpers read two environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ORCHESTRATOR_URL` | `http://localhost:8000` | orchestrator base URL |
| `ORCHESTRATOR_TOKEN` | _(none)_ | bearer token, if the orchestrator enforces auth |

## Running the rfdf REST API

The capability server (the orchestrator's callback target) and the
standalone REST API ship behind the separate `[api]` extra:

```bash
pip install rfdf[api]
rfdf api serve --host 0.0.0.0 --port 8001
```

`RFDF_API_TOKEN`, when set, gates the `POST /capabilities/*` routes with
a bearer token; `GET /healthz` is always open.

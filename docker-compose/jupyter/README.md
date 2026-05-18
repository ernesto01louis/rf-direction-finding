# JupyterLab + code-server

The research scratchpad — ad-hoc experiments, paper figures, exploration.

| Service       | Image (built here)        | Port | Traefik route   |
|---------------|---------------------------|------|-----------------|
| `code-server` | `Dockerfile`              | 8443 | `code.rf.lan`   |
| `jupyterlab`  | `Dockerfile.jupyter`      | 8889 | `jupyter.rf.lan`|

Both images extend an upstream base with base `rfdf` pre-installed. The heavy
`[ml]` extra (torchsig's dependency tree backtracks pip for hours inside a
Docker build) is **not** baked in — add it in-container when needed
(`/opt/rfdf/bin/pip install 'rfdf[ml]'`) or use the rfdf-tools NFS venv.

> **Port note:** JupyterLab runs on **8889**, not the usual 8888 — Hindsight
> already uses 8888 on `192.168.2.203`. Recorded in
> `docs/infrastructure/troubleshooting.md`.

## Deploy

```sh
cp .env.example .env            # set CODE_SERVER_PASSWORD + JUPYTER_TOKEN
docker compose up -d --build
```

The `jupyter` Ansible role does this on `rfdf-jupyter`.

## Mounting the TrueNAS notebooks share

The `notebooks` volume defaults to a local Docker volume. In production, bind
it to the operator's TrueNAS NFS export so notebooks persist off-host — edit
the `volumes:` block in `compose.yml`:

```yaml
volumes:
  notebooks:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=192.168.2.222,ro=false"
      device: ":/mnt/tank/notebooks"
```

No dependency on the ai-orchestrator.

# Monitoring

Prometheus + Grafana + Alertmanager on `rfdf-dashboard`, `node_exporter` on
every LXC.

## Prometheus

`docker-compose/monitoring/prometheus/prometheus.yml` scrapes:

- **node** — `node_exporter` on every LXC (`:9100`).
- **traefik** — Traefik's Prometheus metrics endpoint.
- **rfdf-platform** — the rfdf REST `/metrics` endpoint. This target stays
  **down (and visibly so)** until the Stage-7 REST API ships — that is
  expected, not a fault.

Alerting rules (`alerts.yml`): `HostDown`, `HostHighMemory`, `HostDiskFilling`.

## Grafana

Datasource (Prometheus) and dashboards are **auto-provisioned** from
`grafana/provisioning/`. Two dashboards ship:

- **rfdf — host health** — hosts up, CPU / memory / disk per host.
- **rfdf — Traefik ingress** — requests/s and 5xx/s per service.

The platform-API and ML-jobs dashboards land in Stage 7 (they need the rfdf
`/metrics` endpoint). Grafana is pinned to **11.4.0**.

## Alerting → ntfy

`alertmanager.yml` posts every firing/resolved alert to the operator's
existing **ntfy** instance (`192.168.2.203:8090`, topic `rfdf-infra`) via a
webhook — the same channel the orchestrator uses. ntfy publishes Alertmanager's
native JSON payload; for prettier formatting, place an `alertmanager-ntfy`
transformer in front of ntfy and re-point the webhook.

`99-verify` fires a test message into this topic as its final check.

## Verifying

```sh
make infra-verify
```

confirms Prometheus is healthy with ≥1 node target up, Grafana is healthy with
both dashboards present, and the ntfy bridge is reachable.

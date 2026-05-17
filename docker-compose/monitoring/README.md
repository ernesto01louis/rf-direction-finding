# Monitoring — Prometheus + Grafana + Alertmanager

Cluster-wide observability for the rfdf ecosystem.

| Service        | Image                    | Role                                  |
|----------------|--------------------------|---------------------------------------|
| `prometheus`   | `prom/prometheus:v3.1.0` | Scrapes node-exporters + Traefik.      |
| `alertmanager` | `prom/alertmanager:v0.28.0` | Routes alerts to ntfy.             |
| `grafana`      | `grafana/grafana:11.4.0` | Dashboards (auto-provisioned).         |

`node_exporter` is installed on every LXC by the `06-dashboard` monitoring
play — Prometheus scrapes them at `:9100`.

## Grafana

Datasource (Prometheus) and dashboards are **auto-provisioned** from
`grafana/provisioning/` + `grafana/dashboards/`. Two dashboards ship:
`rfdf — host health` and `rfdf — Traefik ingress`. The platform-API and
ML-jobs dashboards land in Stage 7 once the rfdf REST `/metrics` endpoint
exists.

## Alerting → ntfy

`alertmanager.yml` posts every alert to the operator's existing ntfy instance
(`192.168.2.203:8090`, topic `rfdf-infra`) via a webhook. ntfy publishes
Alertmanager's native JSON payload. For prettier notifications, place an
`alertmanager-ntfy` transformer in front of ntfy and point the webhook at it.

## Deploy

```sh
cp .env.example .env            # set GRAFANA_ADMIN_PASSWORD
docker compose up -d
```

The `monitoring` role does this on `rfdf-dashboard`. No dependency on the
ai-orchestrator.

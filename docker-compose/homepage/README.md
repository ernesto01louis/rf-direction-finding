# Homepage — ecosystem dashboard

`gethomepage/homepage` is the top-level navigation for the rfdf ecosystem —
quick links, live RF widgets, grouped tool tiles, and infra status.

## Layout

| Path                      | Purpose                                          |
|---------------------------|--------------------------------------------------|
| `config/*.example.yaml`   | Templates — copy to `config/*.yaml`.             |
| `widgets/*.html`          | Custom rfdf widgets (raw SVG/JS, no build step). |

The `widgets` nginx sidecar serves the three custom widgets; Homepage embeds
them as `iframe` widgets (see `config/services.example.yaml`).

## Custom widgets

| Widget                          | Polls                          |
|---------------------------------|--------------------------------|
| `widget-rfdf-live-doa.html`     | `/rf/doa/latest` — polar compass of detected bearings |
| `widget-rfdf-recent-captures.html` | `/rf/captures/recent` — 10 latest SigMF captures |
| `widget-rfdf-active-jobs.html`  | `/rf/ml/jobs` — active training jobs + cost/ETA |

All three poll the **rfdf platform's own REST API** — never the
ai-orchestrator. That API is a Stage-7 deliverable; until it exists the
widgets show a "waiting for platform API" state rather than erroring.

## Deploy

```sh
cp .env.example .env
for f in config/*.example.yaml; do cp "$f" "${f%.example.yaml}.yaml"; done
docker network create rfdf-edge   # if not already present
docker compose up -d
```

The `homepage` Ansible role does this on `rfdf-dashboard`.

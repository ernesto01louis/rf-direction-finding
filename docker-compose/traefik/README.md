# Traefik — edge reverse proxy

Traefik v3 is the single ingress point for the rfdf ecosystem. Every browser
request enters here, is authenticated against Authelia (ForwardAuth), and is
proxied to the right service.

## Layout

| File                  | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `compose.yml`         | The Traefik container (ports 80/443).                |
| `config/traefik.yml`  | Static config — entrypoints, providers, metrics.     |
| `config/dynamic.yml`  | Dynamic config — all routers, services, middlewares. |
| `certs/`              | ACME / custom TLS material (gitignored).             |

## Standalone use

```sh
cp .env.example .env
docker network create rfdf-edge      # once, if not already present
docker compose up -d
```

This service has **no dependency on the ai-orchestrator** — it is a plain
reverse proxy.

## Deployed by Ansible

The `traefik` role deploys this stack to `rfdf-dashboard`. Add or change a
route by editing `config/dynamic.yml` and re-running `06-dashboard.yml` — the
file provider hot-reloads, no restart needed.

## TLS

Ships with Traefik's self-signed default certificate (browsers warn; fine on a
`.lan` domain). For browser-trusted certs, enable the ACME resolver in
`traefik.yml` (public hostnames via Tailscale Funnel) or point it at an
internal step-ca. See `docs/infrastructure/tailscale.md`.

## Auth

Every router carries the `authelia` middleware except `auth.rf.lan` (Authelia's
own portal). The Traefik dashboard at `traefik.rf.lan` is SSO-gated like every
other service.

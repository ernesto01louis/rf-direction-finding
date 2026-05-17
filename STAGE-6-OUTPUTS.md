# Stage 6 outputs — software ecosystem hosting

**Tag:** `v0.1.0-beta` · **Shipped:** 2026-05-17 · **PRs:** #38–#42

Stage 6 stands up the operator-facing "workshop" — the open-source RF / SDR /
EM-simulation tool ecosystem, hosted on adjacent Proxmox infrastructure and
reached through a browser. It is **pure infrastructure**: Ansible playbooks,
Docker Compose stacks, Kasm image definitions, Traefik/Authelia/Homepage
configs, monitoring, docs. **Zero `src/rfdf/` changes** — verified by
`git diff v0.1.0-alpha..v0.1.0-beta -- src/` returning empty.

## Sub-sessions (5 PRs)

| PR | Branch | Delivered |
|----|--------|-----------|
| #38 | `infra/stage6-foundation` | Ansible skeleton, LXC/VM bootstrap, Traefik + Authelia |
| #39 | `infra/stage6-kasm-guac` | Kasm Workspaces + 4 images, Guacamole, Win11 VM shell |
| #40 | `infra/stage6-tools-jupyter` | rfdf-tools toolchain, OpenWebRX+, JupyterLab + code-server |
| #41 | `infra/stage6-dashboard-monitoring` | Homepage + custom widgets, Prometheus/Grafana/Alertmanager |
| #42 | `infra/stage6-verify-release` | `99-verify`, docs, CHANGELOG/ROADMAP/SECURITY, tag |

Each PR: green `infra-ci` (yamllint + ansible-lint + `docker compose config` +
hadolint) and the full existing `ci.yml` suite (untouched — Stage 6 adds no
Python).

## Deliverables

- **`ansible/`** — 14 roles, 10 playbooks (`00-bootstrap` … `08-openwebrx`,
  `99-verify`), inventory + group_vars + `ansible-vault` secrets. Idempotent,
  parameterised. `make provision` / `make provision-foundation` /
  `make infra-verify` / `make infra-lint`.
- **`docker-compose/`** — 7 standalone-runnable stacks: `traefik`, `authelia`,
  `guacamole`, `openwebrx`, `jupyter`, `homepage`, `monitoring`. (`kasm` has no
  Compose file — Kasm CE installs via its official installer.)
- **4 Kasm workspace images** under `docker-compose/kasm/dockerfiles/`,
  published to GHCR by the `kasm-images` workflow on Stage-6+ tags.
- **Homepage** dashboard + 3 dependency-free custom widgets (live DOA, recent
  captures, active ML jobs).
- **Authelia** SSO (ForwardAuth on every Traefik route + OIDC provider);
  **Traefik v3** file-provider ingress; **Tailscale** optional on every host.
- **Monitoring** — `node_exporter` everywhere, Prometheus + Grafana
  (2 auto-provisioned dashboards) + Alertmanager → ntfy.
- **Docs** — `docs/infrastructure/` (8) + `docs/workspaces/` (4).
- **CI** — `infra-ci.yml`, `kasm-images.yml`; `infra` commit scope.

## Deviations from the brief (surfaced, not papered over)

1. **IP plan** — the brief's `.220–.230` collided with 4 existing hosts on the
   reference network (`.222` TrueNAS, `.224`, `.226`, `.230`). Shifted to the
   clean contiguous block `.239–.248`. Fully parameterised in `inventory.yml`.
2. **rfdf REST API** — `src/rfdf/api/` is an empty scaffold; the REST API is a
   **Stage-7 deliverable**. The `rfdf_platform` role installs the CLI/library
   and ships the `rfdf-api` systemd unit **disabled**. The `api.rf.lan` route,
   the `99-verify` `/healthz` check, and the Homepage live-RF widgets are
   provisioned but inert until Stage 7 (the widgets degrade gracefully).
3. **Symmetry note** — the brief asked to mirror the ai-orchestrator's
   docker-compose / Authelia / Traefik patterns; the orchestrator has none of
   those (bash install scripts, no Docker/Ansible/SSO). Stage 6 is a fresh
   design per the rfdf brief.
4. **Kasm** — no `compose.yml`; Kasm CE installs via its official multi-
   container installer (the `kasm` role drives it, guarded to run once).
5. **rfdf-daq** — provisioned as a privileged LXC with USB device passthrough
   (simpler than VM USB-controller passthrough; the VM path is documented).
6. **SDRangel** — no Debian package; documented as the one manual
   build-from-source tool. Every other catalogue tool installs automatically.
7. **jupyter stack** — ships two Dockerfiles (code-server, JupyterLab) rather
   than the brief's single file — they need different bases.
8. **Win11 VM** — **GPU passthrough = NO** (GPU reserved for ML training).
9. **alertmanager → ntfy** — a direct webhook (ntfy publishes Alertmanager's
   native JSON); a transformer for prettier output is documented as optional.

## Verification

- `git diff v0.1.0-alpha..v0.1.0-beta -- src/` → **empty**. Coverage delta ≈ 0.
- `yamllint` + `ansible-lint` (passes the `production` profile) — clean.
- `ansible-inventory --graph` + `ansible-playbook --syntax-check` — all OK.
- `infra-ci` green on every PR; `ci.yml`'s 8 jobs stay green throughout.
- **Live provisioning** — the preflight confirmed the Proxmox API is reachable
  at `192.168.2.13:8006` and the `.239–.248` block is free. Running
  `make provision` against the cluster needs the operator to supply a Proxmox
  API token + storage/bridge details + (optionally) a Tailscale auth key in
  `group_vars/all/vault.yml`; until then the IaC is authored, committed, and
  CI-green but the live cluster + `99-verify` run + screenshots are pending
  that operator step.

## Stage 7 (next)

Stage 7 (`v0.1.0` GA) wires the optional ai-orchestrator integration: a
lazy-imported `[orchestrator]` extra under `src/rfdf/orchestrator/`, consumer
registration mirroring `aero-research-platform`, **the rfdf REST API**
(`src/rfdf/api/` — which unblocks the `api.rf.lan` route, the `/healthz`
verify check, and the Homepage live-RF widgets shipped dormant here),
evidence-bundle production with a `quality: degraded` flag for fallback runs,
Hindsight + L5-vault writes, planner-dispatched GNU Radio flowgraphs, ntfy
alerting, coverage held ≥ 80%, and the first PyPI publish of `rfdf`.

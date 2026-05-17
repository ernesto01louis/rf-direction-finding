# rfdf software ecosystem — overview

Stage 6 stands up the **operator-facing workshop**: the open-source RF / SDR /
EM-simulation tool stack, hosted on adjacent infrastructure and reached through
a browser. It is *operator quality-of-life* — **not** a dependency of the
platform.

> `pip install rfdf` does not depend on any of this. A user running `rfdf doa`
> needs none of these services; a user running MMANA-GAL needs no rfdf install.
> Both coexist on the same infrastructure via this stage's deliverables.

## Topology

Ten guests on a Proxmox host (reference IP plan — everything is parameterized
in `ansible/inventory.yml`):

| Host            | IP              | Role                                  |
|-----------------|-----------------|---------------------------------------|
| rfdf-platform   | 192.168.2.239   | rfdf platform (REST API — Stage 7)    |
| rfdf-daq        | 192.168.2.240   | UHD + B210 capture host               |
| rfdf-tools      | 192.168.2.241   | Linux-native RF toolchain             |
| rfdf-kasm       | 192.168.2.242   | Kasm Workspaces server                |
| rfdf-guac       | 192.168.2.243   | Apache Guacamole gateway              |
| rfdf-winrf      | 192.168.2.244   | Win11 VM (Windows-only tools)         |
| rfdf-mltrain    | 192.168.2.245   | ML-training orchestration placeholder |
| rfdf-dashboard  | 192.168.2.246   | Traefik + Authelia + Homepage + Prom  |
| rfdf-jupyter    | 192.168.2.247   | JupyterLab + code-server              |
| rfdf-openwebrx  | 192.168.2.248   | OpenWebRX+ always-on receiver         |

(The brief's `.220–.230` example collided with existing hosts on the reference
network; `.239–.248` is the first clean contiguous block.)

## Every service is standalone

No service in this stage depends on the **ai-orchestrator**. Each Docker
Compose stack under `docker-compose/` runs on its own (`cp .env.example .env`
&rarr; `docker compose up -d`). The Homepage widgets poll the *rfdf platform's
own* REST API — never the orchestrator.

## Documentation map

| Doc                       | Covers                                          |
|---------------------------|-------------------------------------------------|
| `proxmox-setup.md`        | Proxmox host prerequisites, the Win11 VM build  |
| `ansible-deployment.md`   | Running the playbooks, inventory customisation  |
| `tool-catalog.md`         | What is installed where                         |
| `authentication.md`       | Authelia SSO, 2FA, the worked auth flow         |
| `tailscale.md`            | Tailscale ACLs, Funnel guidance                 |
| `monitoring.md`           | Prometheus, Grafana, ntfy alerting              |
| `troubleshooting.md`      | Common issues                                   |
| `../workspaces/`          | Per-Kasm-workspace usage guides                 |

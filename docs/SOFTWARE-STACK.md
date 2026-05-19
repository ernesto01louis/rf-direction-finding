# The Open-Source RF Software Stack

A guide to the hosted **software ecosystem** that ships alongside `rfdf` — a
browser-reachable workshop of ~30 open-source RF / SDR / EM-simulation tools
(GNU Radio, GQRX, OpenWebRX+, URH, xnec2c, JupyterLab, and more), running on a
small fleet of Proxmox containers behind a single sign-on.

This is the companion to the [USER-GUIDE.md](USER-GUIDE.md) (which covers the
`rfdf` platform itself). If you only want to *use the rfdf library*, you do not
need any of this.

---

## Table of contents

1. [What the stack is — and what it is not](#1-what-the-stack-is--and-what-it-is-not)
2. [Current status: it is not running yet](#2-current-status-it-is-not-running-yet)
3. [Prerequisites](#3-prerequisites)
4. [Deploying the stack](#4-deploying-the-stack)
5. [The ten hosts](#5-the-ten-hosts)
6. [Reaching it from your browser](#6-reaching-it-from-your-browser)
7. [Access matrix — every tool, every URL](#7-access-matrix--every-tool-every-url)
8. [The RF tool catalog](#8-the-rf-tool-catalog)
9. [First-login flow](#9-first-login-flow)
10. [Day-2 operations](#10-day-2-operations)

---

## 1. What the stack is — and what it is not

`rfdf` core is a Python library. But RF research also leans on a pile of
*graphical* open-source tools — spectrum analysers, flowgraph editors, antenna
modellers, protocol decoders — that are awkward to install and even more awkward
to run on a headless server.

The **software stack** solves that. It is an *infrastructure-as-code* definition
(Ansible + Docker Compose) that stands up a handful of Proxmox LXC containers,
installs those tools, and publishes them all through one reverse proxy so you
reach every one of them from a browser tab.

**It is operator convenience — never a dependency of `rfdf`.** Nothing in the
`rfdf` library, CLI, or REST API needs the stack to be running. You can use rfdf
forever and never deploy this. The stack and the platform are deliberately
decoupled.

It is delivered as two directories in this repository:

- [`ansible/`](../ansible/) — playbooks and roles that create the containers and
  install the software.
- [`docker-compose/`](../docker-compose/) — the service stacks (Traefik,
  Authelia, Guacamole, OpenWebRX+, JupyterLab, Homepage, monitoring).

---

## 2. Current status: it is not running yet

**Important.** This repository ships the *recipe* for the stack — the Ansible
and Docker Compose code, CI-linted and committed. It does **not** ship a running
cluster. Nothing is live until an operator runs the provisioning against their
own Proxmox host.

So "how do I access the tools?" has two parts:

1. **Deploy it** — [§4](#4-deploying-the-stack). One operator, one time.
2. **Reach it** — [§6](#6-reaching-it-from-your-browser) and
   [§7](#7-access-matrix--every-tool-every-url). Anyone, any browser, afterwards.

Until step 1 happens, the URLs in this guide resolve to nothing. This is by
design — the stack assumes infrastructure (a Proxmox host, an IP range, secrets)
that only the operator can supply.

---

## 3. Prerequisites

To deploy the stack you need:

- **A Proxmox VE host** with spare capacity (the reference host carries 96 GB
  RAM; the ten guests are sized modestly — see [§5](#5-the-ten-hosts)).
- **A Proxmox API token** so Ansible can create the containers. See
  [docs/infrastructure/proxmox-setup.md](infrastructure/proxmox-setup.md).
- **A free contiguous IP range.** The reference inventory uses
  `192.168.2.239–.248`; every address is parameterised, so edit it for your
  network.
- **Ansible** on the machine you deploy from, plus the collections the playbooks
  need (`make infra-deps` installs them).
- **A Windows 11 ISO** if you want the optional Windows-tools VM.
- For two of the hosts (the SDR-attached ones), **USB passthrough** configured on
  the Proxmox host.

---

## 4. Deploying the stack

The whole sequence is driven from this repository's `Makefile`.

### Step 1 — describe your network

```bash
cd ansible
cp inventory.example.yml inventory.yml
$EDITOR inventory.yml          # set IPs, VMIDs, and per-host sizing for YOUR network
```

### Step 2 — supply secrets

The playbooks read secrets (the Proxmox API token, admin passwords, the optional
Tailscale auth key) from an `ansible-vault`-encrypted file at
`ansible/group_vars/all/vault.yml`. Populate and encrypt it following
[docs/infrastructure/ansible-deployment.md](infrastructure/ansible-deployment.md),
which lists every required key.

### Step 3 — provision

```bash
cd /path/to/rf-direction-finding

make provision               # runs every playbook 00-bootstrap … 08-openwebrx in order
```

The playbooks run in numbered order and are all **idempotent** — re-running
`make provision` is safe and reconciles any drift.

| Playbook | Brings up |
|---|---|
| `00-bootstrap` | the LXC/VM containers themselves, base OS |
| `01-platform` | the `rfdf` CLI/library + REST API host |
| `02-daq` | UHD + B210 capture host |
| `03-tools` | the Linux RF toolchain (NFS-exported to Kasm) |
| `04-kasm` | Kasm Workspaces (streamed GUI desktops) |
| `05-guacamole` | Apache Guacamole (Windows RDP gateway) |
| `06-dashboard` | Traefik + Authelia + Homepage + Prometheus/Grafana |
| `07-jupyter` | JupyterLab + code-server |
| `08-openwebrx` | OpenWebRX+ always-on web SDR |

To bring up just the load-bearing foundation first (containers + the platform +
the Traefik/Authelia/dashboard layer):

```bash
make provision-foundation
```

You can also run a single playbook directly, e.g.
`cd ansible && ansible-playbook playbooks/08-openwebrx.yml`.

### Step 4 — verify

```bash
make infra-verify             # runs the 99-verify end-to-end smoke playbook
```

`99-verify` checks every container is reachable, every service answers, and the
monitoring → ntfy alert bridge works.

---

## 5. The ten hosts

`make provision` creates **nine LXC containers and one Windows VM**. The IPs and
VMIDs below are the *reference defaults* from `inventory.example.yml` — yours
will differ if you edited the inventory.

| Host | IP | VMID | Role |
|---|---|---|---|
| `rfdf-platform` | 192.168.2.239 | 6239 | the `rfdf` CLI/library + REST API |
| `rfdf-daq` | 192.168.2.240 | 6240 | UHD + USRP B210 coherent capture (USB passthrough) |
| `rfdf-tools` | 192.168.2.241 | 6241 | the Linux RF toolchain, NFS-exported to Kasm |
| `rfdf-kasm` | 192.168.2.242 | 6242 | Kasm Workspaces — streamed GUI desktops |
| `rfdf-guac` | 192.168.2.243 | 6243 | Apache Guacamole — Windows RDP gateway |
| `rfdf-winrf` | 192.168.2.244 | 6244 | Windows 11 VM (vendor RF tools) |
| `rfdf-mltrain` | 192.168.2.245 | 6245 | ML-training host |
| `rfdf-dashboard` | 192.168.2.246 | 6246 | Traefik + Authelia + Homepage + Prometheus/Grafana |
| `rfdf-jupyter` | 192.168.2.247 | 6247 | JupyterLab + code-server |
| `rfdf-openwebrx` | 192.168.2.248 | 6248 | OpenWebRX+ always-on web SDR (USB passthrough) |

> **Note on the IP range.** The original project brief suggested `.220–.230`,
> but that block collided with existing hosts on the reference network, so the
> shipped inventory uses the clean contiguous block `.239–.248`. It is fully
> parameterised — change it in `inventory.yml`.

---

## 6. Reaching it from your browser

Every web tool is published through a **Traefik** reverse proxy on the
`rfdf-dashboard` host, behind **Authelia** single sign-on. You reach tools by
hostname, not by IP:port.

The reference deployment uses the `rf.lan` domain. Because `.lan` is not real
DNS, point your machine at the dashboard host by adding this line to your
`/etc/hosts` (or your router's DNS, or Tailscale MagicDNS):

```
192.168.2.246  rf.lan auth.rf.lan traefik.rf.lan kasm.rf.lan guac.rf.lan sdr.rf.lan jupyter.rf.lan code.rf.lan api.rf.lan grafana.rf.lan prometheus.rf.lan
```

Then open **`https://rf.lan`** — the Homepage dashboard, which links to
everything else.

Notes:

- TLS is **self-signed** by default, so your browser will warn on first visit.
  That is expected on a `.lan` domain; accept it, or wire ACME / an internal CA
  in `docker-compose/traefik/config/traefik.yml`.
- The domain `rf.lan` is itself an example. To use your own, edit the `Host(...)`
  rules in `docker-compose/traefik/config/dynamic.yml`.
- A few services can also be hit **directly by IP:port**, bypassing SSO — useful
  for debugging (see the access matrix below).

---

## 7. Access matrix — every tool, every URL

| Tool | Browser URL (via Traefik) | Direct URL | What it is | Login |
|---|---|---|---|---|
| **Homepage** | `https://rf.lan` | — | The central dashboard linking every tool | Authelia SSO |
| **Authelia** | `https://auth.rf.lan` | — | The single-sign-on portal itself | password + TOTP |
| **Traefik dashboard** | `https://traefik.rf.lan` | — | Live view of proxy routes + traffic | Authelia SSO |
| **Kasm Workspaces** | `https://kasm.rf.lan` | `https://192.168.2.242` | Streamed GUI Linux desktops full of RF tools | Kasm admin account |
| **Guacamole** | `https://guac.rf.lan` | `http://192.168.2.243:8080` | Gateway to the Windows RDP desktop | `guacadmin` / `guacadmin` (change it!) |
| **OpenWebRX+** | `https://sdr.rf.lan` | `http://192.168.2.248:8073` | Always-on web SDR waterfall + decoders | open / admin panel at `/admin` |
| **JupyterLab** | `https://jupyter.rf.lan` | `http://192.168.2.247:8889` | Notebooks with `rfdf` pre-installed | Jupyter token |
| **code-server** | `https://code.rf.lan` | `http://192.168.2.247:8443` | VS Code in the browser | code-server password |
| **rfdf REST API** | `https://api.rf.lan` | `http://192.168.2.239:8000` | The `rfdf` REST API ([USER-GUIDE §11](USER-GUIDE.md#11-the-rest-api)) | Authelia SSO |
| **Grafana** | `https://grafana.rf.lan` | `http://192.168.2.246:3000` | Infrastructure dashboards | Grafana admin account |
| **Prometheus** | `https://prometheus.rf.lan` | `http://192.168.2.246:9090` | Metrics + query interface | open behind SSO |

Credentials marked "admin account" / "token" / "password" are the values you set
in the Ansible vault during deployment ([§4](#4-deploying-the-stack)).

> The `rfdf REST API` route is provisioned by the stack but its host runs the
> API as a systemd unit that ships **disabled**. Enable it on `rfdf-platform`
> once you want `api.rf.lan` live — see [USER-GUIDE §11](USER-GUIDE.md#11-the-rest-api)
> for what the API does.

### Kasm Workspaces — the GUI tools

Kasm streams a full Linux desktop to your browser tab. After logging in at
`https://kasm.rf.lan`, pick a **workspace** and launch a session. Four workspace
images ship with the stack:

| Workspace image | Contains |
|---|---|
| `rfdf-kasm-ubuntu-rftools` | GNU Radio Companion, GQRX, CubicSDR, inspectrum, URH, Wireshark, Kismet, xnec2c, gpredict — plus the `rfdf` toolchain mounted over NFS |
| `rfdf-kasm-ubuntu-wine-antennas` | Antenna design: MMANA-GAL Pro and 4nec2 under Wine, plus native xnec2c |
| `rfdf-kasm-kali-rf` | Kali Linux with RF + wireless-security tooling |
| `rfdf-kasm-jupyter-rfdf` | JupyterLab inside a Kasm desktop, `rfdf` pre-installed |

The images are built by this repo's `kasm-images` CI workflow and published to
GHCR. After Kasm is up you register them once in **Admin → Workspaces** (by their
`ghcr.io/...` image URI). See
[docs/workspaces/](workspaces/) for the per-image detail.

### Guacamole — the Windows desktop

The `rfdf-winrf` Windows 11 VM is reached *through* Guacamole (no RDP client
needed). After deployment you install Windows by hand, then add an RDP connection
in Guacamole pointing at `192.168.2.244:3389`. Use it for vendor-locked Windows
RF tools (SDR#, HDSDR, …). See
[docs/infrastructure/proxmox-setup.md](infrastructure/proxmox-setup.md).

---

## 8. The RF tool catalog

The full, authoritative list of what is installed where is
[docs/infrastructure/tool-catalog.md](infrastructure/tool-catalog.md). In summary:

**On `rfdf-tools`** (CLI tools + a shared Python environment, NFS-exported into
the Kasm `rftools` workspace):

- *SDR core:* GNU Radio, GQRX, CubicSDR, rtl-sdr, HackRF tools.
- *Analysis:* inspectrum, URH (Universal Radio Hacker), Wireshark, Kismet.
- *Protocol decoders:* `dump1090` (ADS-B), `rtl_433` (IoT), AIS-catcher,
  `multimon-ng`.
- *Antenna / EM:* xnec2c, nec2c.
- *Satellite:* gpredict, SatDump.
- *A shared Python venv* with `rfdf` itself plus torchsig, sigmf, scikit-rf,
  PyNEC, pyargus, pyadi-iio, nanovna-saver.
- *Source checkouts* of KrakenSDR DOA, gr-lora_sdr, gr-iridium, AIS-catcher.

**Through Kasm Workspaces** — the GUI applications above, streamed to the
browser (see [§7](#7-access-matrix--every-tool-every-url)).

**On `rfdf-openwebrx`** — OpenWebRX+ runs a 24/7 web SDR with built-in digital
decoders (FT8, FT4, WSPR, JS8, and more).

**On the Windows VM** — vendor-locked Windows-only tools that have no Linux
build, reached via Guacamole.

> One tool, **SDRangel**, has no Debian package and is documented as a manual
> build-from-source step; every other catalogue tool installs automatically.

---

## 9. First-login flow

When the stack is freshly deployed:

1. **Add the `/etc/hosts` line** from [§6](#6-reaching-it-from-your-browser).
2. **Open `https://rf.lan`.** Traefik redirects you to the Authelia portal.
3. **Log in to Authelia** with the admin username + password you set in the
   vault. On the *first* login Authelia walks you through enrolling a **TOTP
   second factor** — scan the QR code with an authenticator app (Aegis, Authy,
   Google Authenticator) and **save the recovery codes**. Every later login
   needs password + 6-digit code.
4. You land on the **Homepage** dashboard — click through to any tool.
5. **Change the Guacamole default password.** Guacamole ships with
   `guacadmin` / `guacadmin`; log in once and change it immediately.

Authentication details, including the OIDC provider Authelia also exposes, are in
[docs/infrastructure/authentication.md](infrastructure/authentication.md).

---

## 10. Day-2 operations

### Update or restart one service

The Docker Compose stacks live on their hosts under `/opt/stacks/<service>/`:

```bash
# on the relevant host
cd /opt/stacks/openwebrx
docker compose down && docker compose up -d
docker compose logs -f          # tail the logs
```

Or re-run the owning playbook from the deploy machine:

```bash
cd ansible && ansible-playbook playbooks/08-openwebrx.yml
```

### Add or change a Traefik route

Edit `docker-compose/traefik/config/dynamic.yml` — Traefik's file provider
hot-reloads it, no restart needed.

### Re-provision after an inventory change

`make provision` is idempotent. Edit `inventory.yml`, re-run it, and Ansible
reconciles the difference.

### Monitoring

`rfdf-dashboard` runs Prometheus + Grafana + Alertmanager, with `node_exporter`
on every host. Two Grafana dashboards (host health, Traefik ingress) are
auto-provisioned, and Alertmanager forwards firing alerts to an ntfy topic. See
[docs/infrastructure/monitoring.md](infrastructure/monitoring.md).

### Reference documentation

| Topic | Document |
|---|---|
| Proxmox host prep + the Windows VM | [docs/infrastructure/proxmox-setup.md](infrastructure/proxmox-setup.md) |
| Playbook order, idempotency, IP customisation | [docs/infrastructure/ansible-deployment.md](infrastructure/ansible-deployment.md) |
| Which tool is installed where | [docs/infrastructure/tool-catalog.md](infrastructure/tool-catalog.md) |
| Authelia SSO, 2FA, OIDC | [docs/infrastructure/authentication.md](infrastructure/authentication.md) |
| Prometheus / Grafana / Alertmanager | [docs/infrastructure/monitoring.md](infrastructure/monitoring.md) |
| Optional Tailscale remote access | [docs/infrastructure/tailscale.md](infrastructure/tailscale.md) |
| Infrastructure troubleshooting | [docs/infrastructure/troubleshooting.md](infrastructure/troubleshooting.md) |
| The Kasm workspace images | [docs/workspaces/](workspaces/) |
| Infrastructure overview | [docs/infrastructure/README.md](infrastructure/README.md) |

For the `rfdf` platform itself, see [USER-GUIDE.md](USER-GUIDE.md).

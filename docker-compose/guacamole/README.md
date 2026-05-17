# Apache Guacamole

Guacamole is the HTML5 remote-desktop gateway. Kasm + Wine handle ~90% of
"Windows" RF tools; Guacamole covers the rest by serving the **Win11 VM**
(`rfdf-winrf`) — SDR#, HDSDR, and any vendor-locked Windows-only tool — as a
browser desktop, with audio passthrough for listening tests.

## Stack

| Service     | Image                       | Role                          |
|-------------|-----------------------------|-------------------------------|
| `postgres`  | `postgres:15-alpine`        | Connection + user database.   |
| `guacd`     | `guacamole/guacd:1.5.5`     | The protocol (RDP/VNC) daemon.|
| `guacamole` | `guacamole/guacamole:1.5.5` | The web app (port 8080).      |

## Deploy

```sh
cp .env.example .env                           # fill GUACAMOLE_DB_PASSWORD
docker run --rm guacamole/guacamole:1.5.5 \
    /opt/guacamole/bin/initdb.sh --postgresql > postgres/initdb/001-schema.sql
docker compose up -d
```

The `guacamole` Ansible role does all of the above on `rfdf-guac`.

## Connecting to the Win11 VM

Once Guacamole is up, log in (default `guacadmin` / `guacadmin` — change it
immediately) and add an **RDP connection** to `rfdf-winrf` at its static IP.
Enable audio so SDR listening tests work. The Win11 VM is created as a shell by
the `proxmox_vm` role; Windows is installed by hand — see
`docs/infrastructure/proxmox-setup.md`.

No dependency on the ai-orchestrator.

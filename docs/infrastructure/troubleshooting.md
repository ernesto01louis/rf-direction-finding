# Troubleshooting

## Provisioning

**`make provision` fails at the Proxmox API step** — check the API token
(`pveum user token list root@pam`), `proxmox_api_host`, and that the control
node can reach `:8006`. `proxmox_api_validate_certs: false` is set for the
self-signed homelab cert.

**LXC created but SSH times out** — the `proxmox_lxc` role injects the control
node's public key (`proxmox_lxc_pubkey`). Confirm `~/.ssh/id_ed25519.pub`
exists, or override the var.

**Vault decryption error** — `group_vars/all/vault.yml` must be
ansible-vault-encrypted and `ansible/.vault-pass` must contain the password.
`make` exports `ANSIBLE_VAULT_PASSWORD_FILE`.

**A playbook half-failed** — every playbook is idempotent; just re-run it.
There are no silent `failed_when: false` guards, so the failing task is the
real cause.

## Services

**`api.rf.lan` returns 502** — expected. The rfdf REST API (`src/rfdf/api/`)
is a **Stage-7 deliverable**; the route and the `rfdf-api` systemd unit ship
provisioned-but-inert. The Homepage live-RF widgets likewise show "waiting for
platform API" until Stage 7.

**Browser TLS warning on `*.rf.lan`** — Traefik serves its self-signed default
certificate. For browser-trusted TLS, wire ACME or step-ca (see
`docs/infrastructure/tailscale.md`).

**Authelia redirect loop** — usually a cookie-domain mismatch. The
`session.cookies` domain in `configuration.yml` must match the base domain
(`rf.lan`).

**TOTP rejected** — host clock skew. Every LXC runs `systemd-timesyncd`;
confirm NTP sync (`timedatectl`).

## Port conflicts

JupyterLab runs on **8889**, not the usual 8888 — **8888 is taken by Hindsight
on `192.168.2.203`**. code-server is on 8443.

## Kasm

**Re-running `04-kasm.yml` does nothing** — by design. The `kasm` role runs the
official installer only when `/opt/kasm/current` is absent; Kasm CE is not
re-install-idempotent.

## SDRangel missing

SDRangel has no Debian package and is the one catalogue tool not auto-installed
— build it from source on `rfdf-tools` per the upstream README.

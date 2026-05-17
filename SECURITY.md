# SECURITY

## 1. Threat model

`rfdf` is a research platform that runs in trusted environments (operator's homelab,
lab workstation, rented cloud GPU). It is **NOT** hardened for multi-tenant or hostile
network operation. Specifically:

**The platform protects against:**
- Accidental disclosure of operator secrets via committed `.env` files (`.gitignore` +
  `detect-private-key` pre-commit hook).
- Silent regulatory non-compliance via the configurable EIRP cap (§6).
- Untrusted Python dependency installation via pinned versions and Dependabot security
  advisories.

**The platform does NOT protect against:**
- A compromised or malicious operator with shell access to the host.
- A network-level attacker on the LAN with the `rfdf` REST API bound to `0.0.0.0`.
  Bind to `127.0.0.1` (default) and front via Tailscale + Traefik + Authelia per Stage 6.
- Side-channel attacks against running SDR captures (Tempest-style emanation analysis).
- Supply-chain compromise of `numpy` / `scipy` / `torch` (we follow upstream advisories;
  no independent vetting).

## 2. Secret handling

**Decision: `.env` files with `chmod 600`, gitignored.** SOPS / age / Vault-style
secret-at-rest encryption is **deferred** until a non-solo contributor joins.

Rationale: solo-operator environment, `.env` lives only on the operator's encrypted
TrueNAS-backed Proxmox host. Adding SOPS for one operator is process overhead without
proportional security gain. The decision is **documented once** in
`docs/operational-decisions.md` (audit-lesson: do not carry the same decision in
multiple places).

When a second contributor joins, the decision is reopened: SOPS-with-age-keys is the
expected upgrade path.

## 3. USB device permissions

SDRs require `/dev/bus/usb/*` access. The naive workaround (run everything as root) is
unacceptable. The platform ships a `udev` rules generator (Stage 5,
`rfdf hw udev install`) that writes `/etc/udev/rules.d/70-rfdf.rules` granting
`plugdev` group access to known SDR devices (B210, RTL-SDR, HackRF, Pluto, KrakenSDR).

The operator's user must be a member of `plugdev`. The installer prompts before invoking
`sudo udevadm control --reload-rules`.

## 4. Network exposure

The optional REST API (`[api]` extra) binds to `127.0.0.1:8000` by default. To expose to
the LAN safely:

1. Front with Traefik + Authelia (Stage 6 Ansible role).
2. Restrict to the Tailnet via Tailscale ACLs.
3. **Never** bind directly to `0.0.0.0` without auth in front.

The platform's WebSocket path inherits the same model.

## 5. Orchestrator integration trust model

When used as an `ai-orchestrator` consumer, the orchestrator's per-target SSH key
governs what the orchestrator can execute on the `rfdf` host. **Use per-target keys**,
not a shared identity key (audit-lesson from the orchestrator: shared SSH credentials
were flagged for replacement).

The orchestrator's sudo allowlist (`tools._TOOL_CMD_BLOCKLIST` in `ai-orchestrator`)
applies on the consumer LXC. Document the inherited risk in your operational runbook.

## 6. EIRP enforcement

The platform enforces a configurable maximum **Effective Isotropic Radiated Power**
(EIRP) cap on any code path that initiates TX (Stage 5 onward). Default:

```toml
[eirp]
max_eirp_dbm = 14       # 25 mW, EU SRD general limit
override_explicit = false  # operator must set true to exceed cap
```

`override_explicit = true` requires the operator to acknowledge they hold the relevant
amateur licence (Klasse-A in DE for >25 mW outdoor 5.8 GHz) and have completed any
required BEMFV Standortbescheinigung. **The platform does not certify legal compliance;
it provides guardrails.**

Receive-only operation is unregulated in DE / EU. The §89 TKG anonymization requirement
for received non-public transmissions is the operator's responsibility.

## 7. Reporting vulnerabilities

For security issues, do **not** open a public GitHub issue.

- Email: `louis_ernesto@aol.com`
- Subject prefix: `[rfdf SECURITY]`
- GPG: fingerprint placeholder (key to be published before v0.1.0 GA)

We will acknowledge receipt within 7 days. Responsible disclosure window: 90 days from
first contact unless an active exploit is in the wild.

## 8. Operational controls

- **Branch protection on `main`:** currently deferred per operator decision; revisit
  before Stage 5 (real-hardware integration). Documented in
  [docs/operational-decisions.md](docs/operational-decisions.md).
- **CI must be green** before merge (informal during the deferral; enforced by branch
  protection once enabled).
- **Pre-commit hooks** include `detect-private-key` to catch accidental secret commits.
- **Dependabot** is configured for security advisories only (Stage 1).

## 9. Infrastructure secrets (ansible-vault)

The Stage 6 ecosystem deployment (`ansible/`) handles its own secrets —
Proxmox API token, Authelia keys, Grafana/Kasm/Guacamole/OpenWebRX passwords,
the Tailscale auth key. **None of these are committed in plaintext.**

**Decision: `ansible-vault` (not SOPS).** The orchestrator-consumer platform
secrets (§2) stay as `.env`; the infrastructure-as-code secrets use
`ansible-vault` because Ansible decrypts vaulted vars transparently at run
time. Two encryption schemes for two distinct concerns is acceptable; each is
documented in exactly one place (§2 here, this section, and
`docs/operational-decisions.md`).

- The real secrets live in `ansible/group_vars/all/vault.yml`, encrypted with
  `ansible-vault` and **git-ignored**. `ansible/vault/secrets.example.yml` is
  the committed plaintext **template** (placeholder values only — the
  `detect-private-key` hook would block a real key).
- The vault password lives in `ansible/.vault-pass` (git-ignored, `chmod 600`).
  `make provision` exports `ANSIBLE_VAULT_PASSWORD_FILE`; the password file is
  never committed and never leaves the operator's host.
- Rotating a secret: edit `vault.yml` via `ansible-vault edit`, re-run the
  relevant playbook. Rotating the vault password: `ansible-vault rekey`.

When a second contributor joins, the vault password is shared out-of-band (not
via the repo); a shared-key upgrade to `ansible-vault` with multiple vault IDs
is the expected path.

## 10. Tailscale Funnel

Tailscale Funnel can expose an ecosystem service to the **public internet**.
It is **disabled by default** and must stay that way unless a specific service
has been threat-modelled for public exposure.

Before enabling Funnel for any service:

1. Confirm the service is behind **Authelia** (every Traefik route is, except
   the auth portal — which is itself hardened by `regulation` lockout rules).
2. Prefer **2FA-required** access policies for any publicly-reachable route.
3. Funnel only supports a fixed set of ports (443/8443/10000) and logs every
   connection — review the Funnel access logs periodically.
4. Never Funnel the Traefik dashboard, the Kasm/Guacamole admin UIs, or
   Prometheus — these are management surfaces and stay LAN/tailnet-only.
5. Receive-only RF data may still be subject to §89 TKG (§6) — do not expose
   raw capture browsers publicly without anonymisation.

The recommended off-LAN path remains the **Tailnet itself** (member-only
access), not Funnel.

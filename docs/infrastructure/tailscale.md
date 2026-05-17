# Tailscale

Tailscale is the **recommended** off-LAN access path — but it is **optional**.
The playbooks are fully functional on plain LAN; `tailscale_enabled: false` is
the default.

## Enabling

In `group_vars/all/main.yml` set `tailscale_enabled: true`, and put a
`tailscale_auth_key` (a reusable or ephemeral key from the Tailscale admin
console) in `group_vars/all/vault.yml`. The `tailscale` role then joins every
host to the tailnet on the next `make provision`.

Each service is then reachable two ways:

1. **LAN** — `kasm.rf.lan` → `192.168.2.242` (Pi-hole / `/etc/hosts`).
2. **Tailnet** — `rfdf-kasm.<tailnet>.ts.net` → the Tailscale IP.

## MagicDNS preset (no own domain)

An operator without their own DNS can route via Tailscale MagicDNS instead of
`*.rf.lan`. Point Traefik routers at the MagicDNS hostnames and use
Tailscale-managed certificates (`tailscale cert`) for browser-trusted TLS.

## ACL example

Restrict the ecosystem to the operator's own devices:

```jsonc
{
  "acls": [
    { "action": "accept", "src": ["autogroup:member"],
      "dst": ["tag:rfdf:*"] }
  ],
  "tagOwners": { "tag:rfdf": ["autogroup:admin"] }
}
```

Tag the hosts (`tailscale up --advertise-tags=tag:rfdf`) and only tailnet
members reach them.

## Funnel

Tailscale Funnel can expose a service to the public internet. **It is off by
default** — exposing a service publicly is a deliberate, per-service decision
that must be threat-modelled first. See `SECURITY.md` "Tailscale Funnel" before
enabling it for anything.

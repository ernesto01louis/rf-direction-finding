# Authentication — Authelia SSO

Authelia gives the whole ecosystem one login. Traefik calls it on every
request (ForwardAuth middleware); it is *also* configured as an OIDC provider
for services that speak OIDC natively.

## First login (no 2FA yet)

2FA is **not** required for the very first login — bootstrap would be
impossible otherwise. The flow:

1. Browse to any service, e.g. `https://kasm.rf.lan`.
2. Authelia redirects to `https://auth.rf.lan`.
3. Log in with the operator username + password (the argon2 hash in
   `group_vars/all/vault.yml` &rarr; `authelia_admin_password_hash`).
4. Authelia prompts to enrol a **TOTP** device — scan the QR with Aegis /
   Authy / Google Authenticator. Recovery codes are shown **once** — store
   them safely.
5. Subsequent logins require password + TOTP.

## Worked auth flow

A browser opening `https://kasm.rf.lan` and ending up in GNU Radio:

| Step | What happens | Timing | Failure mode |
|------|--------------|--------|--------------|
| 1 | Browser → `kasm.rf.lan` (LAN DNS / Tailscale MagicDNS) | — | DNS unresolved → name error |
| 2 | Traefik matches `Host(\`kasm.rf.lan\`)`, applies the `authelia` middleware | 5–20 ms | no route → Traefik 404 |
| 3 | No session cookie → Authelia 302 → `auth.rf.lan` | 50–150 ms | Authelia down → 502 |
| 4 | Operator authenticates (password; TOTP after enrolment) | 1–5 s | clock skew breaks TOTP → fix NTP |
| 5 | Authelia sets the session cookie, 302 back to `kasm.rf.lan` | 50–150 ms | cookie-domain mismatch → redirect loop |
| 6 | Traefik re-evaluates ForwardAuth — authorized — proxies to Kasm | 5–20 ms | Kasm down → 502 |
| 7 | Kasm session-selection UI → open `kasm-ubuntu-rftools` → launch GRC | seconds | — |

## OIDC provider

`identity_providers.oidc` in `docker-compose/authelia/configuration.yml` makes
Authelia a native OIDC provider (HMAC secret + JWKS key from the vault). The
`clients` list is empty by default; native-OIDC clients are registered as
services adopt them. ForwardAuth already gates every service regardless.

## Scaling beyond one user

`users_database.yml` holds the user list — a homelab needs one. For many users
or LDAP, Authelia supports an `ldap` authentication backend; see the upstream
Authelia docs. (LDAP is a documented pointer here, not implemented.)

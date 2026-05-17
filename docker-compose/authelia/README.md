# Authelia — single sign-on

Authelia provides one login for the whole rfdf ecosystem. Traefik calls it on
every request (ForwardAuth); it is *also* configured as an OIDC provider for
services that speak OIDC natively.

## Layout

| File                          | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `compose.yml`                 | Authelia + Redis (session store).                |
| `configuration.yml`           | Main config — secret-free.                       |
| `users_database.example.yml`  | Template for the user DB.                        |
| `.env.example`                | Template for the secrets `.env`.                 |
| `secrets/`                    | OIDC private key (gitignored).                   |
| `data/`, `redis-data/`        | Runtime state (gitignored).                      |

## Standalone use

```sh
cp .env.example .env                       # fill every REPLACE
cp users_database.example.yml users_database.yml
mkdir -p secrets && openssl genrsa -out secrets/oidc.private.pem 4096
docker compose up -d
```

No dependency on the ai-orchestrator.

## First login (no 2FA yet)

2FA is **not** required for the very first login — Authelia lets the operator
authenticate with the password alone, then enrol a TOTP device (Aegis / Authy /
Google Authenticator). Recovery codes are shown once at enrolment. Full
walkthrough: `docs/infrastructure/authentication.md`.

## Notifications

Ships with the **filesystem notifier** — password-reset / 2FA mails are written
to `data/notification.txt`:

```sh
docker compose exec authelia cat /data/notification.txt
```

For real e-mail, set the `AUTHELIA_NOTIFIER_SMTP_*` values in `.env` and swap
`notifier.filesystem` for `notifier.smtp` in `configuration.yml`.

## OIDC clients

`identity_providers.oidc.clients` is empty in PR 1. Native-OIDC clients (e.g.
Grafana) are appended as those services are deployed. ForwardAuth already gates
every service regardless.

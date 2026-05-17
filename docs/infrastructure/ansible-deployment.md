# Running the playbooks

## One-time setup

```sh
cd ansible
cp inventory.example.yml inventory.yml                  # edit for your network
cp vault/secrets.example.yml group_vars/all/vault.yml   # fill in real secrets
ansible-vault encrypt group_vars/all/vault.yml
echo "<vault-password>" > .vault-pass                   # git-ignored, chmod 600
```

`group_vars/all/vault.yml` is auto-loaded for every host; `.vault-pass`
unlocks it (`make provision` exports `ANSIBLE_VAULT_PASSWORD_FILE`).

## Provision

```sh
make provision              # runs playbooks 00–08 in order
make infra-verify           # runs 99-verify end-to-end smoke tests
```

Or a slice:

```sh
make provision-foundation   # 00-bootstrap + 01-platform + 06-dashboard
cd ansible && ansible-playbook playbooks/04-kasm.yml
```

## Playbook order

`make provision` runs them numerically. The dependency chain:

1. **00-bootstrap** — creates the LXC/VM tree, base OS, Tailscale.
2. **06-dashboard** — Docker, **Authelia**, **Traefik** (load-bearing — every
   other service is reached through Traefik), then Homepage + monitoring.
3. **01/02/03/04/05/07/08** — the individual services, each idempotent and
   isolatable (runnable alone once its prerequisites exist).
4. **99-verify** — asserts the whole deployment is healthy.

> Run **06-dashboard before the service playbooks** if you deploy out of order
> — Traefik + Authelia must be up for the SSO flow to work.

## Idempotency

Every playbook reconciles to the same state on re-run. There are no blanket
`failed_when: false` guards: a task that can fail does so loudly. A half-failed
run is safe to re-run.

## Customising the IP plan

Edit `inventory.yml` (addresses + VMIDs) and `group_vars/proxmox.yml` (sizing,
storage, bridge). Nothing in `roles/` needs to change.

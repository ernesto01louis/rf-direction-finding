# rfdf ecosystem — Ansible deployment

Ansible provisions the Stage 6 software ecosystem: the LXC/VM tree on a Proxmox
host, plus every service (Traefik, Authelia, Kasm, Guacamole, OpenWebRX+,
JupyterLab, Homepage, monitoring) configured for its role.

The platform itself (`pip install rfdf`) has **no dependency** on any of this —
the ecosystem is operator quality-of-life, hosted *alongside* the platform.

## Quick start

```sh
# 1. Install collections + copy the templates
cp inventory.example.yml inventory.yml          # edit IPs/VMIDs for your network
cp vault/secrets.example.yml vault/secrets.yml  # fill in real secrets
ansible-vault encrypt vault/secrets.yml
echo "<vault-password>" > .vault-pass           # git-ignored

# 2. Provision everything (from the repo root)
make provision

# 3. Verify
make infra-verify
```

Individual playbooks are runnable on their own for incremental updates:

```sh
cd ansible && ansible-playbook playbooks/06-dashboard.yml
```

## Layout

```
ansible/
  ansible.cfg              # inventory, roles, vault-password-file
  inventory.example.yml    # the host tree — copy to inventory.yml
  requirements.yml         # Galaxy collections
  group_vars/              # all.yml / proxmox.yml / docker.yml
  vault/secrets.example.yml# template for the encrypted vault
  playbooks/
    00-bootstrap.yml       # create the LXC/VM tree + base OS config
    01-platform.yml        # install the rfdf platform
    06-dashboard.yml       # Traefik + Authelia (+ Homepage/monitoring later)
    ...                    # 02-08 added across the Stage-6 sub-sessions
    99-verify.yml          # end-to-end smoke tests
  roles/                   # one role per concern
```

## Principles

- **Idempotent** — re-running any playbook reconciles to the same state.
- **Isolatable** — any playbook runs on its own if its prerequisites are met.
- **Parameterized** — every IP, VMID, and size is in `inventory.yml` /
  `group_vars/`; an operator with different infrastructure edits those, never
  the roles.
- **No silent failures** — there are no blanket `failed_when: false` guards.
- **Secrets in vault** — `vault/secrets.yml` is `ansible-vault`-encrypted and
  git-ignored. See `SECURITY.md` for the vault-password policy.
- **Tailscale optional** — `tailscale_enabled: false` by default; the
  playbooks are fully functional on plain LAN.

Full operator documentation: `docs/infrastructure/`.

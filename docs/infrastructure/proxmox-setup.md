# Proxmox host setup

The Ansible playbooks target a Proxmox VE host (the reference; the patterns
work on any LXC/Docker-capable Linux host). Tested against **Proxmox VE 8.x**.

## Prerequisites

1. **API token** — the playbooks talk to the Proxmox API, not `pct`/`qm`:
   ```sh
   pveum user token add root@pam rfdf-ansible --privsep 0
   ```
   Put the secret in `group_vars/all/vault.yml` as `proxmox_api_token_secret`.
2. **Debian 12 LXC template** on a storage pool:
   ```sh
   pveam update && pveam download local debian-12-standard_12.7-1_amd64.tar.zst
   ```
3. **Storage + bridge** — set `proxmox_storage`, `proxmox_template_storage`,
   `proxmox_bridge` (and `proxmox_vlan_tag` if VLAN'd) in
   `group_vars/proxmox.yml`.
4. **SSH key** — the control node's public key is injected into every LXC
   (`proxmox_lxc_pubkey`, default `~/.ssh/id_ed25519.pub`).

## USB passthrough (rfdf-daq, rfdf-openwebrx)

These hosts are **privileged LXCs** with USB device passthrough — the
`proxmox_lxc` role appends the device cgroup + a `/dev/bus/usb` bind mount to
the container config. This is the simpler path for SDRs; an operator who
prefers full USB-controller passthrough can instead make `rfdf-daq` a VM and
pass through a controller (not automated).

## Win11 VM (rfdf-winrf)

Proxmox has no reliable unattended-Windows install, so the `proxmox_vm` role
creates the **VM shell only** (q35 + OVMF + TPM 2.0, Windows-11 + VirtIO ISOs
attached). Then, once:

1. Open the Proxmox console for `rfdf-winrf`.
2. Install Windows 11; when the installer finds no disk, **Load driver** from
   the second CD-ROM (VirtIO `vioscsi`), and likewise the `NetKVM` NIC driver.
3. Set a static IP of `192.168.2.244`, install the QEMU guest agent.
4. Install the Windows RF tools (SDR#, HDSDR, …) + the VirtualHere USB client.
5. **Snapshot** the VM as `clean-install` — reuse it by cloning/rolling back.

**GPU passthrough is intentionally OFF** for Stage 6 — the GPU is reserved for
ML training. A tool that needs GPU acceleration can get a separate passthrough
later (`proxmox_vm_gpu_passthrough`).

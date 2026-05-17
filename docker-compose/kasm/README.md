# Kasm Workspaces

Kasm Workspaces Community Edition streams containerized Linux desktops to the
browser. It is the access layer for every GUI RF tool in the ecosystem.

## Why no `compose.yml` here

Kasm CE installs through its **own official installer**, which stands up its
multi-container stack (api / manager / agent / db / redis / proxy). A
hand-written Compose file cannot reproduce that faithfully, so the `kasm`
Ansible role drives the official installer instead (guarded to run once). This
directory holds the custom **workspace images**, not a service stack.

## Custom workspace images

Built from `dockerfiles/` and published to GHCR by the `kasm-images` CI
workflow on every Stage-6+ tag:

| Image                          | Base                        | Contents |
|--------------------------------|-----------------------------|----------|
| `rfdf-kasm-ubuntu-rftools`     | `kasmweb/core-ubuntu-jammy` | RF GUI apps (GNU Radio, GQRX, inspectrum, ...); `/opt/rftools/` NFS-mounted for the venv + git tools |
| `rfdf-kasm-ubuntu-wine-antennas` | `kasmweb/core-ubuntu-jammy` | Wine 9 + MMANA-GAL Pro + 4nec2 + xnec2c |
| `rfdf-kasm-kali-rf`            | `kasmweb/core-kali-rolling` | Kali + RF / wireless-security tools |
| `rfdf-kasm-jupyter-rfdf`       | `kasmweb/core-ubuntu-jammy` | JupyterLab + `rfdf[ml]` |

After Kasm is up, register each image in **Admin → Workspaces** by name. Full
per-workspace usage: `docs/workspaces/`.

## MMANA-GAL Pro / 4nec2

These freeware Windows tools are **not committed** (redistribution terms are
unclear). The `kasm-ubuntu-wine-antennas` image pulls them at build time from
official URLs passed as `--build-arg MMANA_GAL_URL=... NEC2_URL=...`. With the
args empty the image still builds — install the tools by hand in the workspace
afterwards. See `docs/workspaces/kasm-ubuntu-wine-antennas.md`.

## Standalone

Kasm has **no dependency on the ai-orchestrator**. Build an image directly:

```sh
docker build -t rfdf-kasm-ubuntu-rftools docker-compose/kasm/dockerfiles/kasm-ubuntu-rftools
```

# Workspace — kasm-ubuntu-wine-antennas

The antenna-design workspace: Ubuntu + Wine with the Windows antenna tools,
plus the Linux-native xnec2c.

## Tools

- **MMANA-GAL Pro** + **4nec2** — under Wine 9 (prefix at `/opt/wineprefix`,
  pre-seeded with corefonts + vcrun2019).
- **xnec2c** — Linux-native NEC2 modelling.

## MMANA-GAL Pro / 4nec2 are not committed

Both are freeware but their redistribution terms are unclear, so the binaries
are **not** in the repo. The image's Dockerfile pulls them at build time from
official URLs passed as build-args:

```sh
docker build \
  --build-arg MMANA_GAL_URL="https://hamsoft.ca/.../mmana-gal.exe" \
  --build-arg NEC2_URL="https://www.qsl.net/4nec2/4nec2.exe" \
  -t rfdf-kasm-ubuntu-wine-antennas \
  docker-compose/kasm/dockerfiles/kasm-ubuntu-wine-antennas
```

With the args empty the image still builds — install the tools by hand in the
running workspace afterwards (`wine /path/to/installer.exe`).

**If the official URLs rot:** download the installers from the current vendor
page, host them somewhere reachable, and pass the new URLs as build-args. The
repo never needs to change.

## Common workflows

- Design a Yagi/LPDA in MMANA-GAL Pro, export the geometry.
- Cross-check the pattern in xnec2c (Linux-native, no Wine).

## Upstream docs

MMANA-GAL <https://hamsoft.ca/pages/mmana-gal.php> ·
4nec2 <https://www.qsl.net/4nec2/> ·
xnec2c <https://www.xnec2c.org>

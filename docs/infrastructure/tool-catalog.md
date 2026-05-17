# Tool catalogue — what is installed where

The operator chose to install the **full catalogue by default** (no
`install_extras` gate). Everything below is provisioned automatically.

## rfdf-tools (`/opt/rftools`, NFS-exported to Kasm)

**apt** — GNU Radio (+ `gr-osmosdr`), GQRX, CubicSDR, `rtl-sdr`, HackRF,
`multimon-ng`, inspectrum, URH, Wireshark / tshark, Kismet, xnec2c / nec2c,
gpredict, `dump1090-mutability`, `rtl_433`, SatDump (official `.deb`), sox.

**`/opt/rftools/venv`** — `rfdf[ml,antenna,api]`, torchsig, sigmf, scikit-rf,
pynec, necpp, pyargus, pyadi-iio, nanovna-saver.

**`/opt/rftools/source`** (git) — `krakensdr_doa`, `gr-lora_sdr`, `gr-iridium`,
`AIS-catcher`, `rtl_433`.

**Manual** — **SDRangel** has no Debian package; build it from source per
the upstream README if needed. It is the one catalogue tool not automated.

## Kasm workspace images

| Image                          | Tools |
|--------------------------------|-------|
| `kasm-ubuntu-rftools`          | GUI RF apps; mounts `/opt/rftools/` over NFS |
| `kasm-ubuntu-wine-antennas`    | Wine + MMANA-GAL Pro + 4nec2 + xnec2c |
| `kasm-kali-rf`                 | Kali + RF / wireless-security tools |
| `kasm-jupyter-rfdf`            | JupyterLab + `rfdf[ml]` |

## rfdf-daq

UHD (`uhd-host`, FPGA images), `rfdf[sdr-uhd]` — the B210 coherent-capture
host.

## rfdf-openwebrx

`jketterl/openwebrx-full` — 24/7 web SDR, FT8/FT4/WSPR/JS8 decoders.

## rfdf-jupyter

code-server (`code.rf.lan`) + JupyterLab (`jupyter.rf.lan`), both with
`rfdf[ml]` pre-installed.

## rfdf-winrf (manual)

SDR#, HDSDR, and any vendor-locked Windows-only RF tool — installed by hand
in the Win11 VM, reached via Guacamole.

## Optional Docker decoders

`readsb` + `tar1090` (ADS-B), `acarsdec` (ACARS), `dsd-fme` (digital voice)
need an attached SDR and are documented here as optional add-ons rather than
auto-installed on the SDR-less toolchain host.

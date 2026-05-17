# /opt/rftools — Linux-native RF toolchain

Installed by the `rf_tools` Ansible role on the `rfdf-tools` host. This tree is
NFS-exported (read-only) and mounted into the `kasm-ubuntu-rftools` workspace,
so the tools are reachable from a browser via Kasm.

## Layout

| Path             | Contents                                                   |
|------------------|------------------------------------------------------------|
| `venv/`          | Python toolchain — `rfdf`, torchsig, scikit-rf, pyargus, … |
| `source/`        | git-cloned reference trees (krakensdr_doa, gr-lora_sdr, …) |

## apt tools

GNU Radio (+ `gr-osmosdr`), GQRX, CubicSDR, `rtl-sdr`, HackRF, `multimon-ng`,
inspectrum, URH, Wireshark / tshark, Kismet, xnec2c / nec2c, gpredict,
`dump1090-mutability`, `rtl_433`, SatDump. Launch any of them from the Kasm
`ubuntu-rftools` desktop.

## venv tools

```sh
source /opt/rftools/venv/bin/activate
rfdf doa --help
```

## Known manual step — SDRangel

SDRangel has no Debian package; build it from source per the upstream
instructions (https://github.com/f4exb/sdrangel) if you need it. Every other
catalogue tool installs automatically.

Full per-tool reference: `docs/infrastructure/tool-catalog.md` in the repo.

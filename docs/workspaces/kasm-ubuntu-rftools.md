# Workspace — kasm-ubuntu-rftools

Base Ubuntu desktop with the Linux-native RF GUI toolchain. The general-purpose
signal-analysis and decoding workspace.

## Tools

GUI apps baked into the image: GNU Radio Companion, GQRX, CubicSDR,
inspectrum, URH, Wireshark, Kismet, xnec2c, gpredict.

The heavier / Python / git-cloned tools come from **`/opt/rftools/`**, mounted
read-only over NFS from the `rfdf-tools` host:

- `/opt/rftools/venv` — `rfdf`, torchsig, scikit-rf, pyargus, …
- `/opt/rftools/source` — `krakensdr_doa`, `gr-lora_sdr`, `gr-iridium`, …

## Common workflows

- **Flowgraph development** — launch GNU Radio Companion from the desktop.
- **Capture analysis** — open a `.sigmf` recording in inspectrum or URH.
- **rfdf CLI** — `source /opt/rftools/venv/bin/activate && rfdf doa --help`.

## Upstream docs

GNU Radio <https://wiki.gnuradio.org> · URH <https://github.com/jopohl/urh> ·
inspectrum <https://github.com/miek/inspectrum>

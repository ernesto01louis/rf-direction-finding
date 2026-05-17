# OpenWebRX+ — always-on web SDR

A 24/7 shared web SDR. Anyone on the LAN / tailnet opens a browser and watches
the waterfall across HF/VHF/UHF, with FT8 / FT4 / WSPR / JS8 decoders.

It gets its **own LXC** because it owns a dedicated USB SDR full-time — the
platform's B210 capture rig (`rfdf-daq`) is shared and intermittent.

## Deploy

```sh
cp .env.example .env            # set OPENWEBRX_ADMIN_PASSWORD
docker compose up -d
```

The `openwebrx` Ansible role deploys this on `rfdf-openwebrx`, whose LXC has
USB passthrough enabled so the container sees the SDR at `/dev/bus/usb`.

`settings.example.json` is a starting receiver profile — copy it to
`settings/settings.json` or configure live in the admin UI at
`http://<host>:8073/admin`.

No dependency on the ai-orchestrator.

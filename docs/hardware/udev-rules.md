# udev rules — letting rfdf open your SDR

## The problem

When you plug in a USB SDR, Linux creates a device node owned by `root`. A
non-root process — `rfdf`, `rtl_test`, `uhd_usrp_probe` — cannot open it, and
the device "doesn't show up". This is the single most common first-experience
failure in the entire SDR ecosystem.

## The fix

A **udev rule** tells the kernel to grant broader permissions to a specific
USB device the moment it appears. `rfdf` ships a generator + installer:

```sh
rfdf hw udev list        # show the rules rfdf would install
rfdf hw udev generate    # print the rules-file content to stdout
rfdf hw udev install     # write /etc/udev/rules.d/70-rfdf.rules + reload
```

`install` requires root and prompts before invoking `sudo`; re-plug the device
(or it is re-triggered automatically) and it is openable without root.

## What a rule looks like

```
SUBSYSTEM=="usb", ATTRS{idVendor}=="2500", ATTRS{idProduct}=="0020", MODE="0666", SYMLINK+="b210_%n"  # Ettus USRP B210
```

| Field | Meaning |
|---|---|
| `SUBSYSTEM=="usb"` | match USB devices only |
| `ATTRS{idVendor}` / `ATTRS{idProduct}` | the device's USB vendor/product ID (`lsusb`) |
| `MODE="0666"` | world read/write on the device node |
| `SYMLINK+="b210_%n"` | a stable `/dev/b210_0` name (`%n` = kernel number) |

## Devices rfdf knows

`KNOWN_DEVICES` in `rfdf.hw.udev` ships rules for the Ettus USRP B210 / B200 /
B200mini, the RTL-SDR (RTL2832U + R820T2), the HackRF One, and the ADALM-PLUTO.

## Adding a device

1. Plug it in, run `lsusb`, note the `idVendor:idProduct`.
2. Append a `UdevRule` to `KNOWN_DEVICES` in `src/rfdf/hw/udev.py`.
3. `rfdf hw udev install` again.

A `UdevRule` validates that the IDs are 4 hex digits and the mode is octal, so a
typo is caught at construction rather than producing a silently dead rule.

## Security note

`MODE="0666"` grants every local user read/write on the device. On a
multi-user host, prefer `MODE="0660"` plus `GROUP="plugdev"` and add trusted
users to `plugdev`. The default `0666` suits a single-operator research box.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Device still not openable | re-plug it, or `sudo udevadm trigger`; confirm the IDs match `lsusb` |
| `PermissionError` on install | re-run `rfdf hw udev install` with `sudo` |
| `udevadm` not found | install `udev` / `systemd-udev`; the rules file is still written |

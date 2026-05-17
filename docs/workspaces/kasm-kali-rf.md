# Workspace — kasm-kali-rf

Kali Linux + RF tools, for wireless-security research workflows.

## Tools

GNU Radio (+ `gr-osmosdr`), GQRX, `rtl-sdr`, HackRF, `multimon-ng`,
inspectrum, Kismet, aircrack-ng, Wireshark, sox — on the Kali rolling base,
which also brings Kali's wider security toolset.

## Common workflows

- Wireless survey + capture with Kismet.
- 802.11 analysis with aircrack-ng + Wireshark.
- SDR-side investigation with GNU Radio / GQRX / inspectrum.

## Scope reminder

Receive-only RF research is unregulated in DE/EU; transmitting is not. The rfdf
platform enforces a configurable EIRP cap and these tools do not change that —
keep operations RX-only unless you hold the appropriate licence. See
`SECURITY.md` and the project brief's regulatory section.

## Upstream docs

Kali <https://www.kali.org/docs/> · Kismet <https://www.kismetwireless.net> ·
aircrack-ng <https://www.aircrack-ng.org>

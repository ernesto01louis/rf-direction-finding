# kasm-kali-rf — Kali Linux + RF tools, for wireless-security research.
FROM kasmweb/core-kali-rolling:1.16.1

USER root
ENV DEBIAN_FRONTEND=noninteractive

# RF + wireless-security toolchain from the Kali repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnuradio \
        gr-osmosdr \
        gqrx-sdr \
        rtl-sdr \
        hackrf \
        multimon-ng \
        inspectrum \
        kismet \
        aircrack-ng \
        wireshark \
        sox \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R 1000:0 "${HOME}"
ENV HOME=/home/kasm-user
WORKDIR /home/kasm-user
USER 1000

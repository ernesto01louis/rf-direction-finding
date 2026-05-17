# kasm-ubuntu-rftools — base Ubuntu desktop with the Linux-native RF GUI tools.
#
# Heavy / venv / git-cloned tools (rfdf, torchsig, krakensdr_doa, ...) are NOT
# baked in — they come from /opt/rftools/ on the rfdf-tools LXC, NFS-mounted
# into the workspace by the Kasm volume mapping. This image carries the GUI
# apps that need a desktop session.
FROM kasmweb/core-ubuntu-jammy:1.16.1

USER root
ENV DEBIAN_FRONTEND=noninteractive

# RF GUI toolchain available in the Ubuntu 22.04 repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnuradio \
        gnuradio-dev \
        gr-osmosdr \
        gqrx-sdr \
        cubicsdr \
        rtl-sdr \
        librtlsdr-dev \
        hackrf \
        libhackrf-dev \
        multimon-ng \
        inspectrum \
        wireshark \
        tshark \
        kismet \
        xnec2c \
        nec2c \
        gpredict \
        dump1090-mutability \
        nanovna-saver \
        pipx \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# URH (Universal Radio Hacker) — installed from PyPI via pipx.
RUN PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install urh

# Desktop shortcuts for the most-used apps.
COPY ./shortcuts/ /usr/share/applications/

RUN chown -R 1000:0 "${HOME}"
ENV HOME=/home/kasm-user
WORKDIR /home/kasm-user
USER 1000

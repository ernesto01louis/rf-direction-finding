# kasm-ubuntu-rftools — base Ubuntu desktop with the Linux-native RF GUI tools.
#
# Heavy / venv / git-cloned tools (rfdf, torchsig, krakensdr_doa, ...) are NOT
# baked in — they come from /opt/rftools/ on the rfdf-tools LXC, NFS-mounted
# into the workspace by the Kasm volume mapping. This image carries the GUI
# apps that need a desktop session.
FROM kasmweb/core-ubuntu-jammy:1.16.1

USER root
ENV DEBIAN_FRONTEND=noninteractive
# pipefail so a failed curl in the Kismet-key pipe aborts the build (DL4006).
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The kasm core-ubuntu-jammy base ships a trimmed sources.list — enable the
# `universe` component, then add Kismet's own apt repo (Kismet is no longer in
# the Ubuntu archive — upstream ships its own repo per release codename).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        gnupg \
    && add-apt-repository -y universe \
    && curl -fsSL https://www.kismetwireless.net/repos/kismet-release.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/kismet-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kismet-archive-keyring.gpg] https://www.kismetwireless.net/repos/apt/release/jammy jammy main" \
        > /etc/apt/sources.list.d/kismet.list \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# RF GUI toolchain. nanovna-saver is NOT an apt package — it is pip-installed
# in the rftools venv (NFS-mounted at /opt/rftools), so it is not listed here.
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

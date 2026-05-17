# kasm-ubuntu-wine-antennas — antenna-design workspace.
#
# Ubuntu + Wine + the Windows antenna tools (MMANA-GAL Pro, 4nec2) plus the
# Linux-native xnec2c. The MMANA-GAL / 4nec2 installers are freeware but their
# redistribution terms are murky, so they are NOT committed to this repo: the
# build pulls them from official URLs passed as build-args. Leave the args
# empty and the image still builds — just without the Windows tools
# pre-installed (run their installers by hand inside the workspace later).
FROM kasmweb/core-ubuntu-jammy:1.16.1

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Build-args: official installer URLs. See docs/workspaces/kasm-ubuntu-wine-antennas.md.
ARG MMANA_GAL_URL=""
ARG NEC2_URL=""

# Wine 9 + winetricks + the Linux antenna tool.
RUN dpkg --add-architecture i386 \
    && apt-get update && apt-get install -y --no-install-recommends \
        wine \
        wine64 \
        wine32 \
        winetricks \
        cabextract \
        xnec2c \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-seed a Wine prefix with the runtime the antenna tools expect.
ENV WINEPREFIX=/opt/wineprefix WINEARCH=win64 WINEDEBUG=-all
RUN wineboot --init \
    && winetricks -q corefonts vcrun2019 \
    && wineserver -w

# Pull the freeware installers at build time (skipped when the arg is empty).
RUN mkdir -p /opt/installers \
    && if [ -n "${MMANA_GAL_URL}" ]; then \
         wget -qO /opt/installers/mmana-gal.exe "${MMANA_GAL_URL}"; fi \
    && if [ -n "${NEC2_URL}" ]; then \
         wget -qO /opt/installers/4nec2.exe "${NEC2_URL}"; fi

COPY ./shortcuts/ /usr/share/applications/

RUN chown -R 1000:0 "${HOME}" /opt/wineprefix /opt/installers
ENV HOME=/home/kasm-user
WORKDIR /home/kasm-user
USER 1000

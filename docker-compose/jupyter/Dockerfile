# code-server — linuxserver/code-server extended with the rfdf platform.
FROM lscr.io/linuxserver/code-server:4.96.4

ARG RFDF_VERSION=v0.1.0-alpha

# code-server is an editor, not an ML runtime — install base `rfdf` only.
# The [ml] extra's torchsig dependency tree backtracks pip for hours inside a
# Docker build; ML work happens on the rfdf-tools NFS venv or the dedicated
# rfdf-jupyter host. Add it in-container if needed: /opt/rfdf/bin/pip install 'rfdf[ml]'.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv \
        git \
    && python3 -m venv /opt/rfdf \
    && /opt/rfdf/bin/pip install --no-cache-dir \
        "rfdf @ git+https://github.com/ernesto01louis/rf-direction-finding.git@${RFDF_VERSION}" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/rfdf/bin:$PATH

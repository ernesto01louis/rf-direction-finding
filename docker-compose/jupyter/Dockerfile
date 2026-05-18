# code-server — linuxserver/code-server extended with the rfdf platform.
FROM lscr.io/linuxserver/code-server:4.96.4

ARG RFDF_VERSION=v0.1.0-alpha

# Torch from the CPU-only index — the default wheel bundles ~2 GB of unused
# CUDA libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv \
        git \
        build-essential \
    && python3 -m venv /opt/rfdf \
    && /opt/rfdf/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && /opt/rfdf/bin/pip install --no-cache-dir \
        "rfdf[ml] @ git+https://github.com/ernesto01louis/rf-direction-finding.git@${RFDF_VERSION}" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/rfdf/bin:$PATH

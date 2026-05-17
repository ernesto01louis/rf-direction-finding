# kasm-jupyter-rfdf — desktop workspace with JupyterLab + rfdf pre-installed.
#
# A browser-streamed desktop whose primary app is JupyterLab. The user's
# notebooks volume is mounted by the Kasm workspace config.
FROM kasmweb/core-ubuntu-jammy:1.16.1

USER root
ENV DEBIAN_FRONTEND=noninteractive

ARG RFDF_VERSION=v0.1.0-alpha

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# rfdf + JupyterLab in a dedicated venv.
RUN python3 -m venv /opt/rfdf \
    && /opt/rfdf/bin/pip install --no-cache-dir \
        jupyterlab \
        "rfdf[ml] @ git+https://github.com/ernesto01louis/rf-direction-finding.git@${RFDF_VERSION}"

COPY ./shortcuts/ /usr/share/applications/

RUN chown -R 1000:0 "${HOME}" /opt/rfdf
ENV HOME=/home/kasm-user PATH=/opt/rfdf/bin:$PATH
WORKDIR /home/kasm-user
USER 1000

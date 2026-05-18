# Workspace — kasm-jupyter-rfdf

A browser-streamed desktop whose primary app is JupyterLab, with the `rfdf`
platform pre-installed — for notebook-driven analysis inside Kasm.

## Contents

- **JupyterLab** + base `rfdf` in a dedicated venv at `/opt/rfdf`. The heavy
  `[ml]` extra is not baked in — add it with
  `/opt/rfdf/bin/pip install 'rfdf[ml]'`, or use the rfdf-tools NFS venv.
- A desktop shortcut launches `jupyter lab`.

## Notebooks volume

The Kasm workspace mounts the operator's notebooks volume into the session, so
work persists across sessions. Configure the volume mapping in the Kasm admin
UI when registering the workspace.

## kasm-jupyter-rfdf vs the rfdf-jupyter host

This Kasm workspace is the **desktop** route to JupyterLab — useful when you
want a notebook *alongside* GUI RF tools in one streamed session. The
standalone `rfdf-jupyter` host (`jupyter.rf.lan`) is the **headless** route,
better for long-running kernels and the TrueNAS-backed notebooks share. Use
whichever fits the task.

## Common workflows

- `import rfdf` — DOA estimation, dataset generation, model inference.
- Paper figures + ad-hoc exploration.

## Upstream docs

JupyterLab <https://jupyterlab.readthedocs.io> · rfdf — see the repo `docs/`.

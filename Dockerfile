# rfdf training image.
#
# This is the container image the cloud compute backends (RunPod, Vast.ai,
# SkyPilot) pull to run a training job. It bakes the [ml] + [ml-onnx] extras on
# top of a stock PyTorch CUDA base so a job needs no `pip install` at runtime.
#
# The image is built and pushed by .github/workflows/container.yml — which is
# currently SCAFFOLDED BUT DISABLED (the build-and-push job is guarded
# `if: false`). No live GHCR push happens this stage; until that workflow is
# enabled, the cloud backends fall back to `pip install rfdf[ml,ml-onnx]` into a
# stock PyTorch base image, which needs no published image to work.
#
# Build locally:
#   docker build -t rfdf-training:dev .
# Run a training job (recipe injected via $RFDF_RECIPE or a mounted recipe):
#   docker run --rm --gpus all -e RFDF_RECIPE="$(cat recipe.json)" \
#     rfdf-training:dev python -m rfdf.ml.train_entrypoint

FROM pytorch/pytorch:2.5-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Copy the repository in and install rfdf with the ML extras. --no-cache-dir
# keeps the image lean; the PyTorch base already provides torch/torchvision.
COPY . /workspace
RUN pip install --no-cache-dir '.[ml,ml-onnx]'

# The cloud compute backends invoke the training entrypoint; the recipe is
# supplied via the $RFDF_RECIPE environment variable or a recipe file argument.
CMD ["python", "-m", "rfdf.ml.train_entrypoint"]

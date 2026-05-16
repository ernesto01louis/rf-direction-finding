"""Exception taxonomy for the ML layer.

Every ML failure mode raises a specific :class:`MlError` subclass with an
actionable message, so a caller can distinguish "your HDF5 file is missing a
dataset key" from "the requested model export format is not available" without
parsing strings.

This module is pure stdlib — no torch, no torchsig, no numpy required — so it
is safe to import at module top level from any rfdf code.
"""

from __future__ import annotations


class MlError(Exception):
    """Base class for every error raised by the ``rfdf.ml`` package."""


class DatasetError(MlError):
    """A dataset could not be loaded, parsed, or validated.

    Covers missing files, wrong HDF5 schema, unsupported modulation types,
    and unavailable datasets requiring download consent.
    """


class AugmentationError(MlError):
    """An augmentation could not be applied.

    Raised when IQ is the wrong shape/dtype, a parameter is out of physical
    range, or a required dependency (numpy, etc.) is unavailable.
    """


class ModelError(MlError):
    """A model definition, checkpoint, or registry entry is invalid.

    Covers missing checkpoints, architecture mismatches, and registry
    look-up failures for unknown model names.
    """


class TrainingError(MlError):
    """A training run failed to start or was interrupted unrecoverably.

    Distinguishable from a normal early-stop or user cancellation: only raised
    when the training loop itself encounters a hard error (NaN loss, GPU OOM,
    etc.) that makes the checkpoint unusable.
    """


class InferenceError(MlError):
    """Inference failed on a loaded model.

    Covers shape mismatches between the IQ input and the model's expected
    input, missing ONNX/TFLite runtime, and runtime execution errors.
    """


class ExportError(MlError):
    """A model could not be exported to the requested format.

    Covers unsupported export targets (ONNX, CoreML, TFLite), dynamic-shape
    failures during tracing/scripting, and missing optional extras
    (e.g. ``rfdf[ml-onnx]``, ``rfdf[ml-coreml]``).
    """


class RegistryError(MlError):
    """A model could not be found in or registered with the model registry.

    Raised when a caller requests a named model that does not exist in the
    registry, or attempts to register a model with a duplicate key without the
    ``overwrite=True`` flag.
    """

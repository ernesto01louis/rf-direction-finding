# Model export

`rfdf.ml.export` converts a trained PyTorch model into a portable format for
deployment. Four export paths ship, matching the three inference backends in
`rfdf.ml.inference` plus the optional Apple target.

## CLI

```
rfdf ml export <model_id> --format <onnx|hef|tflite|coreml> --output <path>
```

Export reads a model from the [registry](registry.md), reconstructs it, and
writes the requested format. `ExportError` is surfaced as a clean CLI message.

## ONNX — the universal target

```python
from rfdf.ml.export import export_onnx
export_onnx(model, sample_input, Path("model.onnx"), opset=17)
```

`export_onnx` calls `torch.onnx.export` with a dynamic batch axis on both
`input` and `logits` (opset 17 by default). It is **always available** with the
`[ml]` extra. ONNX is the portable fallback: it runs on production CPU and most
GPUs via `onnxruntime` (the `[ml-onnx]` extra), and it is the intermediate
representation the Hailo and CoreML paths consume. `RfdfClassifier.to_onnx` is
the equivalent method on the model object.

## HEF — Hailo-8L for the reference Pi 5

The Hailo HEF format targets the Hailo-8L accelerator on the reference Pi 5.
**The Hailo Dataflow Compiler runs outside this platform.** It is a *vendor*
tool distributed by Hailo AI — it is **not on PyPI** and the platform does not
install it. The operator owns the compile step:

1. The platform emits ONNX (`rfdf ml export <model_id> --format onnx`).
2. On a host with the Hailo Dataflow Compiler installed, run the compiler
   (`hailomz compile` / `hailo` CLI) against that ONNX to produce a `.hef`.
3. Place the `.hef` in the model's registry directory (or run
   `rfdf ml export --format hef`, which shells out to the compiler when it is
   on `PATH`).

`export_hailo` reflects this division of responsibility: if a `.hef` already
exists it validates the magic bytes (`HEF\0`) and returns it unchanged; if the
compiler is on `PATH` it shells out to it; otherwise it raises an `ExportError`
with installation instructions. The platform never bundles the compiler — it
emits ONNX and validates the resulting HEF. To register at Hailo's developer
zone and install the Dataflow Compiler, see <https://hailo.ai/developer-zone/>.

The 2-D architectures (`resnet2d`, `efficientnet`) are the natural Hailo
targets — Conv2D is the Hailo-8L sweet spot.

## TFLite — Coral edge fallback

```python
from rfdf.ml.export import export_tflite
export_tflite(model, sample_input, Path("model.tflite"), quantize="int8")
```

`export_tflite` uses `ai-edge-torch` (a shim for `litert-torch`) — the
`[ml-tflite]` extra, declared as `["ai-edge-torch>=0.2"]`. The converter runs
on Linux; executing the flat-buffer on a device needs the TFLite / LiteRT
runtime. The `quantize` argument (`fp32` / `fp16` / `int8`) is accepted for API
compatibility — full quantization needs an explicit calibration config.

**Install `ml-tflite` in its own environment.** `ai-edge-torch` pins specific
`torch` / `torchvision` versions and installing it alongside the plain `[ml]`
extra upgrades those packages, which can break the other ML paths. For that
reason `ml-tflite` is a standalone extra — it is not co-installed in the
`coverage-ml` CI job, and `export_tflite` is tested adaptively
(`importorskip` / `xfail`).

## CoreML — optional macOS target

```python
from rfdf.ml.export import export_coreml
export_coreml(model, sample_input, Path("model.mlpackage"))
```

`export_coreml` uses `coremltools` (the `[ml-coreml]` extra), converting via an
ONNX intermediate to a CoreML `mlprogram`. The **conversion step runs on
Linux**, but `libcoremlpython` — the native extension that loads and validates
a CoreML model — is **macOS-only**. So conversion works in CI, but
running/validating the model requires a Mac. `export_coreml` tests `xfail` on a
conversion failure rather than blocking the suite.

## Export targets at a glance

| Format | Function | Extra | Inference backend | Notes |
|---|---|---|---|---|
| ONNX | `export_onnx` | `[ml]` (always) | `onnxruntime` (`[ml-onnx]`) | universal; intermediate for HEF / CoreML |
| HEF | `export_hailo` | none (vendor compiler) | `hailo_platform` (`[ml-hailo]`) | compiler runs **outside** the platform |
| TFLite | `export_tflite` | `[ml-tflite]` | TFLite / LiteRT runtime | install in a standalone env |
| CoreML | `export_coreml` | `[ml-coreml]` | CoreML (macOS) | conversion on Linux; validation needs a Mac |

See [models.md](models.md) for the architectures these export from and
[registry.md](registry.md) for where exported artefacts are recorded.

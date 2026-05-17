# ML datasets

The `rfdf.ml.datasets` package supplies the signal data that the training loop
consumes. Every loader returns a `torch.utils.data.Dataset` of
`(complex64 IQ tensor, int label)` pairs and accepts an optional
[`AugmentationConfig`](#augmentation). All `torch` / `torchsig` / `h5py` imports
are lazy — importing `rfdf.ml.datasets` without the `[ml]` extra never pulls in
PyTorch.

`build_datasets(spec)` is the single dispatch point: it takes a `DatasetSpec`
(from a [training recipe](training.md)) and returns a `(train, val, test)`
triple. For the synthetic kinds the three splits are derived from one generator
with disjoint seed offsets (`spec.seed`, `spec.seed ^ 0xABCD_1234`,
`spec.seed ^ 0xDEAD_BEEF`) so no IQ window appears in more than one split. For
`radioml` and `captured` the loader's own `split` parameter selects the
partition.

## What ships

| Kind | Loader | Source | License |
|---|---|---|---|
| `modulation` | `make_modulation_dataset` | TorchSig 2.x synthetic signal generators | TorchSig (MIT) |
| `protocol` | `make_protocol_dataset` | TorchSig generators mapped to protocol families | TorchSig (MIT) |
| `radioml` | `load_radioml` | DeepSig RadioML 2018.01A HDF5 | **CC-BY-NC-SA 4.0** |
| `captured` | `load_captured_dataset` | User SigMF recordings | user's own |

## Synthetic — TorchSig

`make_modulation_dataset` and `make_protocol_dataset` wrap TorchSig's iterable
dataset. They are the zero-download path: no large file fetch, fully
reproducible from a seed.

- **`make_modulation_dataset(num_signals, num_samples_per_signal, impairments,
  augmentation, seed)`** — generates one IQ recording per modulation class
  (`num_signals` classes drawn from the TorchSig catalogue in class-index
  order; the 2.1.x catalogue is ≈53 classes). `num_samples_per_signal` is the
  IQ length of each recording. Long recordings are slow to synthesise, and
  TorchSig's modulation generators occasionally raise `ValueError: Passband
  ripple was unable to meet ripple specs` from SciPy's filter design for
  certain class/length combinations — retry with a different `seed` if you
  hit it. For a fast classification dataset with many samples per class,
  prefer `make_protocol_dataset`.
- **`make_protocol_dataset(protocols, num_samples_per_protocol, num_iq_samples,
  impairments, augmentation, seed)`** — generates `num_samples_per_protocol`
  recordings per protocol. Default protocols are `lora`, `zigbee`, `wifi`,
  `bluetooth`, `noise`, mapped to TorchSig generator families (`chirpss`,
  `ofdm-64`, `qpsk`, …) with a Gaussian-noise class.

`impairments` selects TorchSig's built-in channel model: `clean` (no channel),
`cabled` (moderate), or `ota` (heavy over-the-air). The TorchSig 2.x surface is
isolated in `rfdf.ml.datasets._torchsig_compat` — the only module that imports
`torchsig` — so future TorchSig drift touches exactly one file.

## RadioML 2018.01A

`load_radioml` reads the DeepSig RadioML 2018.01A HDF5 file: 24 modulation
classes, ≈2.5M examples, ≈20 GB on disk. `snr_filter` and `class_filter`
restrict which examples load.

**License — CC-BY-NC-SA 4.0.** RadioML 2018.01A is distributed by DeepSig Inc.
under Creative Commons Attribution-NonCommercial-ShareAlike 4.0. **Commercial
use is prohibited**, and a citation is required (O'Shea & West, "Radio Machine
Learning Dataset Generation with GNU Radio", GNU Radio Conference 2016). The
loader docstring repeats this notice.

The loader **never downloads automatically**. Pass `download=True` to trigger a
first-use fetch into the platform cache
(`~/.local/share/rfdf/datasets/radioml-2018.01a/`); the download takes many
minutes. The recipe-driven path (`build_datasets`) always passes
`download=False` — the operator stages the HDF5 file and sets `dataset_path`.

## Captured SigMF

`load_captured_dataset` loads user SigMF recordings — a single
`.sigmf-meta` / `.sigmf-data` pair or a directory glob. Labels are extracted
from SigMF `annotations`; long recordings are windowed to a fixed length.

The split is **by capture session, not by sample**: every window from one
recording stays in a single partition. This prevents data leakage — a model
must not be validated on windows that share capture-day artefacts with its
training data. Session identity is derived from the `core:hw` global field and
the `.sigmf-meta` filename stem.

## Augmentation

`AugmentationConfig` (in `rfdf.ml.datasets.augmentation`) is a pure-NumPy,
torch-free impairment framework. Each field is either `None` (disabled) or a
`(min, max)` range sampled uniformly per item:

| Field | Impairment |
|---|---|
| `add_awgn` | additive white Gaussian noise to a target SNR (dB) |
| `frequency_shift` | carrier frequency offset (Hz) |
| `gain_variation` | front-end gain error (dB) |
| `iq_imbalance` | quadrature phase imbalance (degrees) |
| `multipath` | random multipath FIR channel (`MultipathConfig`) |
| `impulsive_noise` | `(probability, amplitude)` impulse injection |
| `sample_rate_jitter` | crystal-oscillator sample-rate error (ppm) |

`apply_augmentations` applies the enabled impairments in a physically plausible
order — channel distortion first, receiver-hardware impairments next, noise
last. Dataset loaders call it per item with a deterministic RNG derived from
`seed ^ index`, so an augmented dataset is fully reproducible.

Augmentation is not optional cosmetics. Without aggressive augmentation a model
overfits capture-day artefacts (a specific radio, a specific SNR) and fails on
anything else — see [training.md](training.md) for the train/test SNR-mismatch
discipline. The platform's value is *useful* trained models, not merely trained
ones.

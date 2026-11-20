# Architecture and model registration

## Repository map

| Location | Responsibility |
| --- | --- |
| `configs/` | Hydra data, model, trainer, callback, logger, registry, and experiment groups. |
| `src/data/` | Dataset preparation and Lightning data modules. |
| `src/models/components/` | PyTorch network architectures. |
| `src/models/mnist_module.py` | Classification steps, loss, metrics, optimizer, and scheduler. |
| `src/callbacks/` | Checkpoint and artifact hooks. |
| `src/utils/` | Instantiation, logging, and shared model-publication helpers. |
| `src/train.py`, `src/eval.py` | Hydra training and checkpoint-evaluation entry points. |
| `tests/` | Data, configuration, bounded training, evaluation, and registry tests. |

`notebooks/` is for exploration; `inference/` holds any retained training-side work.
The separately installable inference application lives in
[vision-inference](https://github.com/ohadjerci/vision-inference).

## Data and training flow

Versioned training first verifies committed executed source and calls the shared
`src/data/prepare_dataset.py` implementation. Data modules export an immutable
snapshot and reload its saved files/splits. DVC stores dataset contents; Git stores
only pointers. MLflow stores searchable dataset versions, provenance, and the
existing model artifacts. Code and dataset-metadata revisions are recorded
separately, and a content-based fingerprint allows reuse across metadata commits.
See [configuration](configuration.md#dataset-versioning-and-reproduction) for the
independent data stage, storage and reproduction command.

1. Hydra composes configuration and instantiates the data module, Lightning model,
   callbacks, loggers, and trainer. The default global seed is 1.
2. The data module downloads MNIST if missing, combines the original train/test
   datasets, and splits them into 55,000 training, 5,000 validation, and 10,000 test
   samples using split seed 42. This is not the official MNIST test split.
3. Samples become grayscale tensors normalized with mean `0.1307` and standard
   deviation `0.3081`. The default batch size is 128.
4. A configurable ResNet learns ten classes with cross-entropy loss and Adam.
   ReduceLROnPlateau and early stopping monitor `val/loss`; stopping patience is 50.
5. Epoch metrics include loss, accuracy, precision, recall, and F1. Confusion-matrix
   state is maintained, but this does not imply automatic plotted artifacts.
6. Training selects the checkpoint with lowest `val/loss` and tests it. Standalone
   evaluation instead tests its explicitly supplied checkpoint.

## Registration and inference

After successful testing, a shared helper in
[`src/utils/model_registry.py`](../src/utils/model_registry.py) runs on global rank
zero when an MLflow logger and registry publication are enabled. It copies the
network to CPU, exports TorchScript, and logs a portable PyTorch-flavor artifact
with digit labels and `mnist-v1` preprocessing metadata. The training module and
checkout are not needed to load this network in the inference container.

Each publication registers a numbered `MNIST` version, with source-run information,
checkpoint name, selection metric, and score. `best_model` points to one version.
The inference CLI resolves it once and then downloads a numeric version URI.

Artifact and publication errors are surfaced. Publishers must run sequentially;
there is no atomic cross-process compare-and-promote lock.

## Model selection mismatch

**Agreed requirement:** select `best_model` using validation results.

**Implemented behavior:** the training checkpoint is selected by `val/loss`, but
[`configs/registry/default.yaml`](../configs/registry/default.yaml) sets the registry
metric to **`test/loss`**. After testing, `publish_evaluated_model` reads that test
metric; `promote_if_better` moves the alias only for a strictly lower finite score.
Ties retain the current alias. Standalone evaluation follows the same test-loss
promotion rule. Missing/incompatible comparison metadata raises an error.

Validation-based alias promotion is not implemented. Documentation changes do not
fix this behavior; it needs a separate code change and tests. Do not assume that
changing `registry.metric` to `val/loss` is sufficient, since publication currently
reads finalized test callback metrics.

## Extension limits

MNIST remains the default. The shared classification module, configurable ResNet
channels, and input-shape-aware export also support PCB through `experiment=pcb`.
See [PCB classification](pcb.md) for image-level labels, isolated splits, complete
S-group batches, result artifacts, and just recipes. PCB currently requires one
device/node; MNIST retains its distributed configuration. Some retained
legacy detection utilities, debug/sweep presets, and the custom cluster plugin
still need migration; they are not a supported first-run path.

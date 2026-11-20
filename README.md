# ML Engineering Lab

## Project overview

Train and evaluate image classifiers with PyTorch, Lightning, and Hydra. The
current example is a custom ResNet for MNIST, with configurable training,
checkpoint resume, metrics, and MLflow tracking.

This project publishes packaged models to a shared MLflow registry.
[Vision inference](https://github.com/ohadjerci/vision-inference) downloads those
models and predicts digits independently on CPU.

## Prerequisites

- Linux x86-64, Git, Docker Engine, and Docker Compose 2.30 or later.
- An NVIDIA GPU, compatible driver, and NVIDIA Container Toolkit for the
  **training and test Compose services**, even when selecting a CPU trainer.
- Internet access for images, locked dependencies, and the first MNIST download;
  allow substantial disk space for the CUDA image.

MLflow itself and the separate inference application can run without a GPU.
Python 3.12, uv, PyTorch, and CUDA dependencies are installed in containers;
no host Python environment is needed. See [configuration](docs/configuration.md)
for CPU options and the current Compose limitation.

## Quick start

Run commands in the **training repository root** after cloning.

1. **Get the project.**

   ```bash
   git clone https://github.com/ohadjerci/ml-engineering-lab.git
   cd ml-engineering-lab
   ```

2. **Create persistent directories and build the image.**

   ```bash
   mkdir -p data/.dvc-remote data/.dvc-cache data/.locks
   mkdir -p logs/train logs/eval logs/prepare_dataset logs/reproduce logs/mlflow/db logs/mlflow/artifacts
   export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
   docker compose build train
   ```

3. **Give the MLflow container user access to its new directories.** Training and
   dataset preparation use your configured host UID/GID and the directories you
   created. This command changes directory ownership, not existing model files. Review the
   [storage guidance](docs/mlflow.md#storage-permissions) for an existing checkout.

   ```bash
   docker compose run --rm --no-deps --user root --entrypoint sh mlflow \
     -c 'chown mlcycle:mlcycle /workspace/logs/mlflow /workspace/logs/mlflow/db /workspace/logs/mlflow/artifacts'
   ```

4. **Start MLflow in the background.**

   ```bash
   docker compose up -d mlflow
   ```

   Open [MLflow](http://localhost:50001). Training waits for the server to become
   healthy and starts it automatically if needed.

5. **Run the first training example** shown below.

## Training example and results

From the **training repository root**, run one epoch on the default GPU trainer:

```bash
docker compose run --rm train uv run --locked python src/train.py \
  trainer.max_epochs=1 data.num_workers=0
```

This downloads MNIST, trains, tests the best validation checkpoint, and registers
a packaged `MNIST` model. A short run demonstrates the workflow; its accuracy is
not a quality guarantee. Omitting the epoch override uses a maximum of 500 epochs.

Training now first prepares an immutable dataset snapshot, uploads it to the selected
DVC remote, and commits only its DVC metadata. Equivalent completed MLflow runs are
reused when their model artifacts remain intact. The checkout's relevant source must
be committed and the Docker image rebuilt; metadata commits never include source edits
or unrelated staged work. Configure Git author/committer identity in the repository
or environment before the first launch.

```bash
# Independent dataset preparation, without requesting a GPU.
just prepare-data
just prepare-data-pcb
# Train even if a matching completed run exists.
just train reproducibility.force_retrain=true
# Restore code, data and settings into an isolated checkout and train again.
just reproduce ORIGINAL_MLFLOW_RUN_ID
```

These recipes use Docker Compose. See [dataset configuration and reproduction](docs/configuration.md#dataset-versioning-and-reproduction)
for storage permissions, SSH/S3 credentials, reuse, and recovery. Retain the referenced
Git revisions, DVC objects and MLflow artifacts for as long as experiments must remain
reproducible.

- Checkpoints: `logs/train/runs/<timestamp>/checkpoints/`, including `last.ckpt`.
- Run configuration, logs, and results: `logs/train/runs/<timestamp>/`.
- Metrics and model versions: the **`MNIST`** experiment and **`MNIST`**
  registered model in [MLflow](http://localhost:50001). The UI stays running.

Inference can select a numeric model version or the `best_model` alias.
**Current mismatch:** checkpoint selection uses validation loss, but alias
promotion uses **test loss**, contrary to the agreed validation-based promotion
requirement. See [model selection](docs/architecture.md#model-selection-mismatch).

### Training with uncommitted changes

Use `just train-dev` for MNIST or `just train-pcb-dev` for PCB while editing code
or configuration. These recipes rebuild the image with your local changes and
set `versioning.enabled=false`. The run skips DVC snapshots, automatic Git metadata
commits, and experiment reuse. The usual MLflow logging still applies.

```bash
just train-dev
just train-pcb-dev trainer=cpu trainer.precision=32-true trainer.max_epochs=1
# Equivalent without the shortcut:
just train-pcb versioning.enabled=false
```

Use `just train` or `just train-pcb` after committing relevant source and
configuration to enable versioning again. Local commits are enough for training.
If you only need to version the dataset, `just prepare-data-pcb` can run while
source edits are uncommitted and commits only the dataset metadata.

## Further documentation

- [VS Code breakpoint debugging in Docker](docs/development.md#vs-code-breakpoints-in-docker)
- [MLflow setup, storage, and shared connectivity](docs/mlflow.md)
- [Environment variables and Hydra configuration](docs/configuration.md)
- [Architecture and model registration](docs/architecture.md)
- [Development, tests, coverage, and CI](docs/development.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Historical file-store migration](docs/migration.md)
- [Vision inference repository](https://github.com/ohadjerci/vision-inference)

## PCB classification and command recipes

MNIST remains the default. See [PCB workflow](docs/pcb.md) for complete-group
binary classification, configurable Hydra presets, and the `justfile` recipes
for training, checkpoint validation/evaluation, tests, Ruff, and MyPy.

# Configuration

Run commands from the **training repository root**. Training and evaluation keep
the existing Hydra entry points and YAML configuration groups.

## Environment variables

[`.env.example`](../.env.example) shows the host-client values. Docker Compose
reads a root `.env` for interpolation, but it does not automatically forward every
variable into a container. The image also excludes `.env` files from its build.

| Variable | Implemented behavior |
| --- | --- |
| `MLFLOW_TRACKING_URI` | Hydra fallback: `http://localhost:50001`; Compose `train` explicitly sets `http://mlflow:5000`. |
| `MLFLOW_EXPERIMENT_NAME` | Hydra fallback: `project_http`; pass with Compose `-e` or override the Hydra logger setting. |
| `PROJECT_ROOT` | Set by `pyrootutils` root discovery; normally `/workspace` in the image. |
| `LOCAL_UID`, `LOCAL_GID` | Host UID/GID for the `lock` maintenance service; defaults are `1000:1000`. |
| `MLFLOW_ALLOW_FILE_STORE` | Legacy file-store compatibility, not needed for new HTTP runs; the test service forwards it with default `false`. |

For example, choose another experiment explicitly:

```bash
docker compose run --rm -e MLFLOW_EXPERIMENT_NAME=my_experiment train \
  uv run --locked python src/train.py trainer.max_epochs=1
```

Keep credentials private. Do not stage `.env` even if a local ignore-rule override
makes it visible in Git status; `.env.example` contains shareable values only.

## Hydra overrides

The defaults select MNIST, a grayscale ten-class ResNet, Adam, a GPU trainer,
MLflow, and optional testing after fitting. Supply the full Python command after
`train` when adding overrides, because it replaces the service command.

```bash
docker compose run --build --rm train uv run --locked python src/train.py \
  trainer.max_epochs=20 model.optimizer.lr=0.0001 data.batch_size=64
```

Use `+` for trainer keys not declared in the YAML. A bounded smoke run is:

```bash
docker compose run --rm train uv run --locked python src/train.py \
  trainer.max_epochs=1 +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=2 +trainer.limit_test_batches=2 data.num_workers=0
```

This tests the integration, not model accuracy. Use `registry.enabled=false` to
skip registration, or `test=false` to skip testing and publication together.
`compile=true` enables the existing optional PyTorch compilation path.

## CPU execution

The Python training/evaluation path supports CPU and full precision:

```bash
docker compose run --rm train uv run --locked python src/train.py \
  trainer=cpu trainer.precision=32-true trainer.max_epochs=1 data.num_workers=0
```

**The current `train` and `tests` Compose services still request NVIDIA GPUs.**
This override does not make them usable on a GPU-free Docker host. MLflow alone
and the separate inference service do not request GPUs. Hosted CI runs CPU tests
without using the local training Compose service.

## Paths, resume, and evaluation

Hydra writes single runs to `logs/<task_name>/runs/<timestamp>/` and multiruns to
`logs/<task_name>/multiruns/<timestamp>/<job_number>/`. Default task names are
`train` and `eval`, the two persisted output mounts. Checkpoints are saved every
epoch, with all epoch files retained (`save_top_k: -1`) plus `last.ckpt`.

Replace `RUN` below with a real timestamp directory and preserve the checkpoint's
model architecture configuration.

```bash
docker compose run --rm train uv run --locked python src/train.py \
  ckpt_path=/workspace/logs/train/runs/RUN/checkpoints/last.ckpt trainer.max_epochs=20
```

Standalone evaluation does not use an MLflow logger by default. Select it to
track metrics and publish the evaluated checkpoint:

```bash
docker compose run --rm train uv run --locked python src/eval.py \
  logger=mlflow ckpt_path=/workspace/logs/train/runs/RUN/checkpoints/epoch_000.ckpt
```

For CPU evaluation add `trainer=cpu trainer.precision=32-true`. Publication is
configured in [`configs/registry/default.yaml`](../configs/registry/default.yaml).
Changing its metric string alone does not implement validation-based promotion;
see the [known mismatch](architecture.md#model-selection-mismatch).

MNIST's configured dataset root is `/workspace/data/MNIST/`. The `train` service
does not mount `data/`, so downloads in a disposable training container are not
retained. To reuse downloads, add `-v "$PWD/data:/workspace/data"` before the service
name and ensure that host directory is writable by `mlcycle` first.

Machine-specific Hydra overrides belong in ignored `configs/local/`. CSV and
TensorBoard are available logger presets; other included external logger presets
may require optional dependencies and account settings.

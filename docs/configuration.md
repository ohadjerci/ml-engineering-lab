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
| `MLFLOW_EXPERIMENT_NAME` | Defaults to `data.name` (`MNIST`, `PCB`, etc.); pass with Compose `-e` or override the Hydra logger setting. |
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

Training and debugging mount host `data/` at `/workspace/data`. MNIST uses
`/workspace/data/MNIST/`; PCB uses `/workspace/data/REALIAD/pcb` and
`/workspace/data/REALIAD/pcb.json`, as configured in YAML. Downloads persist in
the host directory; ensure it is writable by `mlcycle` first.

Machine-specific Hydra overrides belong in ignored `configs/local/`. CSV and
TensorBoard are available logger presets; other included external logger presets
may require optional dependencies and account settings.
# Dataset versioning and reproduction

For the storage layout, snapshot behavior, Git/file hashes, and step-by-step
replay flow, see [versioning, storage, and reproduction](versioning-and-reproduction.md).

Dataset selection remains in `configs/data/`. `src/data/prepare_dataset.py` uses
`configs/data/prepare_dataset.yaml`; normal training calls the same implementation.
The existing data modules implement `export_snapshot(destination)` and accept
`snapshot_path` to load recorded samples and splits. Shared code does not branch on
dataset names. A new dataset using the PCB format needs only another data config
inheriting `pcb`, with a different `name`, `root_dir` and `manifest_path`. Select it
with `just train-pcb data=your_dataset` or `just prepare-data data=your_dataset`.
Only a genuinely new format requires extending/adding a data module implementing
the same interface. No separate adapter class is required.

## Local storage and first launch

```bash
mkdir -p data/.dvc-remote data/.dvc-cache data/.locks logs/prepare_dataset logs/reproduce
# Use your actual host UID/GID; services writing metadata run with these IDs.
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
git config user.name 'Your Name'
git config user.email 'you@example.com'
docker compose build train prepare_dataset
docker compose run --rm prepare_dataset
docker compose run --rm train
```

The host checkout is mounted at `/repo` for Git and DVC. Training executes the
image's source at `/workspace`, whose relevant files must match committed Git
contents. Rebuild after source changes. Relevant dirty or untracked source fails
before preparation/training rather than recording an inaccurate code SHA.
The checkout's `logs/` is masked to preserve MLflow server storage isolation.
Ignored `configs/local/` overrides are mounted read-only and their resolved settings
are included in provenance. Keep credentials outside Hydra configuration.

Data and DVC cache persist beneath `data/`. Set `DVC_LOCAL_REMOTE_DIR` to any writable
host folder; it mounts at `/dvc-remote`. The default is `data/.dvc-remote`.
An external/mounted folder can use the same local preset. Ensure existing storage
is writable by the configured UID/GID; do not recursively change ownership of
historical tracking data. MLflow retains its existing separate storage/user setup.

## SSH and S3

```bash
DVC_REMOTE_URL=ssh://user@host/path/to/store \
  just prepare-data 'data/versioning@versioning=ssh'
DVC_REMOTE_URL=s3://bucket/prefix \
  just train 'data/versioning@versioning=s3'
```

Set `DVC_REMOTE_NAME` to choose the named remote. Remote URLs must not contain
passwords, query credentials or tokens. DVC location/cache settings are written
to ignored `.dvc/config.local`; only bootstrap metadata and snapshot pointers
are committed. Use AWS provider environment variables, an AWS role/profile, or
private credential mounts. For SSH use an agent or a read-only private key and
known-hosts mount; never disable host-key verification.

Add private mounts in ignored `compose.private.yaml`, for example:

```yaml
services:
  train:
    volumes:
      - ${HOME}/.ssh:/tmp/.ssh:ro
      - ${HOME}/.aws:/tmp/.aws:ro
  prepare_dataset:
    volumes:
      - ${HOME}/.ssh:/tmp/.ssh:ro
      - ${HOME}/.aws:/tmp/.aws:ro
```

Apply the same override explicitly with Compose's `-f` flags when needed; do not
commit private paths/settings. Reproduction also needs those credential mounts;
environment-based AWS credentials are forwarded automatically. SSH key permissions
must permit the configured UID to read them. Run DVC credential configuration only
with `--local`, and never pass secret Hydra overrides or log shell environments.
For reproduction set `REPRODUCTION_COMPOSE_OVERRIDE` to the absolute path of that
private Compose override so both preparation and isolated training use its mounts.
If the Git remote also uses SSH, set `GIT_SSH_COMMAND` to use the mounted identity
and known-hosts files, for example `ssh -i /tmp/.ssh/id_ed25519 -o UserKnownHostsFile=/tmp/.ssh/known_hosts`.

## Controls and reuse

`versioning.enabled`, `upload`, and `git_commit` default to true. `git_push` defaults
to false; enabling it requires explicit `git_remote` and `git_branch`. Upload and
metadata-commit errors stop training. No force-push or blanket staging is used.
`dvc add` caches data; `git commit` records pointers. These are distinct operations.

`reproducibility.reuse=true` searches the selected MLflow experiment for a successful
run with the same logical dataset, content/splits, training configuration, initialized
weights, source content, dependencies and runtime. It verifies checkpoint/model files
before reusing them. Output directories, run names, DVC locations and metadata-only
Git commits do not affect reuse. Different dataset names remain separate even when
sharing an experiment. Changed evaluation settings reuse compatible training weights
but execute evaluation in a new linked run, without duplicate registration.

```bash
just train reproducibility.force_retrain=true
# Explicit exploration without versioning/reuse/provenance requirements:
just train versioning.enabled=false
```

Legacy runs lacking the completion/provenance schema are never reused. Training
must save a checkpoint. Disabling upload or metadata commits is possible, but those
runs cannot support complete reproduction. Existing standalone eval/validate and
registry selection remain unchanged.

## Reproduction and retention

```bash
just reproduce ORIGINAL_MLFLOW_RUN_ID
# Equivalent without just:
bash scripts/reproduce.sh ORIGINAL_MLFLOW_RUN_ID
```

Start MLflow first. This retrieves provenance through Compose, clones recorded code
under `logs/reproduce/`, checks historical reproduction support and source hashes,
restores dataset pointers at their metadata revision, pulls/verifies every dataset
file and split, builds the recorded Dockerfile/lockfile, and starts fresh training
linked to the original run. Actual initialization weights or resume checkpoints
are restored from MLflow and verified. The isolated build uses its own image tag.
It neither edits historical runs nor overwrites the current checkout. Credentials
come from the current environment. Historical code without the reproduction entry
point fails explicitly; current training code is never substituted silently.

The default local remote location is carried into the isolated Compose project.
When using another local folder set `DVC_LOCAL_REMOTE_DIR`; for SSH/S3 provide the
remote URL and credentials again. Reproduction requires the historical runtime to
remain buildable and compatible hardware. Recorded Python/package versions must
match; other runtime differences are reported and recorded in the new run. Matching data/settings supports
reproducibility; identical floating-point results also depend on hardware and
deterministic operations.

Retain referenced Git code and metadata commits, remote DVC objects, MLflow artifacts,
initialization checkpoints, dependency distributions and base images. Do not run DVC
garbage collection or remove experiment artifacts without accounting for all retained
MLflow run references, including commits not reachable from a current branch.

## Failures and concurrency

Preparation uses atomic snapshots and a publication lock; equivalent training launches
share a fingerprint lock through completion. These guarantees cover Linux launches
sharing the configured local `versioning.coordination_dir`. A single Lightning launcher
may start distributed workers on one host; external multi-node launchers are rejected.
Independent hosts with different lock storage are not deduplicated. Existing registry
alias promotion still requires sequential publishers, including different fingerprints.

If upload or commit fails, repair credentials/permissions/Git identity and rerun the
same command. Completed snapshots and a local metadata journal make retries idempotent.
Unexpected user edits to owned DVC metadata are rejected; inspect and resolve them
without discarding unrelated staged work. Avoid manual Git operations during automatic
publication. A failed/interrupted MLflow run is never considered reusable. Git push
failure leaves the local commit available for recovery and prevents fitting.

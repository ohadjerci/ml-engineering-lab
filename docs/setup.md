# Setup explained

After cloning, run `bash scripts/setup.sh` as your normal host user. It performs
the storage, image-build, and MLflow-start steps from the quick start. It does not
download/version datasets, train a model, commit changes, install host packages,
or delete existing experiments. Subsequent training uses `just train` or
`docker compose run --build --rm train`.

## Prerequisites

Use Linux, Git, Docker Engine with Compose 2.30+, and access to the Docker daemon.
The script checks Git/Docker availability and daemon connectivity. Training still
requires the NVIDIA driver and Container Toolkit; setup does not install or
verify GPU support. `just` is optional because equivalent Compose commands exist.
The first build needs internet access and substantial disk space.

## What the script does

1. Changes to the repository root, so it works when invoked from another directory.
2. Records the current host user's UID/GID as `LOCAL_UID`/`LOCAL_GID` in ignored
   `.env`. Existing entries for those two keys are replaced; other settings are
   preserved. This persists across terminals, unlike `export` inside a script.
   Run without `sudo`; root execution is rejected. `.env` is kept private (mode 600).
3. Resolves `DVC_LOCAL_REMOTE_DIR` through Compose's environment handling and creates
   its directory plus local data, cache, snapshot, coordination, configuration,
   training/preparation output, and MLflow directories before Docker can create
   root-owned bind mounts. Existing shell variables take precedence over `.env`.
4. Builds `mlcycle:uv` using the locked dependencies and Dockerfile. Training,
   preparation, and MLflow share this image.
5. Checks write access from the preparation container as the configured host user.
   It reports an error for incompatible existing ownership; default setup does
   not recursively change dataset ownership.
6. Assigns the three MLflow storage directories to the image's `mlcycle` server
   user. This changes directory ownership only, not existing nested artifacts.
7. Starts MLflow in the background and waits up to 120 seconds for its health check.
   Failure stops setup; inspect `docker compose logs --tail 50 mlflow`.
8. Prints training commands and reports missing Git identity if necessary.

Setup can be rerun: directories and data remain, builds reuse Docker's cache, and
Compose keeps the service running. A rebuild may update the MLflow container to
the current image. The default DVC remote remains local storage, not an independent
backup. See [versioning and storage](versioning-and-reproduction.md).

## Manual equivalent

These commands show each operation for users who prefer manual setup. Run them
from the repository root; create any custom DVC remote directory before Docker.

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
# Persist these values in ignored .env if you want them available in new shells.
mkdir -p "${DVC_LOCAL_REMOTE_DIR:-./data/.dvc-remote}" data/.dvc-cache data/.locks data/snapshots configs/local
mkdir -p logs/train logs/eval logs/prepare_dataset logs/reproduce logs/mlflow/db logs/mlflow/artifacts
docker compose build train
docker compose run --rm --no-deps --user root --entrypoint sh mlflow \
  -c 'chown mlcycle:mlcycle /workspace/logs/mlflow /workspace/logs/mlflow/db /workspace/logs/mlflow/artifacts'
docker compose up -d --wait --wait-timeout 120 mlflow
```

If `DVC_LOCAL_REMOTE_DIR` is defined only in `.env`, use that path explicitly in
the manual `mkdir` command; the host shell does not automatically load `.env`.
The script handles Compose's effective value automatically.

Versioned training commits dataset metadata and needs your Git identity:

```bash
git config user.name 'Your Name'
git config user.email 'you@example.com'
```

Use your own identity; setup does not invent one. The usual Git author/committer
environment variables can also supply it. Relevant source/configuration must be
committed before versioned training. See the
[training quick start](../README.md#training-example-and-results).

## Permission recovery

`mkdir -p` preserves existing ownership. If Docker created storage as root after
cleanup, rerun setup with explicit repair:

```bash
bash scripts/setup.sh --repair-permissions
```

This recursively assigns the mounted DVC remote, local data, and training,
evaluation, preparation, and replay outputs to the current host user. Use it only
when that account should own those files. It does not recursively change MLflow's
historical artifacts/database. A host parent directory must already allow creating
the required paths; otherwise arrange host access first. Write access to the Git
checkout is also required and is not repaired by this option.

For existing MLflow storage, inspect nested permissions and follow
[MLflow storage permissions](mlflow.md#storage-permissions). For SSH/S3, configure
provider credentials and permissions separately; the repair option handles local
mount ownership only. No dataset bytes are recreated if they have been deleted.

## Stop and restart

```bash
docker compose stop mlflow
docker compose up -d --wait --wait-timeout 120 mlflow
```

Stopping the service preserves host data. Avoid deleting `logs/`: it contains
MLflow tracking/registry metadata and artifacts as well as training outputs.

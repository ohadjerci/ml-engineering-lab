# Versioning, storage, and reproduction

Versioned training records enough information to recover the historical code,
dataset, configuration, environment, and initialization. Git, DVC, and MLflow
have different responsibilities; all three are needed by the current replay tool.

## Where things are stored

Paths below are relative to the repository root with the default local setup.

| Location | Contents |
| --- | --- |
| Git history | Code, configuration, lockfile, Docker setup, and small `.dvc` dataset pointers. |
| `data/` | Downloaded/source data and local dataset storage. |
| `data/snapshots/` | Immutable snapshot directories containing files, labels, and recorded train/validation/test membership; `.dvc` pointers beside them are tracked in Git. |
| `data/.dvc-cache/` | DVC's local content-addressed cache. |
| `data/.dvc-remote/` | Default DVC remote containing uploaded dataset versions; mounted as `/dvc-remote` in containers. |
| `data/.locks/` | Coordination locks and metadata ownership journal for preparation retries. |
| `logs/train/`, `logs/eval/` | Local run outputs and checkpoints. |
| `logs/mlflow/db/mlflow.db` | MLflow experiment, run, metric, and registry metadata. |
| `logs/mlflow/artifacts/` | New MLflow run artifacts, including provenance, initial weights, checkpoints, and packaged models. |
| `logs/reproduce/` | Isolated replay checkouts and their outputs. |

The default DVC remote is a folder on the same machine, not an independent
backup. Set `DVC_LOCAL_REMOTE_DIR` to another accessible host location or use
the SSH/S3 presets for external storage. Keep dataset bytes in DVC and model
artifacts in MLflow. Provider configuration is described in
[configuration](configuration.md#ssh-and-s3).

## What happens before training

1. The project checks that relevant source and configuration are committed and
   that files inside the Docker image match the checkout. It never commits
   implementation changes for you.
2. The data module exports files, labels, class mapping, and exact split membership
   into a snapshot. A content-derived identity selects its directory name.
3. DVC tracks the snapshot and uploads its contents to the selected remote.
4. Preparation commits only allowlisted dataset metadata, including the `.dvc`
   pointer. It preserves unrelated staged work. Git push is disabled by default.
5. Training loads the snapshot and records its provenance in MLflow. A matching
   completed experiment may be reused; otherwise fitting runs.

The code commit and dataset-metadata commit can differ. Both are recorded.
You do not need a branch per dataset or experiment: many runs can reference
different commits on the same branch.

## Adding an image or changing the split

Change the source dataset or split settings, then run preparation or versioned
training again. A changed manifest or file produces a new snapshot identity.
The old snapshot stays unchanged. An identical snapshot is verified and reused.

For example, adding one image creates a local snapshot containing all the old
images plus the new image. Changing only split membership creates a new manifest
while image contents remain unchanged. Train/validation/test membership is saved
as sample lists, rather than separate copies for each split inside one snapshot.

DVC identifies files by content, so unchanged files are shared across versions
in its cache and remote. Only new or modified files and metadata need additional
remote storage. However, this implementation creates full local snapshot
directories and configures DVC with `cache.type=copy`: multiple local snapshots
can duplicate image bytes. Remote deduplication does not eliminate that local
disk cost.

## Code commits and file hashes

MLflow's `provenance/source.json` records:

- `repository`: a credential-free Git repository reference, when available.
- `revision`: the exact Git commit SHA used for the code.
- `files`: paths and SHA-256 content hashes of tracked files under `src/`,
  `configs/`, and `Dockerfile/`, plus `pyproject.toml`, `uv.lock`, and `compose.yaml`.
- `identity`: a combined fingerprint of that file mapping.

A commit message is a human description; replay uses the commit SHA. A file's
SHA-256 hash is a 64-character fingerprint of its contents. Replay retrieves
files through the Git commit and compares their hashes with the recorded values.
A hash verifies contents; it cannot reconstruct a missing file. Tests, README
files, datasets, and logs are outside this source-hash selection.

MLflow also stores `config.json`, `data_version.json`, `environment.json`,
`fingerprint.json`, `uv.lock`, dataset pointers, and initialization under
`provenance/`. Its `code/` artifact contains `src/`; it is not a complete repository
archive, and the current replay implementation does not restore code from it.

## Reusing a run versus reproducing it

Normal versioned training can reuse a matching completed run when its required
artifacts remain available. Matching considers dataset identity/name, effective
training settings, source contents, dependency lockfile, environment, seed, and
initialized weights. A metadata-only Git commit does not by itself force fitting.

```bash
# Prepare and version data without GPU training.
just prepare-data
just prepare-data-pcb
# Force fitting instead of reusing a matching run.
just train reproducibility.force_retrain=true
# Explore uncommitted edits without complete versioned replay support.
just train-dev
just train-pcb-dev
```

Independent preparation commits only dataset metadata and can run while source
edits are uncommitted. Versioned training requires committed relevant source.

## What just reproduce does

Start MLflow and supply the original 32-character MLflow run ID:

```bash
docker compose up -d mlflow
just reproduce ORIGINAL_MLFLOW_RUN_ID
```

1. The current preparation container downloads the original run's provenance from
   MLflow. Missing required records cause an error.
2. It clones the local Git repository into a new
   `logs/reproduce/<run-id>-<timestamp>-<process-id>/checkout/` directory. If a required
   commit is unavailable locally, it tries fetching it from the recorded repository.
3. It checks out the code commit in detached mode and verifies source-file hashes.
   No historical branch name is required. Your current checkout stays untouched.
4. It retrieves the `.dvc` pointer from the separate dataset-metadata commit and
   checks its recorded hash. The pointer identifies the exact historical dataset.
5. It restores configuration and downloads the original initial model state, or
   the original starting checkpoint for a resumed run, from MLflow.
6. It builds a separate Docker image using the historical Dockerfile and lockfile,
   then executes the historical replay entry point.
7. That code pulls the referenced dataset from DVC storage and verifies its
   manifest identity, split membership, dataset name, and class mapping.
8. It checks Python/package versions and initialization against the original run,
   then trains again. Reuse is disabled. The new MLflow run links to the original;
   replay does not commit or push dataset metadata.

Replay adjusts output paths for isolation and uses current transport settings and
credentials. The default local DVC remote is shared with the original checkout;
it is not copied into the replay checkout. Additional mounts/provider settings can
be supplied with `REPRODUCTION_COMPOSE_OVERRIDE`; see
[configuration](configuration.md#ssh-and-s3).

Historical code must support the replay entry point. Missing Git commits, DVC
objects, or MLflow artifacts cause failure rather than substitution with current
code or data. Python/package mismatches fail; other runtime differences warn.
Restoring inputs does not guarantee bit-for-bit numerical equality across hardware
and nondeterministic operations.

## Retention and cleanup

Keep referenced Git commits, DVC objects, and MLflow metadata/artifacts together.
Deleting a branch does not necessarily delete its commits immediately, but an
unreferenced commit can eventually be garbage-collected. Preserve important
revisions with durable Git references, such as tags, and an accessible repository.

Deleting the DVC cache can be recovered from an intact remote. Existing source
files/snapshots can also be prepared again. Deleting the remote removes its stored
copy; metadata and hashes alone cannot recover lost dataset bytes. Deleting
`logs/` also deletes MLflow's local database and artifacts with the default setup,
not just disposable console logs. Preserve historical MLflow storage described in
[MLflow storage](mlflow.md) and [migration history](migration.md).

## Permission recovery

For a fresh clone, create persistent directories as your normal user before
starting Docker, following the [quick start](../README.md#quick-start). Export
`LOCAL_UID` and `LOCAL_GID` in each shell that runs Compose, or configure them in
your ignored `.env`. MLflow has its separate server-user ownership step.

If Docker recreated missing directories as root, `mkdir -p` will not repair their
ownership. After recreating the quick-start directories, repair preparation and
training storage with the following commands. They recursively change ownership
of existing dataset/output files, so use the account intended to own that storage.

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm --no-deps --user root --entrypoint sh \
  -e LOCAL_UID -e LOCAL_GID train \
  -c 'chown -R "$LOCAL_UID:$LOCAL_GID" /dvc-remote /workspace/data /workspace/logs/train /workspace/logs/eval'
docker compose run --rm --no-deps --user root --entrypoint sh \
  -e LOCAL_UID -e LOCAL_GID prepare_dataset \
  -c 'chown -R "$LOCAL_UID:$LOCAL_GID" /workspace/logs/prepare_dataset /workspace/logs/reproduce'
```

This covers the mounted `DVC_LOCAL_REMOTE_DIR` as well. SSH/S3 permissions and
credentials need their own configuration. Run `just prepare-data` (or
`just prepare-data-pcb`) again after repair. A non-writable remote can appear as
the generic `RuntimeError: dvc push failed`; the command wrapper suppresses
transport output to avoid logging credentials.

## GitHub Actions and CI

The [CI workflow](../.github/workflows/ci.yml) runs on pushes to all branches,
pull requests targeting `main`, and manual dispatches. Hosted Ubuntu runners use
Python 3.12 and the locked dependencies. They check the lockfile and installed
dependencies, run CPU pytest without sweeps, run Ruff lint/format and core MyPy,
and enforce line/branch coverage floors. GPU tests skip on CPU runners; no hosted
GPU training or automatic model deployment is configured.

The Actions summary contains four badges: Coverage, PR coverage, Mypy, and Ruff.
Numeric quality, coverage, and JUnit artifacts are retained for seven days. A
separate trusted default-branch workflow validates the numeric report and updates
one PR comment without executing PR code with its write token. Local Docker
verification and remote CI are separate checks. See
[development and CI](development.md) for commands and report details.

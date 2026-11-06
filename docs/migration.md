# Historical tracking migration

This guide concerns **existing file-store installations**. A fresh checkout only
needs the [README quick start](../README.md#quick-start); do not migrate an empty
store or rerun migration against an already populated target database.

## What was preserved

The recorded 2026-11-06 migration used MLflow 3.18.0. It backed up the original
file store to `logs/mlflow/backups/20261106T152136Z/mlruns.tar.gz` with a file
manifest, and imported two experiments, two runs, and 1,827 metric records into
SQLite. These are historical verification results, not checks repeated by this
documentation update. Original files remain in `logs/mlflow/mlruns/`.

Supported migration copies metadata but retains artifact URIs. An old experiment
such as `minst_project` can therefore retain local artifact paths even though its
metadata is now in SQLite. Intermediate experiments such as `minst_project_http`
may also remain. The current default `project_http` creates new runs with
server-proxied artifact locations. Do not infer the artifact scheme from a name.

## Equivalent migration on another installation

Run these commands from the **training repository root**, after confirming that
all training/evaluation writers are stopped. Keep a verified backup and use an
empty SQLite target. Review permissions before starting.

```bash
docker compose stop mlflow
mkdir -p logs/mlflow/backups logs/mlflow/db logs/mlflow/artifacts
tar -czf "logs/mlflow/backups/mlruns-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
  -C logs/mlflow mlruns
```

After inspecting the archive, run the migration with the locked image:

```bash
docker compose run --build --rm --no-deps -e MLFLOW_ALLOW_FILE_STORE=true \
  -v "$PWD/logs/mlflow:/workspace/logs/mlflow" tests \
  uv run --locked mlflow migrate-filestore \
  --source /workspace/logs/mlflow/mlruns \
  --target sqlite:////workspace/logs/mlflow/db/mlflow.db
```

This uses the `tests` service and consequently requires its configured NVIDIA
runtime, even though metadata migration itself is CPU work. Check the migration
summary against the source experiments, runs, and metrics. Then restart:

```bash
docker compose up --build -d mlflow
```

Read back historical metrics and inspect a new experiment's artifact URI before
resuming normal work. A reliable metadata migration does not make old local model
artifacts downloadable by an independent client. Republish an evaluated model to
the new registry for that purpose.

If migration fails validation, retain the original store and backup separately;
start with a fresh SQLite target only after deciding how to retain access to the
old history. Never delete old artifacts as part of this procedure.

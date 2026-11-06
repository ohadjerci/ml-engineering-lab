# MLflow setup and shared connectivity

Run commands from the **training repository root**. For a fresh installation,
complete the [README quick start](../README.md#quick-start), including storage
permissions, before starting the server.

## Start, inspect, and stop

```bash
docker compose up --build -d mlflow
docker compose ps mlflow
docker compose logs -f mlflow
```

The `mlflow` and `train` services use the `training` profile; naming a service
activates it directly. Plain `docker compose up` starts only the test service.
The MLflow health check gates training startup. `restart: unless-stopped` keeps
the server available after training exits.

Stop the server explicitly when it is no longer needed:

```bash
docker compose stop mlflow
```

`docker compose down` also stops it and attempts to remove the shared network.
Avoid doing that while inference clients need the server. Host logs remain on disk.

## Addresses and storage

| Client or storage | Address or host path |
| --- | --- |
| Host browser/client | `http://localhost:50001` |
| Containers on the shared network | `http://mlflow:5000` |
| Docker network owned by this project | `vision-mlflow` |
| SQLite tracking metadata and registry | `logs/mlflow/db/mlflow.db` |
| HTTP-served artifacts | `logs/mlflow/artifacts/` |
| Legacy local artifact tree | `logs/mlflow/mlruns/` |

The server runs MLflow 3.18.0 with an explicit SQLite backend,
`--serve-artifacts`, and `--artifacts-destination`. Its only storage mount is
`logs/mlflow/` at `/workspace/logs/mlflow`. New experiments use `mlflow-artifacts:/`
URIs, so clients upload and download through HTTP rather than opening server files.
The published host port binds only to `127.0.0.1`.

Inside a container, `localhost` means that container. The training service sets
`MLFLOW_TRACKING_URI=http://mlflow:5000`. The separate inference Compose project
joins the existing network and uses the same URL. It needs neither the training
checkout nor a mount of this project's logs.

## Storage permissions

The image creates a non-root `mlcycle` user; its UID need not match your host UID.
The README's one-time commands assign only the new storage directories to that
user. On an existing installation, inspect ownership and arrange write access
before running them. Changing top-level directory ownership does not repair
permissions of existing nested files. Do not solve this by making all files
world-writable or deleting old experiments.

Training persists only `logs/train/` and `logs/eval/`; it cannot directly access
the server's database or artifacts. Custom task names/output paths need their own
mount if their outputs must survive container removal.

## Experiments and models

`project_http` is the current default experiment. It groups tracked runs;
`MNIST` is the registered model name and groups numbered model versions. These
are different objects. `best_model` is an alias for one registered version.

Older experiment names such as `minst_project` and `minst_project_http` may remain
in an existing database. Names alone do not determine artifact access: check the
experiment's stored artifact URI. Historical local paths are not automatically
converted when changing a server URI. See [migration notes](migration.md) and
[registration behavior](architecture.md#registration-and-inference).

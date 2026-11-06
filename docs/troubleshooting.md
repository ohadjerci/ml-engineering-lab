# Troubleshooting

Run commands from the **training repository root**.

## Docker cannot start the training container

The `train` and `tests` services request NVIDIA GPUs. Check the host driver and
NVIDIA Container Toolkit before retrying. Choosing `trainer=cpu` does not remove
that Compose device request. MLflow itself can run without a GPU.

The image contains the CUDA 12.6 development toolkit and training dependencies.
Cold builds are large and can exhaust disk space; inspect Docker storage before
assuming dependency resolution is broken. See [development](development.md).

## Permission denied under logs

The container's `mlcycle` user may have a different UID from the host user. Follow
[storage permissions](mlflow.md#storage-permissions). For an existing checkout,
check nested file ownership as well as directory ownership. The README setup
commands are for new directories; they do not recursively change existing files.

## MLflow is unreachable or rejects a host

```bash
docker compose ps mlflow
docker compose logs --tail 50 mlflow
```

Wait for `healthy`. Host access uses `http://localhost:50001`; containers use
`http://mlflow:5000`. The server explicitly allows `mlflow:5000`, `localhost:*`,
and `127.0.0.1:*`. A different server hostname needs an appropriate allowed-host
configuration. Do not use a host-local URL inside the training container.

## The experiment or model is missing

New runs default to `project_http`. An old experiment may still appear with a
previous name. The registry model is separately named `MNIST`.

A model is published only after testing succeeds, with MLflow logging and
`registry.enabled=true`. `test=false` skips publication; standalone evaluation
needs `logger=mlflow`. Read the failure output if artifact export or registration
fails; the presence of tracked metrics alone does not prove model publication.

For the validation-versus-test selection discrepancy, see
[model selection mismatch](architecture.md#model-selection-mismatch).

## A historical artifact cannot be downloaded remotely

Migration retained the original local artifact URI. New server options do not
rewrite it. Keep the legacy files and publish a new portable model version for
standalone inference. See [migration](migration.md).

## Hydra rejects an override

Use the full `uv run --locked python src/train.py` command after the `train`
service name. Existing YAML keys use `key=value`; newly added trainer keys use
`+key=value`. Examples are in [configuration](configuration.md#hydra-overrides).
Checkpoint resume/evaluation must use the same model architecture as the saved
checkpoint, and paths must be container paths rather than arbitrary host paths.

## Model results are poor

A two-batch smoke run checks integration, not useful digit recognition. Inspect
training and validation curves, train longer as appropriate, and use input images
matching the inference preprocessing. Do not treat softmax confidence as a
calibrated correctness guarantee.

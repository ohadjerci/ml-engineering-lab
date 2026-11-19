# All project Python runs in Docker Compose. Compatible with just 1.21+.
set positional-arguments
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# List recipes.
default:
    @just --list

# Build the locked project image.
build:
    docker compose build tests

# Train MNIST; append Hydra overrides (e.g. trainer=cpu trainer.max_epochs=2).
train *args:
    docker compose run --build --rm train uv run --locked python src/train.py "$@"

# Train PCB using the configured data/REALIAD directory.
train-pcb *args:
    docker compose run --build --rm train uv run --locked python src/train.py experiment=pcb "$@"

# Validate an explicit MNIST checkpoint (path inside the container).
validate checkpoint *args:
    docker compose run --build --rm train uv run --locked python src/validate.py "ckpt_path=$1" "${@:2}"

# Test an explicit MNIST checkpoint.
eval checkpoint *args:
    docker compose run --build --rm train uv run --locked python src/eval.py "ckpt_path=$1" "${@:2}"

# Validate a PCB checkpoint; append data.split.assignments_path=... if desired.
validate-pcb checkpoint *args:
    docker compose run --build --rm train uv run --locked python src/validate.py experiment=pcb "ckpt_path=$1" "${@:2}"

# Test a PCB checkpoint. Add logger=mlflow to publish it.
eval-pcb checkpoint *args:
    docker compose run --build --rm train uv run --locked python src/eval.py experiment=pcb "ckpt_path=$1" "${@:2}"

# Start the persistent VS Code debug container.
debug-up:
    docker compose up --build -d debug

# Stop the debug container (MLflow remains running).
debug-down:
    docker compose stop debug

# Stop all running containers whose names contain mlflow, across checkouts.
mlflow-down:
    container_ids=$(docker ps --quiet --filter name=mlflow); if [[ -n "$container_ids" ]]; then mapfile -t containers <<< "$container_ids"; docker stop "${containers[@]}"; else echo "No running MLflow containers."; fi

# Open a shell in the active debug container.
bash:
    docker compose exec debug bash

# Debug MNIST training. Set breakpoints, run this, then attach VS Code.
debug *args:
    docker compose exec debug /opt/venv/bin/python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client src/train.py "$@"

# Debug MNIST validation of a checkpoint (container path).
debug-validate checkpoint *args:
    docker compose exec debug /opt/venv/bin/python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client src/validate.py "ckpt_path=$1" "${@:2}"

# Debug MNIST testing of a checkpoint.
debug-eval checkpoint *args:
    docker compose exec debug /opt/venv/bin/python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client src/eval.py "ckpt_path=$1" "${@:2}"

# Debug PCB training using the configured data/REALIAD directory.
debug-pcb *args:
    docker compose exec debug /opt/venv/bin/python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client src/train.py experiment=pcb "$@"

# Debug PCB validation of a checkpoint.
debug-validate-pcb checkpoint *args:
    docker compose exec debug /opt/venv/bin/python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client src/validate.py experiment=pcb "ckpt_path=$1" "${@:2}"

# Debug PCB testing of a checkpoint; add logger=mlflow to publish it.
debug-eval-pcb checkpoint *args:
    docker compose exec debug /opt/venv/bin/python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client src/eval.py experiment=pcb "ckpt_path=$1" "${@:2}"

# Run repository tests, excluding sweeps. Additional pytest arguments are forwarded.
test *args:
    docker compose run --build --rm tests uv run --locked pytest --ignore=tests/test_sweeps.py "$@"

# Reproduce hosted CPU test selection.
test-cpu *args:
    docker compose run --build --rm -e CUDA_VISIBLE_DEVICES="" tests uv run --locked pytest --ignore=tests/test_sweeps.py -ra "$@"

# Run focused synthetic PCB tests.
test-pcb *args:
    docker compose run --build --rm tests uv run --locked pytest tests/test_pcb.py "$@"

lint:
    docker compose run --build --rm tests uv run --locked ruff check .

format-check:
    docker compose run --build --rm tests uv run --locked ruff format --check .

mypy:
    docker compose run --build --rm tests uv run --locked mypy

# Run the local quality checks and tests; remote CI remains a separate verification.
check: lint format-check mypy test

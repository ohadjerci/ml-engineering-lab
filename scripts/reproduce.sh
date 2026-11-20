#!/usr/bin/env bash
set -euo pipefail
run_id=${1:?Usage: bash scripts/reproduce.sh MLFLOW_RUN_ID}
if [[ ! "$run_id" =~ ^[[:xdigit:]]{32}$ ]]; then
    echo 'Expected a 32-character MLflow run ID' >&2
    exit 2
fi
repository=$(git rev-parse --show-toplevel)
cd "$repository"
relative="logs/reproduce/${run_id}-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p logs/reproduce
export DVC_LOCAL_REMOTE_DIR
DVC_LOCAL_REMOTE_DIR=$(realpath -m "${DVC_LOCAL_REMOTE_DIR:-$repository/data/.dvc-remote}")
current_flags=(-f "$repository/compose.yaml")
historical_flags=()
if [[ -n "${REPRODUCTION_COMPOSE_OVERRIDE:-}" ]]; then
    override=$(realpath "$REPRODUCTION_COMPOSE_OVERRIDE")
    current_flags+=(-f "$override")
    historical_flags+=(-f "$override")
fi
docker compose "${current_flags[@]}" run --build --rm prepare_dataset --reproduce-prepare "$run_id" "/workspace/$relative"
checkout="$repository/$relative/checkout"
printf 'services:\n  train:\n    image: mlcycle:reproduce-%s\nnetworks:\n  mlflow:\n    external: true\n' "$run_id" > "$checkout/replay.compose.yaml"
# This builds and executes the recorded code and lockfile, not the current source.
docker compose --project-name "reproduce-${run_id:0:12}" \
    --project-directory "$checkout" -f "$checkout/compose.yaml" -f "$checkout/replay.compose.yaml" "${historical_flags[@]}" \
    run --build --rm --no-deps train uv run --locked python \
    src/data/prepare_dataset.py --reproduce-run /repo/replay.json

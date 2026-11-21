#!/usr/bin/env bash
# One-time local setup; project Python stays inside Docker Compose.
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo 'Usage: bash scripts/setup.sh [--repair-permissions]'
    echo 'Create storage, record host IDs in .env, build, and start healthy MLflow.'
    echo 'Optional repair recursively changes dataset and training-output ownership.'
    exit 0
fi
repair=false
if [[ ${1:-} == --repair-permissions && $# == 1 ]]; then
    repair=true
elif (( $# )); then
    echo 'Usage: bash scripts/setup.sh [--repair-permissions]' >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
for tool in git docker; do
    command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
docker compose version >/dev/null
docker info >/dev/null
if [[ $(id -u) == 0 ]]; then
    echo 'Run setup as your normal host user, without sudo.' >&2
    exit 1
fi

# A child script cannot export variables back into its caller's shell.
# Persist only these two settings; preserve all other .env contents.
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
if [[ -L .env || ( -e .env && ! -f .env ) ]]; then
    echo '.env must be a regular file, not a symlink or directory.' >&2
    exit 1
fi
temporary=$(mktemp .env.setup.XXXXXX)
trap 'rm -f "$temporary"' EXIT
if [[ -f .env ]]; then
    awk '!/^[[:space:]]*(export[[:space:]]+)?LOCAL_(UID|GID)[[:space:]]*=/' .env > "$temporary"
fi
printf '\nLOCAL_UID=%s\nLOCAL_GID=%s\n' "$LOCAL_UID" "$LOCAL_GID" >> "$temporary"
chmod 600 "$temporary"
mv "$temporary" .env

# Let Compose resolve shell/.env interpolation without sourcing .env as code.
remote=$(docker compose config --environment | sed -n 's/^DVC_LOCAL_REMOTE_DIR=//p')
remote=${remote:-./data/.dvc-remote}
mkdir -p "$remote" data/.dvc-cache data/.locks data/snapshots configs/local \
    logs/train logs/eval logs/prepare_dataset logs/reproduce \
    logs/mlflow/db logs/mlflow/artifacts

echo 'Building the locked training/MLflow image...'
docker compose build train

if "$repair"; then
    docker compose run --rm --no-deps --user root --entrypoint sh \
        -e LOCAL_UID -e LOCAL_GID prepare_dataset -ec \
        'chown -R "$LOCAL_UID:$LOCAL_GID" /dvc-remote /workspace/data /workspace/logs/prepare_dataset /workspace/logs/reproduce'
    docker compose run --rm --no-deps --user root --entrypoint sh \
        -e LOCAL_UID -e LOCAL_GID train -ec \
        'chown -R "$LOCAL_UID:$LOCAL_GID" /workspace/logs/train /workspace/logs/eval'
fi

docker compose run --rm --no-deps --entrypoint sh \
    -v "$PWD/logs/train:/setup/train" -v "$PWD/logs/eval:/setup/eval" prepare_dataset -ec '
    for dir in /repo /dvc-remote /workspace/data /workspace/data/.dvc-cache /workspace/data/.locks /repo/data/snapshots /workspace/logs/prepare_dataset /workspace/logs/reproduce /setup/train /setup/eval; do
        if ! test -w "$dir"; then
            echo "Not writable: $dir. See docs/setup.md#permission-recovery." >&2
            exit 1
        fi
    done
'

# MLflow owns its storage independently of the training user's host IDs.
docker compose run --rm --no-deps --user root --entrypoint sh mlflow -ec \
    'chown mlcycle:mlcycle /workspace/logs/mlflow /workspace/logs/mlflow/db /workspace/logs/mlflow/artifacts'
echo 'Starting MLflow and waiting for its health check...'
docker compose up -d --wait --wait-timeout 120 mlflow

echo 'Setup complete. MLflow: http://localhost:50001'
if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1 || ! git var GIT_COMMITTER_IDENT >/dev/null 2>&1; then
    echo 'Before versioned training, configure Git identity:'
    echo '  git config user.name "Your Name"'
    echo '  git config user.email "you@example.com"'
fi
echo 'Train MNIST: just train'
echo 'Without just: docker compose run --build --rm train'

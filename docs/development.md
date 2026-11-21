# Development, tests, and CI

Run commands from the **training repository root**, using Docker Compose for
project code and tools. The image pins Python 3.12, uv 0.13.7, PyTorch 2.6.0,
torchvision 0.21.0, and CUDA 12.6; no host uv environment is required.

## Local checks

The test service requests NVIDIA GPUs. Tests themselves include bounded CPU and
GPU training, two-process CPU DDP, checkpoint resume, evaluation, data, configuration,
and model-promotion checks. Dataset-dependent tests may download MNIST.

```bash
docker compose run --build --rm tests uv run --locked pytest --ignore=tests/test_sweeps.py -ra
docker compose run --rm tests uv run --locked ruff check .
docker compose run --rm tests uv run --locked ruff format --check .
docker compose run --rm tests uv run --locked mypy
```

For configuration-only verification and discovery without training:

```bash
docker compose run --rm tests uv run --locked pytest tests/test_configs.py
docker compose run --rm tests uv run --locked pytest --collect-only
```

The plain pytest command also includes sweep tests. Sweeps are excluded from the
routine verification command and CI unless deliberately requested. GPU tests skip
when CUDA is unavailable. To reproduce CI's CPU selection on this GPU-enabled
Compose setup:

```bash
docker compose run --build --rm -e CUDA_VISIBLE_DEVICES= tests \
  uv run --locked pytest --ignore=tests/test_sweeps.py -ra
```

Coverage is collected with branches enabled and emits XML and JSON. Normal
one-off test-container reports disappear with `--rm`; mount a writable checkout
or copy results out if you need local report files. Do not confuse the combined
pytest headline coverage with the separate line and branch percentages in CI.

## CI gates and reporting

[CI](../.github/workflows/ci.yml) runs on pushes to all branches, PRs targeting
`main`, and manual dispatch. The hosted Ubuntu CPU job checks the lockfile,
installs locked dependencies, checks dependency consistency, and runs tests.
Ruff lint, Ruff format, and configured core MyPy checks remain independent steps.
The hosted job does not build the CUDA image or verify GPU execution.

The reusable local action
[quality-report](../.github/actions/quality-report/action.yml) enforces the measured
CPU **line and branch** floors in
[coverage-baseline.json](../.github/coverage-baseline.json). Both must pass.
Review a baseline change with its tests and source scope; do not lower it simply
to hide a regression. Missing or inconsistent coverage reports fail verification.

The Actions summary contains exactly four adjacent badges: **Coverage**,
**PR coverage**, **Mypy**, and **Ruff**. They show overall line/branch percentages,
changed executable-line coverage, MyPy error/file counts, and Ruff lint/format
counts. PR coverage is informational; pushes and PRs without changed executable
source lines show `N/A`, rather than a fabricated percentage.

A separate [comment workflow](../.github/workflows/ci-comment.yml) runs trusted
default-branch code after a PR workflow finishes. It validates a numeric artifact
and creates or updates one bot comment with the same four badges and no extra
text. PR code is never executed with the comment-writing token. This workflow
and action must be present on the default branch before comments can work.
No PR comment is posted for a branch push alone.

Artifacts `coverage-cpu` (XML/JSON), `quality-report` (numeric statistics), and
`pytest-cpu` (JUnit) are retained for seven days. Dependency/setup failures may
leave reports unavailable; inspect the failed step logs. Branch protection is a
repository-administrator setting and is not enabled by adding a workflow.

## Scope of quality tools

Ruff uses a 99-character target and checks common errors and import ordering;
outputs, datasets, build artifacts, caches, and notebooks are excluded. MyPy checks
the training/evaluation core paths listed in `pyproject.toml`. Full-source
`mypy src` is a backlog audit, not the enforced gate: retained detection helpers
and the unfinished cluster plugin still have deferred typing work. Historical
error counts are not current guarantees.

The report action also has standard-library unit tests, run by CI. To run them:

```bash
docker compose run --build --rm tests uv run --locked python -m unittest discover \
  -s .github/actions/quality-report -p 'test_*.py'
```

## Dependencies and debugging

After changing dependencies, regenerate and commit `uv.lock` through the tools
service, then rerun checks. The maintenance service does not request a GPU.

```bash
LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" docker compose run --build --rm lock uv lock
docker compose run --build --rm tests uv lock --check
docker compose run --rm tests uv pip check
```

The image copies source at build time. Rebuild after edits, or open a shell with
the checkout mounted for development:

```bash
docker compose run --build --rm -v "$PWD:/workspace" -w /workspace tests bash
```

Inside that container, `uv run --locked pytest tests/test_configs.py --pdb` is a
small debugging entry point. The environment lives at `/opt/venv`, outside the
checkout mount. Keep datasets, credentials, checkpoints, and outputs out of commits.

## VS Code breakpoints in Docker

Install the VS Code **Python** and **Python Debugger** extensions and open this
repository root. Debug commands use the normal GPU, training duration, logging,
checkpoints, and registry publication settings. The NVIDIA Container Toolkit is
required, including when selecting a CPU trainer. Prepare writable data and output
directories as described in the README and [storage guidance](mlflow.md#storage-permissions).

1. Run `just debug-up` to build and start the persistent container and MLflow.
2. Set a breakpoint in `src/train.py`, for example on the first line inside `train`.
3. Run `just debug` in a terminal. It waits before executing project code.
4. Select **Attach to Docker training** in VS Code's Run and Debug panel and press
   **F5**. Execution pauses at your breakpoint; inspect variables, step, or continue.
5. After the process exits, run `just debug` and attach again for another session.
   Use `just debug-down` when finished; MLflow remains running.

Source and configuration changes are mounted live, so restart the debug command
after edits. Rebuild with `just debug-up` after dependency changes. `just bash`
opens a shell in the active container. Only one debug process can listen at a time;
finish or interrupt the current command before starting another. If you detach
while leaving the process running, wait for it to exit or interrupt its terminal.

Hydra overrides are forwarded normally. For a short run, or to hit breakpoints
inside dataset-loading code in the main process, use:

```bash
just debug trainer.max_epochs=1 data.num_workers=0
# Optional CPU trainer (the container still requests NVIDIA GPU access):
just debug trainer=cpu trainer.max_epochs=1 data.num_workers=0
```

Validation and test commands require a checkpoint path inside the container:

```bash
just debug-validate /workspace/logs/train/runs/RUN/checkpoints/last.ckpt
just debug-eval /workspace/logs/train/runs/RUN/checkpoints/last.ckpt
```

PCB uses its YAML paths, just like MNIST. The shared `./data:/workspace/data`
mount provides `data/REALIAD/pcb.json` and `data/REALIAD/pcb/`:

```bash
just debug-up
just debug-pcb data.num_workers=0
just debug-validate-pcb /workspace/logs/train/runs/RUN/checkpoints/last.ckpt
just debug-eval-pcb /workspace/logs/train/runs/RUN/checkpoints/last.ckpt
```

PCB commands accept additional Hydra overrides, including
`data.split.assignments_path=...`. Evaluation keeps its usual logging/publication
behavior; add `logger=mlflow` when publication is intended.

The VS Code configuration maps the local checkout to `/workspace`. Port 5678 is
published only on `127.0.0.1`; stop the vision-inference debug container first if
it uses the same host port. This attachment workflow does not enable the separate
legacy Hydra `debug=` presets.
# Dataset-versioning checks

Core dataset tests use tiny fixtures, temporary Git repositories/local DVC remotes,
and isolated SQLite MLflow tracking stores. They do not contact paid storage or
need a GPU. Existing Compose test services still reserve GPUs; run the same tests
on hosted CPU CI or an appropriate local CPU Compose override when no NVIDIA
runtime is available. Existing test fixtures explicitly disable automatic versioning;
focused versioning tests enable it only in temporary repositories.

```bash
just test tests/test_datamodules.py tests/test_model_registry.py tests/test_pcb.py
just prepare-data --cfg job
just check
```

Relevant uncommitted source intentionally fails normal versioned training. For
breakpoint debugging before committing use `versioning.enabled=false`; versioned
training must execute committed code matching its image. Dirty read-only source
mounts are not accepted as reproducible code snapshots.

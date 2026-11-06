# Repository Guidelines

## Project Structure & Module Organization

`src/train.py` and `src/eval.py` are Hydra entry points. Place dataset loaders in `src/data/`, Lightning modules in `src/models/`, network architectures in `src/models/components/`, and shared helpers in `src/utils/`. Keep callbacks in `src/callbacks/`.

`configs/` contains YAML groups for data, models, trainers, callbacks, loggers, and experiments. Tests live in `tests/`, with shared fixtures in `tests/conftest.py`. Use `notebooks/` for exploration and `inference/` for inference work. Downloaded data and generated outputs belong in ignored `data/` and `logs/` directories.

## Default Training & Logging Flow

Hydra composes the configuration, then `src/train.py` instantiates the data module, model, callbacks, loggers, and Lightning Trainer. The MNIST data module downloads missing data, converts images to normalized tensors, combines the original train/test datasets, and reproducibly splits them into 55,000 training, 5,000 validation, and 10,000 test samples. The default batch size is 128.

The custom ResNet classifies 10 digits using cross-entropy loss and Adam. Training and validation log epoch-level loss, accuracy, precision, recall, and F1. Validation loss controls learning-rate reduction, early stopping, and best-checkpoint selection. Checkpoints are saved every epoch, including `last.ckpt`, under `logs/train/runs/<timestamp>/checkpoints/`. After fitting, the best checkpoint is evaluated on the test split.

The MLflow logger sends parameters, metrics, and artifacts through HTTP. The server at `http://localhost:50001` owns `logs/mlflow/db/mlflow.db` (SQLite tracking/registry) and `logs/mlflow/artifacts/` (proxied artifacts). Containers use `http://mlflow:5000` on the shared `vision-mlflow` network. Training mounts only host `logs/train/` and `logs/eval/` for outputs; the separate inference application never mounts training files. Ensure storage directories are writable by the container's non-root `mlcycle` user.

The Hydra `registry` group controls publication after successful testing. Training publishes its best-validation-loss checkpoint; standalone evaluation publishes its evaluated checkpoint when `logger=mlflow` is selected. The current implementation exports a self-contained TorchScript network, registers it as `MNIST`, and moves `best_model` when finite `test/loss` strictly improves (ties keep the incumbent). This alias-promotion rule differs from the agreed validation-based selection requirement; see `docs/architecture.md#model-selection-mismatch`. Publishers run sequentially. Inference resolves the alias to an exact numeric version before downloading. Do not swallow publication errors or reintroduce imports from the training checkout into inference.

New runs use `project_http` with server-proxied artifact URIs. Original file history remains in `logs/mlflow/mlruns/`, with a verified backup under `logs/mlflow/backups/`; its metadata was migrated using MLflow 3.18's `migrate-filestore`. Migration preserves historical artifact paths rather than converting them to HTTP URIs. Preserve these files and back up tracking data before future migrations.

## Build, Test, and Development Commands

When asked to run project code, always use Docker Compose, including training, evaluation, tests, and diagnostic scripts. Do not run project Python or uv commands directly on the host unless explicitly requested.

Run commands from the repository root:

- `docker compose build tests`: build the CUDA image with pinned uv and locked dependencies.
- `docker compose run --build --rm tests uv run pytest tests/test_configs.py`: verify configuration instantiation without training.
- `docker compose run --build --rm tests uv run pytest tests/test_train.py`: run bounded CPU/GPU training, distributed training, and checkpoint-resume tests.
- `docker compose run --build --rm tests uv run pytest --collect-only`: check test discovery and imports.
- `docker compose run --build --rm tests uv run pytest`: run the full suite, including bounded training and sweep tests.
- `docker compose run --build --rm train`: launch default GPU training and automatically start the background MLflow UI.
- `docker compose run --build --rm train uv run --locked python src/train.py trainer=cpu trainer.precision=32-true`: launch CPU training only when intended; pass other Hydra overrides using the same explicit-command pattern.
- `docker compose up --build -d mlflow`: start the UI alone to inspect existing runs.
- `docker compose logs -f mlflow`: follow UI service logs.
- `docker compose stop mlflow`: stop the UI; `docker compose down` also stops it without deleting host logs.

The `train` and `mlflow` services belong to the `training` profile; targeting them directly activates them. Default Compose startup runs only `tests`, independently of the UI. The `train` and `tests` services request NVIDIA GPUs; the host needs NVIDIA Container Toolkit even when overriding the trainer to CPU. Preserve Python 3.12, PyTorch 2.6.0, torchvision 0.21.0, and CUDA 12.6 compatibility. Update `uv.lock` with uv whenever dependencies change.

## Continuous Integration

`.github/workflows/ci.yml` runs Python 3.12 CPU checks on pushes to all branches, PRs targeting `main`, and manual dispatches. Reuse an existing matching run; use the manual trigger if verification is missing, without changing triggers or creating a PR solely to run CI.

The hosted CPU job checks the lockfile with `uv lock --check`, installs with `uv sync --locked`, verifies dependencies with `uv pip check`, and runs pytest without sweeps. Preserve independent Ruff lint/format and configured core MyPy steps, branch-enabled XML/JSON coverage, and seven-day `pytest-cpu` JUnit artifacts. The reusable `.github/actions/quality-report` action enforces the measured line and branch floors in `.github/coverage-baseline.json`. Full-source `mypy src` remains a backlog audit, not the enforced command; no self-hosted GPU job is configured.

The Actions summary and PR comment must contain exactly four adjacent badges labeled `Coverage`, `PR coverage`, `Mypy`, and `Ruff`, with no extra text. Overall coverage reports line and branch percentages; PR coverage measures changed executable lines and is `N/A` when unavailable or not applicable. A trusted default-branch `ci-comment.yml` workflow validates the numeric report artifact and creates/updates one bot comment after PR runs. Never execute untrusted PR code with its write-scoped token; comment publishing requires those files on the default branch.

Direct uv commands in hosted workflows run on GitHub Actions runners; local reproduction must use Docker Compose. Reproduce CPU selection with `docker compose run --build --rm -e CUDA_VISIBLE_DEVICES="" tests uv run --locked pytest --ignore=tests/test_sweeps.py -ra`. GPU tests skip on CPU runners. Retain local Compose GPU configuration and distinguish CPU coverage baselines from GPU verification. See `docs/development.md` for report artifacts and checks.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Follow existing type annotations and Hydra `_target_` configuration patterns. Match configuration names to modules, such as `data/mnist.yaml`. Ruff enforces lint, import ordering, and formatting with a 99-character line target. MyPy checks the core training path configured in `pyproject.toml`; avoid broad typing suppressions. Run quality tools through Docker Compose, for example `docker compose run --build --rm tests uv run --locked ruff check .`, `docker compose run --build --rm tests uv run --locked ruff format --check .`, and `docker compose run --build --rm tests uv run --locked mypy`.

## Testing Guidelines

Use pytest files named `test_*.py` and functions named `test_*`. Reuse fixtures and mark expensive integration checks with `@pytest.mark.slow`. CI enforces measured CPU line and branch coverage floors from `.github/coverage-baseline.json`. Report actual commands, counts, failures, and skips rather than reusing historical results; preserve meaningful assertions.

## Commit & Pull Request Guidelines

History uses concise imperative subjects, such as “Restore Hydra training configuration.” Keep commits focused. PR descriptions should explain behavior changes, affected configurations, validation results, and relevant issue links. Distinguish configuration verification from executed training.

For every “commit & push” request, complete both local verification and remote GitHub Actions verification. A successful push or local test run alone does not complete this workflow. The request includes running the existing manual CI workflow for the pushed branch when needed.

1. Inspect the branch, upstream, working tree, and diff. Stage only the intended changes; preserve unrelated changes and exclude credentials and generated artifacts.
2. Before committing, run `docker compose run --build --rm tests uv run pytest --ignore=tests/test_sweeps.py`. Exclude sweeps unless explicitly requested. For Python, dependency, or quality-configuration changes, also run the Compose Ruff lint, formatting, and core MyPy commands documented above. Check `uv lock --check` and `uv pip check` through Compose for dependency changes; validate workflow edits with actionlint. Fix failures without weakening assertions or introducing broad suppressions.
3. Commit and push the verified changes to the intended remote branch. Record its full commit SHA and branch name, then look for a CI run for that exact commit:

   ```bash
   branch=$(git branch --show-current)
   commit_sha=$(git rev-parse HEAD)
   gh run list --workflow ci.yml --branch "$branch" --commit "$commit_sha" --limit 10 --json databaseId,headSha,event,status,conclusion,url
   ```

4. If no matching run appears, check whether the branch has an open PR targeting `main` and briefly recheck for a delayed automatic trigger. If the push does not trigger CI, run `gh workflow run ci.yml --ref "$branch"`, then repeat the lookup. Reuse an existing matching queued/running run rather than dispatching duplicates. Verify the selected run's `headSha` equals the pushed SHA; an earlier green run is not evidence for the new commit.
5. Monitor the selected run with `gh run view RUN_ID --json status,conclusion,headSha,jobs,url`, replacing `RUN_ID` with its database ID. Poll at reasonable intervals and keep the user informed during long runs. Wait for `status=completed` and `conclusion=success` before declaring the commit-and-push task complete; queued, running, cancelled, or skipped runs are not a pass. Inspect the CPU job's pytest, Ruff, MyPy, coverage, and report-upload steps.
6. If CI fails, inspect `gh run view RUN_ID --log-failed`. For test failures, download `gh run download RUN_ID --name pytest-cpu --dir /tmp/pytest-cpu-RUN_ID` and inspect the JUnit report; installation failures may produce no artifact. Fix the actual failure, rerun the affected local checks and required pre-commit pytest command, commit and push the fix, then verify CI for the new SHA. Do not relax checks to obtain a green result. If access or infrastructure blocks verification, report that CI remains unverified and give the specific blocker.
7. Report the commit SHA, branch, local test counts/failures/skips, sweep exclusion, and a link to the exact remote CI run with its final result. Keep local GPU results distinct from hosted CPU results. Never claim remote CI passed based only on local validation.

## Security & Configuration

Keep credentials in ignored `.env` files and machine overrides in `configs/local/`. Never commit secrets, datasets, checkpoints, or experiment outputs. Preserve unrelated working-tree changes.

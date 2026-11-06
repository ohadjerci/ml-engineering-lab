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

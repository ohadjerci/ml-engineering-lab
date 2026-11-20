import fcntl
import re
import subprocess
import warnings
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlsplit

from omegaconf import DictConfig

from src.data.components import digest, file_hash
from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str | None) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    value = metric_dict[metric_name]
    metric_value = value.item() if hasattr(value, "item") else float(value)
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


SOURCE_PATHS = ("src", "configs", "pyproject.toml", "uv.lock", "Dockerfile", "compose.yaml")


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Locks coordinate Linux processes sharing the same local filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def command(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        # Transport output may contain credentials. Keep it out of Hydra/MLflow logs.
        raise RuntimeError(
            f"{args[0]} {args[1] if len(args) > 1 else ''} failed "
            f"(exit {result.returncode}). Check repository permissions, credentials and "
            "remote connectivity; retry the dataset stage before training."
        )
    return result.stdout.strip()


def git(root: Path, *args: str) -> str:
    try:
        return command(root, "git", "-c", f"safe.directory={root.resolve()}", *args)
    except RuntimeError:
        raise RuntimeError(
            f"Git {args[0]} failed. Check repository access, Git identity and remote credentials; "
            "resolve the problem and rerun dataset preparation."
        ) from None


def public_location(value: str) -> str:
    """Require credential-free references, allowing an SSH username."""
    parsed = urlsplit(value)
    if parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Remote/repository URLs must not contain credentials or query strings")
    if parsed.username and parsed.scheme not in {"ssh", ""}:
        raise ValueError("Use provider credentials instead of credentials in a URL")
    return value


def source_provenance(repository: Path, executed_root: Path) -> dict:
    revision = git(repository, "rev-parse", "HEAD")
    dirty = git(repository, "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_PATHS)
    if dirty:
        raise RuntimeError(
            "Relevant source/configuration is uncommitted. Commit it and rebuild the Docker "
            "image before versioned training; use versioning.enabled=false for exploration."
        )
    files = git(repository, "ls-files", "--", *SOURCE_PATHS).splitlines()
    if not files:
        raise RuntimeError("Cannot identify committed training source")
    hashes = {}
    for relative in files:
        expected = file_hash(repository / relative)
        actual = executed_root / relative
        if not actual.is_file() or file_hash(actual) != expected:
            raise RuntimeError(f"Executed source differs from Git: {relative}. Rebuild the image.")
        hashes[relative] = expected
    # Repository URL can be absent for a local-only repo; do not invent a remote.
    remotes = git(repository, "remote").splitlines()
    reference = git(repository, "remote", "get-url", "origin") if "origin" in remotes else ""
    public_location(reference)
    return {
        "revision": revision,
        "identity": digest(hashes),
        "files": hashes,
        "repository": reference,
    }


def commit_metadata(repository: Path, paths: list[str]) -> str:
    """Commit exactly these generated paths, preserving unrelated index entries."""
    if not paths or any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise ValueError("Metadata commit requires explicit repository-relative paths")
    git(repository, "symbolic-ref", "--quiet", "HEAD")
    try:
        git(repository, "var", "GIT_AUTHOR_IDENT")
        git(repository, "var", "GIT_COMMITTER_IDENT")
    except RuntimeError:
        raise RuntimeError(
            "Configure Git user.name/user.email or author/committer environment variables before publishing data"
        ) from None
    changed = git(repository, "status", "--porcelain", "--", *paths)
    if changed:
        git(repository, "add", "--", *paths)
        git(repository, "commit", "--only", "-m", "Version dataset snapshot", "--", *paths)
    return git(repository, "rev-parse", "HEAD")


def validate_remote(name: str, url: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("DVC remote name must contain letters, digits, underscores or hyphens")
    if not url:
        raise ValueError("Set the selected DVC remote URL before preparing data")
    public_location(url)

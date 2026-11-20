"""Transactional dataset preparation and DVC metadata publication."""

import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from ruamel.yaml import YAML

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components import build_manifest, digest, file_hash, verify_snapshot, write_json  # noqa: E402
from src.utils.utils import (  # noqa: E402
    command,
    commit_metadata,
    file_lock,
    git,
    validate_remote,
)

REPRODUCTION_SCHEMA = 1


@dataclass
class DatasetVersion:
    snapshot: Path
    manifest: dict[str, Any]
    provenance: dict[str, Any]


def configure_dvc(repository: Path, cfg: DictConfig) -> list[str]:
    bootstrap = [".dvc/.gitignore", ".dvc/config", ".dvcignore"]
    if not (repository / ".dvc").exists():
        command(repository, "dvc", "init")
    # Location and credential-provider settings are local, not committed metadata.
    try:
        # Update only URL so existing ignored auth/provider settings survive.
        command(
            repository,
            "dvc",
            "remote",
            "modify",
            "--local",
            str(cfg.remote.name),
            "url",
            str(cfg.remote.url),
        )
    except RuntimeError:
        command(
            repository,
            "dvc",
            "remote",
            "add",
            "--local",
            str(cfg.remote.name),
            str(cfg.remote.url),
        )
    cache = Path(str(cfg.cache_dir)).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    command(repository, "dvc", "config", "--local", "cache.dir", str(cache))
    command(repository, "dvc", "config", "--local", "cache.type", "copy")
    command(repository, "dvc", "config", "--local", "core.autostage", "false")
    return [p for p in bootstrap if (repository / p).is_file()]


def check_owned_metadata(repository: Path, relative: str, journal: dict) -> None:
    path = repository / relative
    status = git(repository, "status", "--porcelain", "--", relative)
    if not status:
        return
    staged = git(repository, "diff", "--cached", "--name-only", "--", relative)
    if (
        not path.is_file()
        or path.is_symlink()
        or journal.get(relative) != file_hash(path)
        or (staged and git(repository, "diff", "--name-only", "--", relative))
    ):
        raise RuntimeError(f"DVC metadata has user changes: {relative}")


def prepare_dataset(cfg: DictConfig) -> DatasetVersion:
    options = cfg.versioning
    repository = Path(str(options.repository_dir)).resolve()
    root = repository / str(options.snapshot_dir)
    if not root.resolve().is_relative_to(repository) or root == repository:
        raise ValueError("versioning.snapshot_dir must be a subdirectory of the repository")
    if root.relative_to(repository).parts[0] in {".git", ".dvc", "src", "configs"}:
        raise ValueError(
            "Dataset snapshots must not be stored inside source or repository metadata"
        )
    validate_remote(str(options.remote.name), str(options.remote.url))
    backend = options.remote.get("backend")
    if backend in {"ssh", "s3"} and not str(options.remote.url).startswith(f"{backend}://"):
        raise ValueError(f"Set DVC_REMOTE_URL to a {backend}:// location for the selected preset")
    if options.git_push and not options.git_commit:
        raise ValueError("Git push requires metadata commits")
    if options.git_push and (not options.git_remote or not options.git_branch):
        raise ValueError("Explicit git_remote and git_branch are required for Git push")
    if options.git_push:
        if str(options.git_remote) not in git(repository, "remote").splitlines():
            raise ValueError("git_remote must name an existing Git remote")
        git(repository, "check-ref-format", f"refs/heads/{options.git_branch}")
    lock_root = Path(str(options.coordination_dir))
    with file_lock(lock_root / "dataset-publication.lock"):
        if options.get("restore"):
            return restore_dataset(cfg, Path(str(options.restore)))
        journal_path = lock_root / "metadata-journal.json"
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
        owned_paths = [
            ".dvc/.gitignore",
            ".dvc/config",
            ".dvcignore",
            (root / ".gitignore").relative_to(repository).as_posix(),
        ]
        for relative in owned_paths:
            check_owned_metadata(repository, relative, journal)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=root) as temporary:
            staging = Path(temporary)
            datamodule = hydra.utils.instantiate(cfg.data)
            if not callable(getattr(datamodule, "export_snapshot", None)):
                raise TypeError("Data module must implement export_snapshot(destination)")
            description = datamodule.export_snapshot(staging)
            manifest = build_manifest(staging, description)
            # Two logical datasets can share bytes without sharing a manifest name.
            snapshot = root / f"{digest(manifest['name'])[:16]}-{manifest['identity']}"
            pointer = snapshot.with_suffix(".dvc")
            relative = pointer.relative_to(repository).as_posix()
            # Never replace user-modified pointers, including staged modifications.
            check_owned_metadata(repository, relative, journal)
            if snapshot.exists():
                verify_snapshot(snapshot, manifest["identity"])
            else:
                os.rename(staging, snapshot)
                staging.mkdir()  # TemporaryDirectory still owns this now-empty path.
        bootstrap = configure_dvc(repository, options)
        command(repository, "dvc", "add", str(snapshot.relative_to(repository)))
        owned_paths.extend([relative, *bootstrap])
        journal.update(
            {p: file_hash(repository / p) for p in owned_paths if (repository / p).is_file()}
        )
        write_json(journal_path, journal)
        if options.upload:
            command(repository, "dvc", "push", "--remote", str(options.remote.name), relative)
        allowlist = [relative, *bootstrap]
        ignore = root / ".gitignore"
        if ignore.is_file():
            allowlist.append(ignore.relative_to(repository).as_posix())
        revision = commit_metadata(repository, allowlist) if options.git_commit else None
        if revision is None and git(repository, "ls-files", "--", relative):
            if not git(repository, "status", "--porcelain", "--", relative):
                revision = git(repository, "rev-parse", "HEAD")
        if options.git_push:
            git(
                repository,
                "push",
                "--no-follow-tags",
                "--",
                str(options.git_remote),
                f"HEAD:refs/heads/{options.git_branch}",
            )
        pointer_data = YAML(typ="safe").load(pointer.read_text())
        provenance = {
            "schema_version": 1,
            "dataset_name": manifest["name"],
            "identity": manifest["identity"],
            "metadata_revision": revision,
            "snapshot_path": snapshot.relative_to(repository).as_posix(),
            "remote": {"name": str(options.remote.name), "url": str(options.remote.url)},
            "pointers": [
                {"path": relative, "sha256": file_hash(pointer), "outs": pointer_data["outs"]}
            ],
            "splits": manifest["split_info"],
            "class_mapping": manifest["class_mapping"],
            "annotation_format": manifest["annotation_format"],
            "manifest_schema_version": manifest["schema_version"],
            "uploaded": bool(options.upload),
            "committed": bool(revision),
        }
        return DatasetVersion(snapshot, manifest, provenance)


def restore_dataset(cfg: DictConfig, provenance_file: Path) -> DatasetVersion:
    """Restore only in the isolated reproduction checkout, never the user's checkout."""
    provenance = json.loads(provenance_file.read_text())
    if provenance.get("schema_version") != 1:
        raise ValueError("Unsupported data provenance schema")
    options = cfg.versioning
    repository = Path(str(options.repository_dir)).resolve()
    revision = provenance.get("metadata_revision")
    if not revision or not provenance.get("uploaded"):
        raise ValueError("This dataset was not committed/uploaded and cannot be reproduced")
    configure_dvc(repository, options)
    for pointer in provenance["pointers"]:
        path = repository / pointer["path"]
        if not path.resolve().is_relative_to(repository) or path.suffix != ".dvc":
            raise ValueError("Unsafe dataset pointer path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(git(repository, "show", f"{revision}:{pointer['path']}") + "\n")
        if file_hash(path) != pointer["sha256"]:
            raise ValueError("Restored DVC pointer hash mismatch")
        command(repository, "dvc", "pull", "--remote", str(options.remote.name), pointer["path"])
    snapshot = repository / provenance["snapshot_path"]
    if not snapshot.resolve().is_relative_to(repository):
        raise ValueError("Unsafe dataset snapshot path")
    manifest = verify_snapshot(snapshot, provenance["identity"])
    if (
        manifest["split_info"] != provenance["splits"]
        or manifest["name"] != provenance["dataset_name"]
        or manifest["class_mapping"] != provenance["class_mapping"]
    ):
        raise ValueError("Restored dataset name, classes or split identity mismatch")
    return DatasetVersion(snapshot, manifest, provenance)


def prepare_reproduction(run_id: str, destination: Path) -> None:
    """Download replay inputs and create a separate checkout; execute no historical code."""
    from mlflow import MlflowClient

    if not re.fullmatch(r"[a-fA-F0-9]{32}", run_id):
        raise ValueError("Expected an MLflow run ID")
    if destination.exists():
        raise ValueError("Reproduction destination already exists; choose a fresh directory")
    destination.mkdir(parents=True)
    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    artifacts = {item.path for item in client.list_artifacts(run_id, "provenance")}
    required = (
        "config.json",
        "data_version.json",
        "source.json",
        "fingerprint.json",
        "environment.json",
    )
    if any(f"provenance/{artifact}" not in artifacts for artifact in required):
        raise ValueError("Historical run is missing the provenance needed for reproduction")
    for artifact in required:
        client.download_artifacts(run_id, f"provenance/{artifact}", str(destination))
    source = json.loads((destination / "source.json").read_text())
    data = json.loads((destination / "data_version.json").read_text())
    if data.get("schema_version") != REPRODUCTION_SCHEMA or not data.get("metadata_revision"):
        raise ValueError(
            "Historical run lacks committed dataset provenance required for reproduction"
        )
    repository = Path(os.environ.get("GIT_WORKTREE", ".")).resolve()
    checkout = destination / "checkout"
    command(
        destination,
        "git",
        "-c",
        f"safe.directory={repository}",
        "clone",
        "--no-hardlinks",
        "--no-checkout",
        str(repository),
        str(checkout),
    )
    from src.utils.utils import public_location

    reference = public_location(source["repository"])
    if reference:
        git(checkout, "remote", "set-url", "origin", reference)
    else:
        git(checkout, "remote", "remove", "origin")
    for revision in (source["revision"], data["metadata_revision"]):
        if not re.fullmatch(r"[a-f0-9]{40,64}", revision):
            raise ValueError("Invalid historical Git revision")
        try:
            git(checkout, "cat-file", "-e", f"{revision}^{{commit}}")
        except RuntimeError:
            if not reference:
                raise ValueError(
                    "Historical Git revision is unavailable; retain or fetch its repository"
                ) from None
            git(checkout, "fetch", reference, revision)
    git(checkout, "checkout", "--detach", source["revision"])
    entry = checkout / "src/data/prepare_dataset.py"
    if not entry.is_file() or "REPRODUCTION_SCHEMA = 1" not in entry.read_text():
        raise ValueError(
            "Recorded code does not support the reproduction entry point; current code will not be substituted"
        )
    # Docker must not create root-owned host bind directories for the replay user.
    for relative in (
        "data/.dvc-cache",
        "data/.locks",
        "logs/train",
        "logs/eval",
        "logs/prepare_dataset",
        "logs/reproduce",
    ):
        (checkout / relative).mkdir(parents=True, exist_ok=True)
    for relative, expected in source["files"].items():
        path = checkout / relative
        if (
            not path.resolve().is_relative_to(checkout)
            or not path.is_file()
            or file_hash(path) != expected
        ):
            raise ValueError(f"Historical source hash mismatch: {relative}")
    cfg = OmegaConf.create(json.loads((destination / "config.json").read_text()))
    for pointer in data["pointers"]:
        path = checkout / pointer["path"]
        if path.suffix != ".dvc" or not path.resolve().is_relative_to(checkout):
            raise ValueError("Unsafe dataset pointer")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            git(checkout, "show", f"{data['metadata_revision']}:{pointer['path']}") + "\n"
        )
        if file_hash(path) != pointer["sha256"]:
            raise ValueError("Dataset pointer hash mismatch")
    shutil.copyfile(destination / "data_version.json", checkout / "replay-data.json")
    shutil.copyfile(destination / "environment.json", checkout / "replay-environment.json")
    cfg.paths.root_dir = "/workspace"
    cfg.paths.data_dir = "/workspace/data"
    cfg.paths.log_dir = "/workspace/logs/train"
    cfg.paths.output_dir = f"/workspace/logs/train/reproductions/{run_id}"
    cfg.paths.work_dir = "/workspace"
    cfg.trainer.default_root_dir = cfg.paths.output_dir
    if cfg.get("callbacks", {}).get("model_checkpoint"):
        cfg.callbacks.model_checkpoint.dirpath = f"{cfg.paths.output_dir}/checkpoints"
    cfg.versioning.repository_dir = "/repo"
    cfg.versioning.cache_dir = "/workspace/data/.dvc-cache"
    cfg.versioning.coordination_dir = "/workspace/data/.locks"
    cfg.versioning.restore = "/repo/replay-data.json"
    cfg.versioning.git_commit = False
    cfg.versioning.git_push = False
    cfg.reproducibility.reuse = False
    cfg.reproducibility.force_retrain = True
    cfg.reproducibility.original_run_id = run_id
    cfg.reproducibility.expected_environment_path = "/repo/replay-environment.json"
    cfg.reproducibility.expected_initialization = json.loads(
        (destination / "fingerprint.json").read_text()
    )["initialization"]
    cfg.logger.mlflow.pop("run_id", None)
    if cfg.get("ckpt_path"):
        path = client.download_artifacts(
            run_id, "provenance/initialization/initialization.ckpt", str(checkout)
        )
        cfg.ckpt_path = f"/repo/{Path(path).name}"
    else:
        path = client.download_artifacts(
            run_id, "provenance/initialization/model_state.pt", str(checkout)
        )
        cfg.reproducibility.initial_state_path = f"/repo/{Path(path).name}"
    write_json(
        checkout / "replay.json", OmegaConf.to_container(cfg, resolve=True), preserve_order=True
    )


def run_reproduction(configuration: Path) -> None:
    from src.train import train

    cfg = OmegaConf.create(json.loads(configuration.read_text()))
    # Transport settings come from the current environment, not historical secrets.
    cfg.logger.mlflow.tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    cfg.versioning.remote.url = os.environ.get("DVC_REMOTE_URL", cfg.versioning.remote.url)
    cfg.versioning.remote.name = os.environ.get("DVC_REMOTE_NAME", cfg.versioning.remote.name)
    train(cfg)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="data/prepare_dataset")
def main(cfg: DictConfig) -> None:
    if not cfg.versioning.enabled:
        raise ValueError("Independent preparation requires versioning.enabled=true")
    version = prepare_dataset(cfg)
    print(
        json.dumps(
            {**version.provenance, "manifest_path": str(version.snapshot / "manifest.json")}
        )
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--reproduce-prepare":
        prepare_reproduction(sys.argv[2], Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--reproduce-run":
        run_reproduction(Path(sys.argv[2]))
    else:
        main()

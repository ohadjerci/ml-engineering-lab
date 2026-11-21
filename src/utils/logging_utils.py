import hashlib
import importlib.metadata
import json
import platform
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, InputTag
from mlflow.exceptions import MlflowException
from omegaconf import OmegaConf

from src.data.components import canonical, digest, file_hash
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def sanitized(value: Any) -> Any:
    """Remove credentials before extending existing parameter/artifact logging."""
    if OmegaConf.is_config(value):
        if OmegaConf.is_dict(value) and "hydra" in value:
            value = OmegaConf.masked_copy(value, [k for k in value.keys() if k != "hydra"])
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {
            str(k): (
                "<redacted>"
                if re.search(
                    r"password|passphrase|secret|token|api_key|access_key|private_key|credential",
                    str(k),
                    re.I,
                )
                else sanitized(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitized(v) for v in value]
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.password or parsed.query or parsed.fragment:
            host = parsed.hostname or ""
            if parsed.port:
                host += f":{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return value


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(sanitized(hparams))


def runtime_information() -> dict[str, Any]:
    """An explicit environment allowlist; never log arbitrary environment variables."""
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        if cpu_info.exists()
        else platform.processor()
    )
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "packages": sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "threads": torch.get_num_threads(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def initialization_identity(model: torch.nn.Module, checkpoint: Path | None) -> str:
    if checkpoint is not None:
        return file_hash(checkpoint)
    checksum = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        checksum.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        checksum.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def experiment_identity(cfg, dataset: dict, source: dict, initialization: str, runtime: dict):
    resolved = sanitized(cfg)
    if not isinstance(resolved, dict):
        raise TypeError("Expected a mapping configuration")
    data = dict(resolved["data"])
    for key in ("data_dir", "root_dir", "manifest_path", "snapshot_path"):
        data.pop(key, None)
    if isinstance(data.get("split"), dict):
        data["split"] = {k: v for k, v in data["split"].items() if k != "assignments_path"}
    trainer = dict(resolved["trainer"])
    evaluation = {"test": resolved.get("test", False)}
    for key in ("limit_test_batches", "inference_mode"):
        if key in trainer:
            evaluation[key] = trainer.pop(key)
    for key in ("default_root_dir", "logger", "enable_progress_bar", "enable_model_summary"):
        trainer.pop(key, None)
    callbacks = {}
    for name, settings in (resolved.get("callbacks") or {}).items():
        if settings is None:
            continue
        if name in {
            "rich_progress_bar",
            "mlflow_checkpoints",
            "mlflow_results",
            "classification_results",
        }:
            continue
        callbacks[name] = {
            k: v for k, v in settings.items() if k not in {"dirpath", "filename", "verbose"}
        }
    # Resolved configuration already covers config values; file locations and comments
    # in YAML do not constitute a change to executed training code.
    code = {k: v for k, v in source["files"].items() if k.startswith("src/")}
    payload = {
        "schema_version": 1,
        "dataset": dataset["identity"],
        "dataset_name": dataset["dataset_name"],
        "data": data,
        "model": resolved["model"],
        "trainer": trainer,
        "callbacks": callbacks,
        "seed": resolved.get("seed"),
        "compile": resolved.get("compile", False),
        "initialization": initialization,
        "code": code,
        "environment": runtime,
        "lockfile": source["files"].get("uv.lock"),
    }
    return digest(payload), digest(evaluation), payload


def verified_receipt(client: MlflowClient, run_id: str, fingerprint: str) -> dict | None:
    """A FINISHED status alone is insufficient: Lightning can finalize after fit."""
    run = client.get_run(run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("reproducibility.complete") != "true":
        return None
    with tempfile.TemporaryDirectory() as directory:
        try:
            receipt = json.loads(
                Path(
                    client.download_artifacts(run_id, "provenance/completion.json", directory)
                ).read_text()
            )
            if receipt.get("schema_version") != 1 or receipt.get("fingerprint") != fingerprint:
                return None
            for item in receipt["artifacts"]:
                destination = Path(directory) / digest(item)
                destination.mkdir()
                if item.get("model_uri"):
                    path = Path(
                        mlflow.artifacts.download_artifacts(
                            artifact_uri=item["model_uri"],
                            dst_path=str(destination),
                            tracking_uri=client.tracking_uri,
                        )
                    )
                    checks = {
                        p.relative_to(path).as_posix(): file_hash(p)
                        for p in path.rglob("*")
                        if p.is_file()
                    }
                    if checks != item["files"]:
                        return None
                else:
                    path = Path(client.download_artifacts(run_id, item["path"], str(destination)))
                    if file_hash(path) != item["sha256"]:
                        return None
            if not receipt["artifacts"] or not receipt.get("checkpoint"):
                return None
            return receipt
        except (MlflowException, OSError, ValueError, KeyError):
            return None


def find_reusable(client: MlflowClient, experiment_id: str, fingerprint: str):
    token = None
    while True:
        runs = client.search_runs(
            [experiment_id],
            filter_string=f"tags.`training.fingerprint` = '{fingerprint}' "
            "AND attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=100,
            page_token=token,
        )
        for run in runs:
            receipt = verified_receipt(client, run.info.run_id, fingerprint)
            if receipt is not None:
                return run, receipt
        token = runs.token
        if not token:
            return None


def log_provenance(
    logger: MLFlowLogger,
    cfg,
    version,
    source: dict,
    runtime: dict,
    fingerprint: str,
    evaluation: str,
    payload: dict,
    initialization: Path | None,
) -> None:
    client, run_id = logger.experiment, logger.run_id
    if not isinstance(run_id, str):
        raise RuntimeError("MLflow did not create a run")
    provenance = {
        **version.provenance,
        "repository": source["repository"],
        "code_revision": source["revision"],
        "code_identity": source["identity"],
        "references": {
            "source": "code/",
            "configuration": "provenance/config.json",
            "lockfile": "provenance/uv.lock",
            "environment": "provenance/environment.json",
        },
    }
    for key, value in {
        "dataset.name": provenance["dataset_name"],
        "dataset.identity": provenance["identity"],
        "dataset.metadata_revision": provenance["metadata_revision"] or "uncommitted",
        "training.code_revision": source["revision"],
        "training.fingerprint": fingerprint,
        "evaluation.fingerprint": evaluation,
        "reproducibility.complete": "false",
        "reproducibility.schema_version": "1",
    }.items():
        client.set_tag(run_id, key, value)
    if cfg.reproducibility.original_run_id:
        client.set_tag(run_id, "reproduction.original_run_id", cfg.reproducibility.original_run_id)
    resolved = sanitized(cfg)
    # Configurations must contain provider references, not secret values.
    client.log_dict(run_id, provenance, "provenance/data_version.json")
    client.log_dict(run_id, resolved, "provenance/config.json")
    client.log_dict(run_id, source, "provenance/source.json")
    client.log_dict(run_id, runtime, "provenance/environment.json")
    client.log_dict(run_id, payload, "provenance/fingerprint.json")
    client.log_artifact(run_id, str(Path(cfg.paths.root_dir) / "uv.lock"), "provenance")
    repository = Path(cfg.versioning.repository_dir)
    for pointer in provenance["pointers"]:
        client.log_artifact(run_id, str(repository / pointer["path"]), "provenance/dvc")
    if initialization:
        client.log_artifact(run_id, str(initialization), "provenance/initialization")
    for name, info in provenance["splits"].items():
        dataset = Dataset(
            name=f"{provenance['dataset_name']}/{name}",
            # MLflow 3.18 limits this display field to 36 characters. Full SHA-256
            # remains authoritative in tags, provenance and dataset source metadata.
            digest=provenance["identity"][:32],
            source_type="dvc",
            source=canonical(
                {
                    "repository": source["repository"],
                    "revision": provenance["metadata_revision"],
                    "pointers": provenance["pointers"],
                    "identity": provenance["identity"],
                }
            ).decode(),
            profile=canonical(info).decode(),
        )
        client.log_inputs(
            run_id, datasets=[DatasetInput(dataset, [InputTag("mlflow.data.context", name)])]
        )


def complete_run(
    logger: MLFlowLogger, trainer, cfg, fingerprint: str, evaluation: str, metrics: dict
) -> dict:
    client, run_id = logger.experiment, logger.run_id
    if not isinstance(run_id, str):
        raise RuntimeError("MLflow did not create a run")
    # Preserve the existing source artifact even when testing is disabled.
    if not client.list_artifacts(run_id, "code"):
        client.log_artifacts(run_id, "src", "code")
    checkpoint = trainer.checkpoint_callback
    path = Path(checkpoint.best_model_path or checkpoint.last_model_path)
    if not path.is_file():
        raise RuntimeError("Reproducible training requires a saved model checkpoint")
    checkpoint_artifact = f"checkpoints/{path.name}"
    if not any(
        item.path == checkpoint_artifact for item in client.list_artifacts(run_id, "checkpoints")
    ):
        client.log_artifact(run_id, str(path), "checkpoints")
    artifact = {"path": checkpoint_artifact, "sha256": file_hash(path)}
    artifacts: list[dict[str, Any]] = [artifact]
    model_uri = client.get_run(run_id).data.tags.get("reproducibility.model_uri")
    if cfg.registry.enabled and cfg.test and not model_uri:
        raise RuntimeError("Published model reference is missing; run cannot be reused")
    if model_uri:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=model_uri, dst_path=directory, tracking_uri=client.tracking_uri
                )
            )
            artifacts.append(
                {
                    "model_uri": model_uri,
                    "files": {
                        p.relative_to(root).as_posix(): file_hash(p)
                        for p in root.rglob("*")
                        if p.is_file()
                    },
                }
            )
    receipt = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "evaluation": evaluation,
        "checkpoint": artifact["path"],
        "model_uri": model_uri or f"runs:/{run_id}/{artifact['path']}",
        "metrics": {k: float(v) for k, v in metrics.items()},
        "artifacts": artifacts,
    }
    client.log_dict(run_id, receipt, "provenance/completion.json")
    client.set_tag(run_id, "reproducibility.complete", "true")
    client.set_terminated(run_id, "FINISHED")
    return receipt

"""Publish evaluated models independently of the training package."""

import copy
import json
import math
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.pytorch
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf


def _json_value(value: Any) -> Any:
    """Convert common runtime/config values to values accepted by MLflow metadata."""
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def extract_model_metadata(
    datamodule: LightningDataModule,
    model: LightningModule,
    data_cfg: DictConfig,
    model_cfg: DictConfig,
) -> dict[str, Any]:
    """Collect descriptive model metadata from runtime objects, then config fallbacks."""
    metadata: dict[str, Any] = {}

    def add(key: str, runtime_value: Any, config_value: Any = None) -> None:
        value = runtime_value if runtime_value is not None else config_value
        if value is not None:
            metadata[key] = _json_value(value)

    add(
        "dataset",
        getattr(datamodule, "name", None),
        data_cfg.get("name") or data_cfg.get("_target_"),
    )
    if "dataset" not in metadata:
        metadata["dataset"] = datamodule.__class__.__name__

    transforms = getattr(datamodule, "transforms_test", None)
    transform_list = getattr(transforms, "transforms", None)
    if transform_list is not None:
        metadata["preprocessing"] = [item.__class__.__name__ for item in transform_list]
        mean = None
        std = None
        for item in transform_list:
            if hasattr(item, "mean"):
                mean = item.mean
            if hasattr(item, "std"):
                std = item.std
        add("mean", mean, data_cfg.get("mean"))
        add("std", std, data_cfg.get("std"))
    else:
        add("preprocessing", None, data_cfg.get("preprocessing"))
        add("mean", None, data_cfg.get("mean"))
        add("std", None, data_cfg.get("std"))

    sample_shape = None
    dataset = getattr(datamodule, "data_test", None)
    if dataset is not None:
        try:
            sample, _ = dataset[0]
            sample_shape = list(sample.shape)
        except (IndexError, TypeError, ValueError, AttributeError):
            pass
    add("input_shape", sample_shape, data_cfg.get("input_shape"))
    if sample_shape and len(sample_shape) >= 2:
        add("image_size", sample_shape[-2:])
    else:
        add("image_size", None, data_cfg.get("image_size"))

    class_labels = None
    pending_datasets = [dataset]
    visited: set[int] = set()
    while pending_datasets and class_labels is None:
        source_dataset = pending_datasets.pop()
        if source_dataset is None or id(source_dataset) in visited:
            continue
        visited.add(id(source_dataset))
        class_labels = getattr(source_dataset, "classes", None)
        nested = getattr(source_dataset, "dataset", None)
        if nested is not None:
            pending_datasets.append(nested)
        pending_datasets.extend(getattr(source_dataset, "datasets", []))
    if class_labels is None:
        class_labels = getattr(datamodule, "classes", None)
    add("labels", class_labels, data_cfg.get("labels"))
    class_count = getattr(datamodule, "num_classes", None)
    if callable(class_count):
        class_count = class_count()
    if class_count is None and class_labels is not None:
        class_count = len(class_labels)
    add("num_classes", class_count, data_cfg.get("num_classes"))

    network = getattr(model, "net", None)
    add(
        "model_architecture",
        network.__class__.__name__ if network is not None else None,
        model_cfg.get("_target_"),
    )
    if "model_architecture" not in metadata:
        metadata["model_architecture"] = model.__class__.__name__
    network_cfg = model_cfg.get("net")
    if network_cfg is not None:
        config_values = OmegaConf.to_container(network_cfg, resolve=True)
        if isinstance(config_values, dict):
            metadata["model_config"] = _json_value(
                {
                    key: value
                    for key, value in config_values.items()
                    if not isinstance(key, str) or not key.startswith("_")
                }
            )

    # Confirm the output has only JSON-safe structures before handing it to MLflow.
    return json.loads(json.dumps(metadata, allow_nan=False))


def promote_if_better(
    client: MlflowClient, name: str, alias: str, version: str, metric: str, score: float
) -> bool:
    """Promote only strictly better finite scores; local publishers run sequentially."""
    if not math.isfinite(score):
        raise ValueError(f"Non-finite promotion metric {metric}: {score}")
    current_version = client.get_registered_model(name).aliases.get(alias)
    if current_version is not None:
        current = client.get_model_version(name, current_version)
        if current.tags.get("selection_metric") != metric:
            raise ValueError(f"Alias {alias} has incompatible selection metadata")
        previous = float(current.tags["selection_score"])
        if not math.isfinite(previous):
            raise ValueError(f"Alias {alias} has a non-finite selection score")
        if score >= previous:
            return False
    client.set_registered_model_alias(name, alias, version)
    return True


def publish_evaluated_model(
    trainer: Trainer,
    model: LightningModule,
    cfg: DictConfig,
    checkpoint: str | None,
    metadata: dict[str, Any],
) -> str | None:
    """Export the tested network, register it, and compare finalized test loss."""
    if not cfg.enabled or not trainer.is_global_zero:
        return None
    logger = next((item for item in trainer.loggers if isinstance(item, MLFlowLogger)), None)
    if logger is None:
        return None
    metric = str(cfg.metric)
    score = float(trainer.callback_metrics[metric])
    if not math.isfinite(score):
        raise ValueError(f"Non-finite promotion metric {metric}: {score}")
    network = copy.deepcopy(cast(torch.nn.Module, model.net)).cpu().eval()
    example = torch.zeros(1, 1, 28, 28)
    with torch.inference_mode():
        packaged = torch.jit.trace(network, example)
        torch.testing.assert_close(packaged(example), network(example))
    client = logger.experiment
    mlflow.set_tracking_uri(client.tracking_uri)
    mlflow.set_registry_uri(client.tracking_uri)
    with mlflow.start_run(run_id=logger.run_id):
        info = mlflow.pytorch.log_model(
            packaged,
            name="model",
            registered_model_name=str(cfg.model_name),
            serialization_format="pickle",
            pip_requirements=["torch==2.6.0", "mlflow==3.18.0"],
            metadata=metadata,
        )
    version = str(info.registered_model_version)
    for key, value in {
        "selection_metric": metric,
        "selection_score": str(score),
        "checkpoint": Path(checkpoint).name if checkpoint else "evaluated_weights",
    }.items():
        client.set_model_version_tag(str(cfg.model_name), version, key, value)
    if "preprocessing" in metadata:
        client.set_model_version_tag(
            str(cfg.model_name), version, "preprocessing", str(metadata["preprocessing"])
        )
    promote_if_better(client, str(cfg.model_name), str(cfg.alias), version, metric, score)
    return version

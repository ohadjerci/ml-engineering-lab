"""Publish evaluated models independently of the training package."""

import copy
import math
from pathlib import Path
from typing import cast

import mlflow
import mlflow.pytorch
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from omegaconf import DictConfig


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
    trainer: Trainer, model: LightningModule, cfg: DictConfig, checkpoint: str | None
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
    metadata = {
        "preprocessing": "mnist-v1",
        "image_size": [28, 28],
        "mean": 0.1307,
        "std": 0.3081,
        "labels": list(range(10)),
    }
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
        "preprocessing": "mnist-v1",
    }.items():
        client.set_model_version_tag(str(cfg.model_name), version, key, value)
    promote_if_better(client, str(cfg.model_name), str(cfg.alias), version, metric, score)
    return version

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, cast

# Version 1
import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf

# from lightning.pytorch.plugins.environments import KubeflowEnvironment


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------ #

from src import utils  # noqa: E402 -- root setup must precede local imports
from src.data.components import canonical, write_json  # noqa: E402
from src.data.prepare_dataset import prepare_dataset  # noqa: E402
from src.utils.logging_utils import (  # noqa: E402
    complete_run,
    experiment_identity,
    find_reusable,
    initialization_identity,
    log_provenance,
    runtime_information,
    sanitized,
)
from src.utils.model_registry import (  # noqa: E402
    extract_model_metadata,
    publish_evaluated_model,
)
from src.utils.utils import file_lock, source_provenance  # noqa: E402

log = utils.get_pylogger(__name__)


def _train(
    cfg: DictConfig,
    datamodule=None,
    model=None,
    logger=None,
    evaluation_checkpoint: str | None = None,
) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if model is None and cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = datamodule or hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    if model is None:
        model = hydra.utils.instantiate(cfg.model)
        state = cfg.get("reproducibility", {}).get("initial_state_path")
        if state:
            model.load_state_dict(torch.load(state, map_location="cpu", weights_only=True))

    """log.info("Instantiating cluster...")
    cluster: ClusterEnvironment = hydra.utils.instantiate(cfg.cluster)"""

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = logger if logger is not None else utils.instantiate_loggers(cfg.get("logger"))

    # LearningRateMonitor requires a logger; logger=[] also disables it for sweeps.
    if not logger:
        callbacks = [
            callback for callback in callbacks if not isinstance(callback, LearningRateMonitor)
        ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )  # , plugins = cluster)

    results_dir = os.path.join(trainer.default_root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        # Compiling an nn.Module preserves its module interface at runtime.
        model = cast(LightningModule, torch.compile(model))

    if evaluation_checkpoint is not None:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=evaluation_checkpoint)
        return dict(trainer.callback_metrics), object_dict

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        # The configured model_checkpoint callback supplies best_model_path.
        ckpt_path: str | None = cast(ModelCheckpoint, trainer.checkpoint_callback).best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        metadata = extract_model_metadata(datamodule, model, cfg.data, cfg.model)
        publish_evaluated_model(trainer, model, cfg.registry, ckpt_path, metadata)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    if not cfg.get("versioning", {}).get("enabled", False):
        return _train(cfg)
    # Lightning subprocess children use the parent's exact snapshot, configuration
    # and run. They must not acquire the parent's experiment lock or publish data.
    session_path = os.environ.get("DEASYML_TRAIN_SESSION")
    if os.environ.get("LOCAL_RANK") is not None:
        if not session_path:
            raise RuntimeError(
                "Versioned training supports one launcher per host; external ranks are unsupported"
            )
        session = json.loads(Path(session_path).read_text())
        child_cfg = OmegaConf.create(session["configuration"])
        child_cfg.data.snapshot_path = session["snapshot"]
        child_cfg.logger.mlflow.run_id = session["run_id"]
        return _train(child_cfg, evaluation_checkpoint=session.get("evaluation_checkpoint"))
    if cfg.get("seed") is None:
        raise ValueError("Set a seed for reproducible training")
    if not cfg.get("train", True):
        raise ValueError("Use src/eval.py for evaluation without training")
    if int(cfg.trainer.get("num_nodes", 1)) != 1:
        raise ValueError("Versioned training currently requires a single host")
    source = source_provenance(Path(cfg.versioning.repository_dir), Path(cfg.paths.root_dir))
    version = prepare_dataset(cfg)
    if (
        source_provenance(Path(cfg.versioning.repository_dir), Path(cfg.paths.root_dir))["files"]
        != source["files"]
    ):
        raise RuntimeError("Training source changed while the dataset was being prepared")
    # Snapshot binding is a data-module contract, independent of dataset names.
    datamodule = hydra.utils.instantiate(cfg.data, snapshot_path=str(version.snapshot))
    datamodule.prepare_data()
    datamodule.setup()
    L.seed_everything(cfg.seed, workers=True)
    model = hydra.utils.instantiate(cfg.model)
    if cfg.reproducibility.get("initial_state_path"):
        model.load_state_dict(
            torch.load(
                cfg.reproducibility.initial_state_path, map_location="cpu", weights_only=True
            )
        )
    initialization = None
    initial_state = None
    if cfg.get("ckpt_path"):
        initialization = Path(cfg.paths.output_dir) / "provenance" / "initialization.ckpt"
        initialization.parent.mkdir(parents=True, exist_ok=True)
        if Path(cfg.ckpt_path).resolve() != initialization.resolve():
            shutil.copyfile(Path(cfg.ckpt_path), initialization)
        cfg.ckpt_path = str(initialization)
    else:
        initial_state = Path(cfg.paths.output_dir) / "provenance" / "model_state.pt"
        initial_state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), initial_state)
    runtime = runtime_information()
    if cfg.reproducibility.get("expected_environment_path"):
        recorded = json.loads(Path(cfg.reproducibility.expected_environment_path).read_text())
        for key in ("python", "packages"):
            if canonical(recorded[key]) != canonical(runtime[key]):
                raise RuntimeError(
                    f"Recorded {key} environment could not be restored; use the historical locked image"
                )
        changed = [
            key for key in recorded if canonical(recorded[key]) != canonical(runtime.get(key))
        ]
        if changed:
            log.warning(
                f"Reproduction runtime differs in: {', '.join(changed)}; numerical results may differ"
            )
    initialized = initialization_identity(model, initialization)
    expected = cfg.reproducibility.get("expected_initialization")
    if expected and initialized != expected:
        raise ValueError("Restored initialization differs from the original run")
    fingerprint, evaluation, payload = experiment_identity(
        cfg, version.provenance, source, initialized, runtime
    )
    lock = Path(cfg.versioning.coordination_dir) / f"experiment-{fingerprint}.lock"
    with file_lock(lock):
        loggers = utils.instantiate_loggers(cfg.get("logger"))
        mlflow_logger = next((item for item in loggers if isinstance(item, MLFlowLogger)), None)
        if mlflow_logger is None:
            raise ValueError(
                "Versioned training requires the existing MLflow logger; disable versioning for local exploration"
            )
        client = MlflowClient(tracking_uri=str(cfg.logger.mlflow.tracking_uri))
        experiment = client.get_experiment_by_name(str(cfg.logger.mlflow.experiment_name))
        match = None
        if experiment and cfg.reproducibility.reuse and not cfg.reproducibility.force_retrain:
            match = find_reusable(client, experiment.experiment_id, fingerprint)
        if match is not None:
            run, receipt = match
            reused = {
                "reused": True,
                "run_id": run.info.run_id,
                "model_uri": receipt["model_uri"],
                "dataset_version": version.provenance,
            }
            log.info(f"Reusing completed training run {run.info.run_id}: {receipt['model_uri']}")
            if not cfg.test:
                return {
                    k: v for k, v in receipt["metrics"].items() if not k.startswith("test/")
                }, reused
            if receipt["evaluation"] == evaluation:
                return receipt["metrics"], reused
            # Preserve historical results; evaluate the checkpoint in a new run.
            checkpoint = Path(
                client.download_artifacts(
                    run.info.run_id, receipt["checkpoint"], str(Path(cfg.paths.output_dir))
                )
            )
            log_provenance(
                mlflow_logger,
                cfg,
                version,
                source,
                runtime,
                fingerprint,
                evaluation,
                payload,
                initialization or initial_state,
            )
            client.set_tag(
                str(mlflow_logger.run_id), "evaluation.training_run_id", run.info.run_id
            )
            session = Path(cfg.paths.output_dir) / "provenance" / "session.json"
            write_json(
                session,
                {
                    "run_id": mlflow_logger.run_id,
                    "snapshot": str(version.snapshot),
                    "configuration": sanitized(cfg),
                    "evaluation_checkpoint": str(checkpoint),
                },
                preserve_order=True,
            )
            os.environ["DEASYML_TRAIN_SESSION"] = str(session)
            try:
                evaluated, objects = _train(
                    cfg,
                    datamodule=datamodule,
                    model=model,
                    logger=loggers,
                    evaluation_checkpoint=str(checkpoint),
                )
                client.set_terminated(str(mlflow_logger.run_id), "FINISHED")
            except BaseException:
                client.set_terminated(str(mlflow_logger.run_id), "FAILED")
                raise
            finally:
                os.environ.pop("DEASYML_TRAIN_SESSION", None)
            metrics = {k: v for k, v in receipt["metrics"].items() if not k.startswith("test/")}
            metrics.update(evaluated)
            return metrics, {
                **reused,
                "evaluation_run_id": mlflow_logger.run_id,
                "trainer": objects["trainer"],
            }
        try:
            log_provenance(
                mlflow_logger,
                cfg,
                version,
                source,
                runtime,
                fingerprint,
                evaluation,
                payload,
                initialization or initial_state,
            )
            session = Path(cfg.paths.output_dir) / "provenance" / "session.json"
            write_json(
                session,
                {
                    "run_id": mlflow_logger.run_id,
                    "snapshot": str(version.snapshot),
                    "configuration": sanitized(cfg),
                },
                preserve_order=True,
            )
            os.environ["DEASYML_TRAIN_SESSION"] = str(session)
            metrics, objects = _train(cfg, datamodule=datamodule, model=model, logger=loggers)
            if objects["trainer"].is_global_zero:
                receipt = complete_run(
                    mlflow_logger, objects["trainer"], cfg, fingerprint, evaluation, metrics
                )
                objects.update(
                    reused=False,
                    run_id=mlflow_logger.run_id,
                    model_uri=receipt["model_uri"],
                    dataset_version=version.provenance,
                )
            return metrics, objects
        except BaseException:
            client.set_tag(str(mlflow_logger.run_id), "reproducibility.complete", "false")
            client.set_terminated(str(mlflow_logger.run_id), "FAILED")
            raise
        finally:
            os.environ.pop("DEASYML_TRAIN_SESSION", None)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

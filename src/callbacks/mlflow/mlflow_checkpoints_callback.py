"""Upload artifacts; publish the model after test metrics have finalized."""

from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


class MLFlowCheckpointsCallback(Callback):
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        for logger in trainer.loggers:
            if not isinstance(logger, MLFlowLogger):
                continue
            checkpoint = trainer.checkpoint_callback
            if isinstance(checkpoint, ModelCheckpoint):
                for path in {checkpoint.best_model_path, checkpoint.last_model_path}:
                    if path and Path(path).is_file():
                        logger.experiment.log_artifact(logger.run_id, path, "checkpoints")
            logger.experiment.log_artifacts(logger.run_id, "src", "code")
            results = Path(trainer.default_root_dir) / "results"
            if results.is_dir():
                logger.experiment.log_artifacts(logger.run_id, str(results), "results")

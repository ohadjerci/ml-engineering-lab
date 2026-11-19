from typing import Any, Callable, Literal

import torch
from lightning import LightningModule
from torchmetrics import ConfusionMatrix, F1Score, MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall


class ClassificationLitModule(LightningModule):
    """Shared image classification with optional record metadata."""

    def __init__(
        self,
        net: torch.nn.Module | Callable[..., torch.nn.Module],
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[
            ..., torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
        ]
        | None,
        class_mapping: Any = None,
        defective_class: str | None = None,
        monitor: str = "val/loss",
        metric_average: Literal["micro", "macro", "weighted"] = "micro",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.num_classes = len(class_mapping) if class_mapping is not None else 10
        self.class_mapping = (
            {k: list(v) for k, v in class_mapping.items()} if class_mapping is not None else None
        )
        self.weighted_loss = class_mapping is not None
        self.defective_index = (
            list(class_mapping).index(defective_class) if defective_class is not None else None
        )
        configured_count = getattr(net, "keywords", {}).get("n_classes")
        if class_mapping is not None and configured_count not in (None, self.num_classes):
            raise ValueError("Configured network class count conflicts with class_mapping")
        self.net = net if isinstance(net, torch.nn.Module) else net(n_classes=self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.train_precision = Precision(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )
        self.val_precision = Precision(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )
        self.test_precision = Precision(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )

        self.train_recall = Recall(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )

        self.train_f1 = F1Score(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=self.num_classes, average=metric_average
        )

        self.train_confusion = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.val_confusion = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_accuracy_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_accuracy_best.reset()
        self.val_precision_best.reset()
        self.val_recall_best.reset()
        self.val_f1_best.reset()
        self.val_loss_best.reset()

        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def model_step(self, batch: Any):
        x, y = (batch["image"], batch["target"]) if isinstance(batch, dict) else batch
        logits = self.forward(x)
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError("Network logits must have shape [images, configured classes]")
        loss = self.criterion(logits, y)
        self.batch_probabilities = logits.detach().softmax(dim=1)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss, weight=len(targets) if self.weighted_loss else 1)
        self.train_accuracy(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)
        self.train_confusion(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss, weight=len(targets) if self.weighted_loss else 1)
        self.val_accuracy(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        self.val_confusion(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss, weight=len(targets) if self.weighted_loss else 1)
        self.test_accuracy(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        self.test_confusion(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        self._log_defective_metrics("train")
        # Preserve the original metric names used by checkpoint consumers and tests.
        self.log("train/acc", self.train_accuracy.compute())

    def on_validation_epoch_end(self):
        self._log_defective_metrics("val")
        self.log("val/acc", self.val_accuracy.compute())
        self.val_accuracy_best(self.val_accuracy.compute())  # update best so far val acc
        self.val_precision_best(self.val_precision.compute())
        self.val_recall_best(self.val_recall.compute())
        self.val_f1_best(self.val_f1.compute())
        self.val_loss_best(self.val_loss.compute())
        self.log("val_best/accuracy", self.val_accuracy_best.compute(), prog_bar=True)
        self.log("val_best/precision", self.val_precision_best.compute(), prog_bar=True)
        self.log("val_best/recall", self.val_recall_best.compute(), prog_bar=True)
        self.log("val_best/f1", self.val_f1_best.compute(), prog_bar=True)
        self.log("val_best/loss", self.val_loss_best.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self._log_defective_metrics("test")
        self.log("test/acc", self.test_accuracy.compute())

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams["monitor"],
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_train_epoch_start(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        sampler = getattr(datamodule, "sampler", None)
        if sampler is not None:
            sampler.set_epoch(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        if self.class_mapping is not None:
            checkpoint["class_mapping"] = self.class_mapping

    def on_load_checkpoint(self, checkpoint):
        if self.class_mapping is not None and (
            checkpoint.get("class_mapping") != self.class_mapping
            or list(checkpoint.get("class_mapping", {})) != list(self.class_mapping)
        ):
            raise ValueError("Checkpoint class mapping does not match configured targets")

    def _log_defective_metrics(self, phase: str) -> None:
        if self.defective_index is None:
            return
        matrix = getattr(self, f"{phase}_confusion").compute()
        i = self.defective_index
        tp = matrix[i, i].float()
        self.log(f"{phase}/defective_precision", tp / matrix[:, i].sum().clamp_min(1))
        self.log(f"{phase}/defective_recall", tp / matrix[i, :].sum().clamp_min(1))
        self.log(
            f"{phase}/defective_f1",
            2 * tp / (matrix[:, i].sum() + matrix[i, :].sum()).clamp_min(1),
        )
        getattr(self, f"{phase}_confusion").reset()

    def on_fit_start(self):
        if not self.weighted_loss:
            return
        dm = getattr(self.trainer, "datamodule", None)
        shape = getattr(dm, "input_shape", None)
        if shape is None:
            raise ValueError("PCB model requires a datamodule input_shape")
        training = self.net.training
        try:
            self.net.eval()
            with torch.inference_mode():
                output = self(torch.zeros(1, *shape, device=self.device))
            if output.shape != (1, self.num_classes):
                raise ValueError("Network output shape conflicts with class_mapping")
        except RuntimeError as error:
            raise ValueError(
                f"Network is incompatible with configured input shape {shape}"
            ) from error
        finally:
            self.net.train(training)

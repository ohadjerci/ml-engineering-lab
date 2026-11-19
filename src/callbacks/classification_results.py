"""Persist PCB diagnostics before the existing MLflow upload callback runs."""

import json
from collections import Counter
from pathlib import Path

from lightning.pytorch import Callback


class ClassificationResults(Callback):
    def _directory(self, trainer):
        path = Path(trainer.default_root_dir) / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _data(self, trainer):
        directory = self._directory(trainer)
        dm = trainer.datamodule
        (directory / "splits.json").write_text(json.dumps(dm.split_state, indent=2))
        (directory / "data_summary.json").write_text(json.dumps(dm.summary(), indent=2))

    def on_validation_start(self, trainer, pl_module):
        self._data(trainer)

    def on_fit_start(self, trainer, pl_module):
        self._data(trainer)

    def on_train_epoch_start(self, trainer, pl_module):
        self.seen_train = []
        self.observed_batches = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        records = batch["metadata"]
        self.seen_train.extend(record["image_path"] for record in records)
        sampler = trainer.datamodule.sampler
        self.observed_batches.append(
            {
                "images": len(records),
                "board_groups": sorted({record["board_group"] for record in records}),
                "groups": len({record["board_group"] for record in records}),
                "defective_fraction": sum(
                    record["target"] == sampler.positive_index for record in records
                )
                / len(records),
            }
        )

    def on_train_epoch_end(self, trainer, pl_module):
        dm = trainer.datamodule
        sampler = dm.sampler
        records = dm.data_train.records
        stats = self.observed_batches
        group_counts = Counter(g for batch in stats for g in batch["board_groups"])
        expected_groups = {record["board_group"] for record in records}
        result = {
            "epoch": trainer.current_epoch,
            "target_defective_fraction": sampler.target,
            "batches": stats,
            "repeated_groups": sum(count - 1 for count in group_counts.values()),
            "omitted_groups": len(expected_groups - set(group_counts)),
            "repeated_images": len(self.seen_train) - len(set(self.seen_train)),
            "omitted_images": len(records) - len(set(self.seen_train)),
        }
        for batch in stats:
            batch["target_deviation"] = batch["defective_fraction"] - sampler.target
        (self._directory(trainer) / f"sampling_epoch_{trainer.current_epoch:03d}.json").write_text(
            json.dumps(result, indent=2)
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            (self._directory(trainer) / "validation_confusion.json").write_text(
                json.dumps(pl_module.val_confusion.compute().cpu().tolist())
            )

    def on_test_start(self, trainer, pl_module):
        self._data(trainer)
        self.predictions = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        probabilities = pl_module.batch_probabilities.cpu().tolist()
        for record, probs in zip(batch["metadata"], probabilities):
            self.predictions.append(
                {
                    **record,
                    "probabilities": probs,
                    "prediction": max(range(len(probs)), key=probs.__getitem__),
                    "predicted_class": trainer.datamodule.classes[
                        max(range(len(probs)), key=probs.__getitem__)
                    ],
                }
            )

    def on_test_epoch_end(self, trainer, pl_module):
        directory = self._directory(trainer)
        paths = [record["image_path"] for record in self.predictions]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate image predictions during PCB evaluation")
        (directory / "test_predictions.json").write_text(json.dumps(self.predictions, indent=2))
        (directory / "test_confusion.json").write_text(
            json.dumps(pl_module.test_confusion.compute().cpu().tolist())
        )

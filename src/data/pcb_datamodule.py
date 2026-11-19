"""PCB manifest ingestion, isolated splitting, and independent batch scheduling."""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.components.group_batch_sampler import GroupBatchSampler
from src.data.components.pcb_dataset import PCBDataset, collate_pcb, label_lookup, read_manifest


class PCBDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        manifest_path: str,
        name: str,
        class_mapping: Any,
        preprocessing: Any,
        split: Any,
        sampling: Any,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        def plain(value):
            return (
                OmegaConf.to_container(value, resolve=True)
                if OmegaConf.is_config(value)
                else value
            )

        self.class_mapping = plain(class_mapping)
        label_lookup(self.class_mapping)
        self.classes = list(self.class_mapping)
        self.num_classes = len(self.classes)
        self.name = name
        self.preprocessing = plain(preprocessing)
        self.split_config = plain(split)
        self.sampling_config = plain(sampling)
        p = self.preprocessing
        if (
            p["channels"] not in (1, 3)
            or len(p["image_size"]) != 2
            or any(not isinstance(n, int) or n < 1 for n in p["image_size"])
        ):
            raise ValueError("PCB requires channels 1 or 3 and two positive image dimensions")
        if (
            any(len(p[k]) != p["channels"] for k in ("mean", "std"))
            or any(not math.isfinite(v) for k in ("mean", "std") for v in p[k])
            or any(v <= 0 for v in p["std"])
        ):
            raise ValueError("Normalization needs finite channel-matched mean and positive std")
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative")
        if self.sampling_config["defective_class"] not in self.classes:
            raise ValueError("sampling.defective_class must name a configured target class")
        sizes = self.split_config["proportions"]
        if (
            len(sizes) != 3
            or any(not math.isfinite(x) or x <= 0 for x in sizes)
            or not math.isclose(sum(sizes), 1)
        ):
            raise ValueError("split.proportions must contain three positive values summing to one")
        interpolation = InterpolationMode(p["interpolation"])
        base = [transforms.Resize(p["image_size"], interpolation=interpolation)]
        augmentations = [hydra.utils.instantiate(item) for item in p["train_transforms"]]
        tail = [transforms.ToTensor(), transforms.Normalize(p["mean"], p["std"])]
        self.transforms_train = transforms.Compose(base + augmentations + tail)
        self.transforms_test = transforms.Compose(base + tail)
        self.input_shape = [p["channels"], *p["image_size"]]
        self.data_train: PCBDataset | None = None
        self.data_val: PCBDataset | None = None
        self.data_test: PCBDataset | None = None
        self.sampler: GroupBatchSampler | None = None
        self.split_state: dict[str, Any] = {}

    def prepare_data(self) -> None:
        # Local files only: no implicit download or extraction.
        if not Path(self.hparams["manifest_path"]).is_file():
            raise ValueError(f"Manifest not found: {self.hparams['manifest_path']}")

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None and self.trainer.world_size != 1:
            raise ValueError("PCB complete-group batching requires a single device and node")
        if self.data_train is not None:
            return
        info = read_manifest(
            self.hparams["root_dir"], self.hparams["manifest_path"], self.class_mapping
        )
        self.manifest_info = info
        records = info["records"]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        key = self.split_config["group_by"]
        for record in records:
            if key not in record or not isinstance(record[key], str):
                raise ValueError(f"Invalid split grouping field: {key}")
            groups[record[key]].append(record)
        if len(groups) < 3:
            raise ValueError("At least three split groups are required")
        names = ("train", "val", "test")
        order = sorted(groups)
        random.Random(self.split_config["seed"]).shuffle(order)
        order.sort(key=lambda gid: -len(groups[gid]))
        totals = np.bincount([r["target"] for r in records], minlength=self.num_classes)
        desired = (
            np.asarray(self.split_config["proportions"])[:, None] * np.r_[len(records), totals]
        )
        counts = np.zeros_like(desired)
        assignment = {}
        for offset, gid in enumerate(order):
            subset = groups[gid]
            vector = np.r_[
                len(subset), np.bincount([r["target"] for r in subset], minlength=self.num_classes)
            ]
            candidates = list(range(3))
            empty = [i for i in candidates if counts[i, 0] == 0]
            if len(order) - offset == len(empty):
                candidates = empty

            def objective(i):
                proposed = counts.copy()
                proposed[i] += vector
                return float(np.square((proposed - desired) / np.maximum(desired, 1)).sum())

            chosen = min(candidates, key=objective)
            assignment[gid] = names[chosen]
            counts[chosen] += vector
        # Large defect-heavy groups are allocated first. Revisit earlier choices after
        # the smaller OK-heavy groups are known, avoiding avoidable proportion drift.
        vectors = {
            gid: np.r_[
                len(groups[gid]),
                np.bincount([r["target"] for r in groups[gid]], minlength=self.num_classes),
            ]
            for gid in order
        }
        scale = np.maximum(desired, 1)
        while True:
            score = float(np.square((counts - desired) / scale).sum())
            best_move = None
            for gid in order:
                origin = names.index(assignment[gid])
                for destination in range(3):
                    if origin == destination:
                        continue
                    proposed = counts.copy()
                    proposed[origin] -= vectors[gid]
                    proposed[destination] += vectors[gid]
                    if np.any(proposed[:, 1:] <= 0):
                        continue
                    candidate = float(np.square((proposed - desired) / scale).sum())
                    if candidate < score - 1e-12:
                        score = candidate
                        best_move = (gid, destination, proposed)
            if best_move is None:
                break
            gid, destination, counts = best_move
            assignment[gid] = names[destination]
        self.split_state = {
            "fingerprint": info["fingerprint"],
            "class_mapping": self.class_mapping,
            "classes": self.classes,
            "preprocessing": self.preprocessing,
            "split": {k: v for k, v in self.split_config.items() if k != "assignments_path"},
            "assignments": {r["image_path"]: assignment[r[key]] for r in records},
        }
        saved = self.split_config["assignments_path"]
        if saved:
            self._validate_state(json.loads(Path(saved).read_text()))
        self._build_datasets()

    def _validate_state(self, state: dict[str, Any]) -> None:
        for key in ("fingerprint", "class_mapping", "classes", "split", "preprocessing"):
            if state.get(key) != self.split_state[key]:
                raise ValueError(f"PCB checkpoint/saved split has incompatible {key}")
        assigned = state.get("assignments", {})
        if set(assigned) != set(self.split_state["assignments"]) or set(assigned.values()) != {
            "train",
            "val",
            "test",
        }:
            raise ValueError("Saved split must assign every image once to train/val/test")
        self.split_state = state

    def _build_datasets(self) -> None:
        records = self.manifest_info["records"]
        assignments = self.split_state["assignments"]
        for field in ("board_group", "capture_group", self.split_config["group_by"]):
            seen: dict[str, str] = {}
            for record in records:
                split = assignments[record["image_path"]]
                if seen.setdefault(record[field], split) != split:
                    raise ValueError(f"Split leaks {field} {record[field]} across partitions")
        for name in ("train", "val", "test"):
            subset = [r for r in records if assignments[r["image_path"]] == name]
            if not subset or len({r["target"] for r in subset}) != self.num_classes:
                raise ValueError(f"Split {name} must contain images of both target classes")
            setattr(
                self,
                f"data_{name}",
                PCBDataset(
                    self.hparams["root_dir"],
                    subset,
                    self.transforms_train if name == "train" else self.transforms_test,
                    self.classes,
                    self.preprocessing["channels"],
                ),
            )
        assert self.data_train is not None
        s = self.sampling_config
        self.sampler = GroupBatchSampler(
            self.data_train.records,
            self.hparams["batch_size"],
            s["target_defective_fraction"],
            self.classes.index(s["defective_class"]),
            s["seed"],
            s["oversized"],
        )

    def _epoch(self) -> int:
        return self.trainer.current_epoch if self.trainer is not None else 0

    def summary(self) -> dict[str, Any]:
        result = {
            "meta": self.manifest_info["meta"],
            "fingerprint": self.manifest_info["fingerprint"],
            "preprocessing": self.preprocessing,
            "classes": self.classes,
        }
        for name in ("train", "val", "test"):
            dataset = getattr(self, f"data_{name}")
            result[name] = {
                "images": len(dataset),
                "groups": len({r["board_group"] for r in dataset.records}),
                "labels": dict(Counter(r["anomaly_class"] for r in dataset.records)),
                "targets": dict(Counter(r["target"] for r in dataset.records)),
                "group_size_histogram": dict(
                    Counter(Counter(r["board_group"] for r in dataset.records).values())
                ),
            }
        return result

    def _loader(self, dataset, training=False):
        options = dict(
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            collate_fn=collate_pcb,
        )
        if training:
            assert self.sampler is not None
            self.sampler.epoch_provider = self._epoch
            return DataLoader(dataset, batch_sampler=self.sampler, **options)
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            drop_last=False,
            **options,
        )

    def train_dataloader(self):
        return self._loader(self.data_train, training=True)

    def val_dataloader(self):
        return self._loader(self.data_val)

    def test_dataloader(self):
        return self._loader(self.data_test)

    def state_dict(self) -> dict[str, Any]:
        return self.split_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.setup()
        self._validate_state(state_dict)
        self._build_datasets()

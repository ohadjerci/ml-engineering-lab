"""Coverage-first, image-balanced packing of indivisible training groups."""

import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from torch.utils.data import Sampler


class GroupBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        batch_size: int,
        target_defective_fraction: float,
        defective_index: int,
        seed: int,
        oversized: str,
    ):
        if batch_size < 1 or not 0 <= target_defective_fraction <= 1:
            raise ValueError("batch_size must be positive and target fraction must be in [0, 1]")
        if oversized not in {"error", "singleton"}:
            raise ValueError("oversized must be 'error' or 'singleton'")
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[record["board_group"]].append(index)
        if oversized == "error" and any(len(indices) > batch_size for indices in groups.values()):
            raise ValueError("A training S group exceeds batch_size; increase it or use singleton")
        self.groups = dict(sorted(groups.items()))
        self.positive = {
            gid: sum(records[i]["target"] == defective_index for i in indices)
            for gid, indices in self.groups.items()
        }
        self.batch_size = batch_size
        self.positive_index = defective_index
        self.target = target_defective_fraction
        self.seed = seed
        self.epoch = 0
        self.epoch_provider: Callable[[], int] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batches(self) -> list[list[int]]:
        remaining = list(self.groups)
        random.Random(self.seed + self.epoch).shuffle(remaining)
        batches = []
        while remaining:
            batch: list[int] = []
            positives = 0
            while remaining:
                fitting = [
                    g for g in remaining if len(batch) + len(self.groups[g]) <= self.batch_size
                ]
                if not fitting:
                    if not batch:
                        gid = remaining.pop(0)
                        batch.extend(self.groups[gid])
                    break
                gid = min(
                    fitting,
                    key=lambda g: abs(
                        (positives + self.positive[g]) / (len(batch) + len(self.groups[g]))
                        - self.target
                    ),
                )
                batch.extend(self.groups[gid])
                positives += self.positive[gid]
                remaining.remove(gid)
            batches.append(batch)
        return batches

    def __iter__(self):
        if self.epoch_provider is not None:
            self.set_epoch(self.epoch_provider())
        yield from self.batches()

    def __len__(self) -> int:
        return len(self.batches())

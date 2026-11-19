"""Image-level PCB records. Directory names never determine targets."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


def label_lookup(class_mapping: Mapping[str, Sequence[str]]) -> dict[str, int]:
    if (
        not isinstance(class_mapping, Mapping)
        or len(class_mapping) != 2
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(labels, Sequence)
            or isinstance(labels, str)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
            for name, labels in class_mapping.items()
        )
    ):
        raise ValueError("PCB class_mapping must contain two nonempty target classes")
    lookup = {}
    for target, labels in enumerate(class_mapping.values()):
        for label in labels:
            if label in lookup:
                raise ValueError(f"Anomaly label appears in multiple classes: {label}")
            lookup[label] = target
    return lookup


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path must be relative to image root: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing dataset file: {path}")
    return path


def read_manifest(root_dir: str, manifest_path: str, class_mapping: Mapping) -> dict[str, Any]:
    root = Path(root_dir)
    lookup = label_lookup(class_mapping)
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read PCB manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("PCB manifest must be a JSON object")
    records, group_ids, paths = [], set(), set()
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Manifest must contain a nonempty groups list")
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Each manifest group must be an object")
        gid = group.get("board_group")
        if not isinstance(gid, str) or not gid or gid in group_ids:
            raise ValueError(f"Invalid or duplicate board_group: {gid}")
        group_ids.add(gid)
        if not isinstance(group.get("images"), list) or not group["images"]:
            raise ValueError(f"Empty or invalid images in group {gid}")
        for source in group["images"]:
            if not isinstance(source, dict):
                raise ValueError(f"Group {gid}: each image must be an object")
            record = dict(source)
            for field in ("anomaly_class", "image_path", "capture_group", "view_id"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"Group {gid}: missing or invalid {field}")
            path = record["image_path"]
            canonical = str(checked_path(root, path))
            if canonical in paths:
                raise ValueError(f"Duplicate image path: {path}")
            paths.add(canonical)
            label = record["anomaly_class"]
            if label not in lookup:
                raise ValueError(f"Unknown anomaly_class {label!r} for {path}")
            if record.get("mask_path") is not None:
                if not isinstance(record["mask_path"], str):
                    raise ValueError(f"Invalid mask_path for {path}")
                checked_path(root, record["mask_path"])
            record.update(board_group=gid, target=lookup[label])
            records.append(record)
    records.sort(key=lambda record: record["image_path"])
    # Include all original manifest fields, but make group/image ordering immaterial.
    canonical_groups = []
    for group in sorted(groups, key=lambda item: item["board_group"]):
        canonical_groups.append(
            {**group, "images": sorted(group["images"], key=lambda item: item["image_path"])}
        )
    canonical_manifest = {**manifest, "groups": canonical_groups}
    fingerprint = hashlib.sha256(
        json.dumps(canonical_manifest, sort_keys=True).encode()
    ).hexdigest()
    return {
        "records": records,
        "meta": manifest.get("meta", {}),
        "fingerprint": fingerprint,
        "label_counts": dict(Counter(r["anomaly_class"] for r in records)),
        "group_sizes": dict(Counter(r["board_group"] for r in records)),
    }


class PCBDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        records: list[dict[str, Any]],
        transform: Any,
        classes: list[str],
        channels: int,
    ):
        self.root = Path(root_dir)
        self.records = records
        self.transform = transform
        self.classes = classes
        self.channels = channels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(self.root / record["image_path"]) as image:
            tensor = self.transform(image.convert("RGB" if self.channels == 3 else "L"))
        return {"image": tensor, "target": record["target"], "metadata": dict(record)}


def collate_pcb(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "target": torch.tensor([s["target"] for s in samples], dtype=torch.long),
        "metadata": [s["metadata"] for s in samples],
    }

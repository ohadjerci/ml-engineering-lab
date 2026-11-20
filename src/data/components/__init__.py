"""Portable dataset manifests and immutable, content-addressed snapshots."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Snapshot path escapes its root: {relative}")
    if path.is_symlink():
        raise ValueError(f"Snapshot symlinks are unsupported: {relative}")
    return path


def copy_file(source: Path, root: Path, relative: str) -> None:
    target = safe_path(root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def write_json(path: Path, value: Any, *, preserve_order: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preserve_order:
        path.write_text(json.dumps(value, allow_nan=False))
    else:
        path.write_bytes(canonical(value))


def build_manifest(root: Path, description: dict[str, Any]) -> dict[str, Any]:
    """Adapter output contains records, ordered classes and explicit ordered split IDs."""
    files = {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }
    payload = {"schema_version": SCHEMA_VERSION, **description, "files": files}
    payload["split_info"] = {
        name: {"identity": digest(description["splits"][name]), "samples": len(ids)}
        for name, ids in description["splits"].items()
    }
    # A name is a logical MLflow namespace, not an alternative to hashing bytes.
    payload["identity"] = digest({k: v for k, v in payload.items() if k != "name"})
    write_json(root / "manifest.json", payload)
    verify_snapshot(root)
    return payload


def verify_snapshot(root: Path, expected: str | None = None) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset manifest schema")
    identity = digest({k: v for k, v in manifest.items() if k not in {"name", "identity"}})
    if identity != manifest["identity"] or (expected is not None and identity != expected):
        raise ValueError("Dataset manifest identity mismatch")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Dataset snapshot has missing or additional files")
    for relative, checksum in manifest["files"].items():
        if file_hash(safe_path(root, relative)) != checksum:
            raise ValueError(f"Dataset file hash mismatch: {relative}")
    records = manifest["records"]
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records) or set(manifest["splits"]) != set(SPLITS):
        raise ValueError("Dataset requires unique sample IDs and train/val/test splits")
    assigned = [sid for name in SPLITS for sid in manifest["splits"][name]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_id):
        raise ValueError("Splits must partition every sample exactly once")
    for name in SPLITS:
        ids = manifest["splits"][name]
        if manifest["split_info"][name] != {"identity": digest(ids), "samples": len(ids)}:
            raise ValueError("Dataset split identity mismatch")
    for record in records:
        for key in ("image_path", "mask_path"):
            if record.get(key) and record[key] not in manifest["files"]:
                raise ValueError(f"Untracked sample file: {record[key]}")
        if not 0 <= record["target"] < len(manifest["class_mapping"]):
            raise ValueError("Sample target is outside the class mapping")
    return manifest

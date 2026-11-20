import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.components import build_manifest, copy_file, verify_snapshot, write_json
from src.data.mnist_datamodule import MNISTDataModule
from src.data.prepare_dataset import prepare_dataset, restore_dataset
from src.utils.utils import git


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


class FixtureDataModule:
    """Small explicit source fixture; production orchestration is format-independent."""

    def __init__(self, root_dir, name="fixture", snapshot_path=None):
        self.root = Path(root_dir)
        self.name = name

    def prepare_data(self):
        pass

    def setup(self):
        pass

    def export_snapshot(self, destination):
        description = json.loads((self.root / "annotations.json").read_text())
        description["name"] = self.name
        for record in description["records"]:
            copy_file(self.root / record["image_path"], destination, record["image_path"])
        return description


@pytest.fixture
def dataset_repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "Fixture")
    git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / ".gitignore").write_text(
        "/data/**\n!/data/**/\n!/data/**/*.dvc\n!/data/**/.gitignore\n"
    )
    (repository / "unrelated.txt").write_text("original")
    git(repository, "add", "--", ".gitignore", "unrelated.txt")
    git(repository, "commit", "-m", "Initial fixture")
    source = tmp_path / "source"
    source.mkdir()
    for i in range(3):
        (source / f"{i}.png").write_bytes(bytes([i, 42]))
    write_json(
        source / "annotations.json",
        {
            "annotation_format": "fixture-v1",
            "class_mapping": [{"name": "one", "index": 0}],
            "records": [{"id": str(i), "image_path": f"{i}.png", "target": 0} for i in range(3)],
            "splits": {"train": ["0"], "val": ["1"], "test": ["2"]},
        },
    )
    cfg = OmegaConf.create(
        {
            "data": {
                "_target_": "tests.test_datamodules.FixtureDataModule",
                "root_dir": str(source),
                "name": "fixture",
            },
            "versioning": {
                "enabled": True,
                "repository_dir": str(repository),
                "snapshot_dir": "data/snapshots",
                "cache_dir": str(tmp_path / "cache"),
                "coordination_dir": str(tmp_path / "locks"),
                "remote": {"name": "fixture", "url": str(tmp_path / "remote")},
                "upload": True,
                "git_commit": True,
                "git_push": False,
                "restore": None,
            },
        }
    )
    return cfg, repository, source


def test_dataset_publication_idempotent_and_git_index_isolated(dataset_repository):
    cfg, repository, source = dataset_repository
    (repository / "unrelated.txt").write_text("staged user work")
    git(repository, "add", "--", "unrelated.txt")
    first = prepare_dataset(cfg)
    second = prepare_dataset(cfg)
    assert first.manifest["identity"] == second.manifest["identity"]
    assert first.provenance["metadata_revision"] == second.provenance["metadata_revision"]
    assert git(repository, "show", "HEAD:unrelated.txt") == "original"
    assert git(repository, "show", ":unrelated.txt") == "staged user work"
    assert git(repository, "diff", "--cached", "--name-only") == "unrelated.txt"
    changed = git(
        repository, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"
    ).splitlines()
    assert all(p.startswith(".dvc") or p.endswith((".dvc", ".gitignore")) for p in changed)
    (source / "0.png").write_bytes(b"changed after snapshot")
    assert (first.snapshot / "0.png").read_bytes() == bytes([0, 42])
    assert verify_snapshot(first.snapshot)["identity"] == first.manifest["identity"]


def test_concurrent_dataset_publication_has_one_metadata_version(dataset_repository):
    import multiprocessing

    cfg, repository, _ = dataset_repository
    context = multiprocessing.get_context("fork")
    results = context.Queue()

    def publish():
        try:
            version = prepare_dataset(cfg)
            results.put((version.provenance["identity"], version.provenance["metadata_revision"]))
        except Exception as error:
            results.put(("error", str(error)))

    processes = [context.Process(target=publish) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        published = [results.get(timeout=90) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)
    assert published[0] == published[1] and published[0][0] != "error"
    assert git(repository, "rev-list", "--count", "HEAD") == "2"


@pytest.mark.parametrize("change", ["image", "label", "mapping", "split", "annotation"])
def test_dataset_identity_tracks_all_consumed_content(dataset_repository, tmp_path, change):
    cfg, _, source = dataset_repository
    module = FixtureDataModule(source)
    original = tmp_path / "original"
    original.mkdir()
    description = module.export_snapshot(original)
    # Keep two classes so label and class-order changes are valid.
    description["class_mapping"].append({"name": "two", "index": 1})
    baseline = build_manifest(original, description)
    modified = tmp_path / "modified"
    modified.mkdir()
    mutated = copy.deepcopy(description)
    for item in mutated["records"]:
        copy_file(source / item["image_path"], modified, item["image_path"])
    if change == "image":
        (modified / "0.png").write_bytes(b"other image")
    elif change == "label":
        mutated["records"][0]["target"] = 1
    elif change == "mapping":
        mutated["class_mapping"][0]["name"] = "renamed"
    elif change == "split":
        mutated["splits"]["train"], mutated["splits"]["val"] = ["1"], ["0"]
    else:
        mutated["records"][0]["annotation"] = {"box": [0, 0, 1, 1]}
    assert build_manifest(modified, mutated)["identity"] != baseline["identity"]


def test_provenance_restores_and_verifies_without_source(dataset_repository, tmp_path):
    import shutil

    cfg, repository, source = dataset_repository
    published = prepare_dataset(cfg)
    provenance = tmp_path / "data_version.json"
    write_json(provenance, published.provenance)
    checkout = tmp_path / "restore"
    git(tmp_path, "clone", str(repository), str(checkout))
    replay = cfg.copy()
    replay.versioning.repository_dir = str(checkout)
    replay.versioning.cache_dir = str(tmp_path / "fresh-cache")
    shutil.rmtree(source)
    restored = restore_dataset(replay, provenance)
    assert restored.manifest == published.manifest
    (restored.snapshot / "0.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_snapshot(restored.snapshot)


def test_dataset_stage_preserves_user_metadata_changes(dataset_repository):
    cfg, repository, _ = dataset_repository
    prepare_dataset(cfg)
    (repository / ".dvc/config").write_text("[core]\n    autostage = true\n")
    with pytest.raises(RuntimeError, match="user changes"):
        prepare_dataset(cfg)


@pytest.mark.parametrize("operation", ["export", "upload", "commit"])
def test_dataset_failures_prevent_training(dataset_repository, monkeypatch, operation):
    import importlib
    from unittest.mock import Mock

    stage = importlib.import_module("src.data.prepare_dataset")
    training = importlib.import_module("src.train")
    cfg, repository, _ = dataset_repository
    cfg.paths = {"root_dir": str(repository), "output_dir": str(repository / "outputs")}
    cfg.trainer = {"num_nodes": 1}
    cfg.seed = 1
    cfg.train = True
    monkeypatch.setattr(training, "source_provenance", lambda *a: {})
    fitting = Mock()
    monkeypatch.setattr(training, "_train", fitting)
    if operation == "export":
        monkeypatch.setattr(
            FixtureDataModule, "export_snapshot", Mock(side_effect=RuntimeError("snapshot failed"))
        )
    elif operation == "upload":
        execute = stage.command

        def fail_upload(root, *args, **kwargs):
            if args[:2] == ("dvc", "push"):
                raise RuntimeError("upload failed")
            return execute(root, *args, **kwargs)

        monkeypatch.setattr(stage, "command", fail_upload)
    else:
        monkeypatch.setattr(
            stage, "commit_metadata", Mock(side_effect=RuntimeError("commit failed"))
        )
    with pytest.raises(RuntimeError, match="failed"):
        training.train(cfg)
    fitting.assert_not_called()


@pytest.mark.parametrize("supported", [True, False])
def test_reproduction_uses_isolated_recorded_code(
    dataset_repository, tmp_path, monkeypatch, supported
):
    from mlflow import MlflowClient

    from src.data.components import file_hash
    from src.data.prepare_dataset import prepare_reproduction

    cfg, repository, _ = dataset_repository
    entry = repository / "src/data/prepare_dataset.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        "REPRODUCTION_SCHEMA = 1\n" if supported else "# Historical code without replay\n"
    )
    git(repository, "add", "--", "src/data/prepare_dataset.py")
    git(repository, "commit", "-m", "Add historical reproduction capability")
    code_revision = git(repository, "rev-parse", "HEAD")
    version = prepare_dataset(cfg)
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/tracking.db")
    experiment = client.create_experiment(
        "replay", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    run_id = client.create_run(experiment).info.run_id
    replay_cfg = {
        "data": {"class_mapping": {"z-first": ["Z"], "a-second": ["A"]}},
        "paths": {},
        "trainer": {},
        "callbacks": {},
        "versioning": OmegaConf.to_container(cfg.versioning),
        "reproducibility": {},
        "logger": {"mlflow": {}},
        "ckpt_path": None,
    }
    source = {
        "revision": code_revision,
        "repository": "",
        "files": {"src/data/prepare_dataset.py": file_hash(entry)},
    }
    for name, contents in {
        "config": replay_cfg,
        "data_version": version.provenance,
        "source": source,
        "fingerprint": {"initialization": "weights"},
        "environment": {"python": "3.12"},
    }.items():
        client.log_dict(run_id, contents, f"provenance/{name}.json")
    state = tmp_path / "model_state.pt"
    torch.save({"weight": torch.ones(1)}, state)
    client.log_artifact(run_id, str(state), "provenance/initialization")
    client.set_terminated(run_id, "FINISHED")
    before = client.get_run(run_id)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", client.tracking_uri)
    monkeypatch.setenv("GIT_WORKTREE", str(repository))
    destination = tmp_path / "reproduction"
    if not supported:
        with pytest.raises(ValueError, match="does not support"):
            prepare_reproduction(run_id, destination)
        assert git(repository, "rev-parse", "HEAD") == version.provenance["metadata_revision"]
        return
    prepare_reproduction(run_id, destination)
    checkout = destination / "checkout"
    assert git(checkout, "rev-parse", "HEAD") == code_revision
    replay = json.loads((checkout / "replay.json").read_text())
    assert list(replay["data"]["class_mapping"]) == ["z-first", "a-second"]
    assert replay["reproducibility"]["force_retrain"] is True
    assert replay["reproducibility"]["reuse"] is False
    assert replay["reproducibility"]["original_run_id"] == run_id
    assert git(repository, "rev-parse", "HEAD") == version.provenance["metadata_revision"]
    after = client.get_run(run_id)
    assert after.data.tags == before.data.tags
    assert after.data.params == before.data.params
    assert after.data.metrics == before.data.metrics
    assert client.get_run(run_id).info.status == before.info.status


def test_mnist_snapshot_matches_raw_training_samples(tmp_path, monkeypatch):
    from PIL import Image
    from torch.utils.data import Dataset

    import src.data.mnist_datamodule as module

    class TinyMNIST(Dataset):
        classes = [str(i) for i in range(10)]

        def __init__(self, root, train=True, transform=None, download=False):
            self.data = torch.full((3, 28, 28), 42 if train else 128, dtype=torch.uint8)
            self.targets = torch.tensor([0, 1, 2])
            self.transform = transform

        def __len__(self):
            return 3

        def __getitem__(self, index):
            image = Image.fromarray(self.data[index].numpy())
            return self.transform(image) if self.transform else image, int(self.targets[index])

    monkeypatch.setattr(module, "MNIST", TinyMNIST)
    original = MNISTDataModule(train_val_test_split=(2, 2, 2))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    build_manifest(snapshot, original.export_snapshot(snapshot))
    restored = MNISTDataModule(snapshot_path=str(snapshot), train_val_test_split=(2, 2, 2))
    restored.prepare_data()
    restored.setup()
    for split in ("train", "val", "test"):
        expected = getattr(original, f"data_{split}")
        actual = getattr(restored, f"data_{split}")
        for index in range(len(expected)):
            torch.testing.assert_close(expected[index][0], actual[index][0])
            assert expected[index][1] == actual[index][1]

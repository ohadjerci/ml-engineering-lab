"""PCB contracts use small synthetic images; the real dataset is never needed in CI."""

import copy
import json
from collections import Counter, defaultdict
from functools import partial

import hydra
import pytest
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from PIL import Image

from src.data.components.group_batch_sampler import GroupBatchSampler
from src.data.components.pcb_dataset import read_manifest
from src.eval import evaluate
from src.models.classification_module import ClassificationLitModule
from src.train import train
from src.utils.model_registry import extract_model_metadata


@pytest.fixture
def pcb(tmp_path):
    groups = []
    for g in range(18):
        images = []
        for i in range(2 + g % 3):
            label = "OK" if i % 2 == 0 else "HS"
            # Deliberately keep every image, including OK images, under an NG path.
            path = f"NG/S{g:04d}/image_{i}.jpg"
            absolute = tmp_path / path
            absolute.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), (i * 20, g * 10, 100)).save(absolute)
            images.append(
                dict(
                    anomaly_class=label,
                    image_path=path,
                    mask_path=None,
                    capture_group=f"capture_{g}",
                    view_id=f"C{i}",
                    extra="retained",
                )
            )
        groups.append(dict(board_group=f"S{g:04d}", images=images))
    manifest = tmp_path / "pcb.json"
    manifest.write_text(
        json.dumps(dict(meta={"prefix": "pcb/", "note": "retained"}, groups=groups))
    )
    return tmp_path, manifest


def pcb_config(pcb, mode="train"):
    root, manifest = pcb
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name=mode,
            overrides=["experiment=pcb", "trainer=cpu"],
            return_hydra_config=True,
        )
    with open_dict(cfg):
        cfg.logger = None
        cfg.data.root_dir = str(root)
        cfg.data.manifest_path = str(manifest)
        cfg.data.preprocessing.image_size = [16, 16]
        cfg.data.num_workers = 0
        cfg.data.batch_size = 8
        cfg.model.net.stem_channels = 4
        cfg.model.net.bottleneck_dims = [4, 4, 4, 4]
        cfg.trainer.max_epochs = 1
        cfg.trainer.enable_progress_bar = False
        cfg.callbacks.pop("rich_progress_bar", None)
        cfg.registry.enabled = False
        cfg.paths.output_dir = str(root / mode)
        cfg.paths.log_dir = str(root / mode)
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
    return cfg


def test_manifest_labels_metadata_and_order(pcb):
    root, manifest = pcb
    mapping = {"OK": ["OK"], "defective": ["HS"]}
    info = read_manifest(str(root), str(manifest), mapping)
    assert len(info["records"]) == 54
    assert info["records"][0]["target"] == 0
    assert info["records"][0]["extra"] == "retained"
    assert info["records"][0]["mask_path"] is None
    assert info["meta"]["note"] == "retained"
    source = json.loads(manifest.read_text())
    source["groups"].reverse()
    for group in source["groups"]:
        group["images"].reverse()
    manifest.write_text(json.dumps(source))
    assert read_manifest(str(root), str(manifest), mapping) == info


@pytest.mark.parametrize(
    "problem,match",
    [
        ("label", "Unknown anomaly_class"),
        ("duplicate", "Duplicate image"),
        ("empty", "Empty"),
        ("missing", "Missing dataset"),
        ("mask", "Missing dataset"),
        ("group", "duplicate board_group"),
        ("field", "capture_group"),
    ],
)
def test_invalid_manifest(pcb, problem, match):
    root, manifest = pcb
    source = json.loads(manifest.read_text())
    record = source["groups"][0]["images"][0]
    if problem == "label":
        record["anomaly_class"] = "UNKNOWN"
    elif problem == "duplicate":
        source["groups"][0]["images"].append(record)
    elif problem == "empty":
        source["groups"][0]["images"] = []
    elif problem == "missing":
        record["image_path"] = "absent.jpg"
    elif problem == "mask":
        record["mask_path"] = "absent.png"
    elif problem == "group":
        source["groups"].append(source["groups"][0])
    else:
        del record["capture_group"]
    manifest.write_text(json.dumps(source))
    with pytest.raises(ValueError, match=match):
        read_manifest(str(root), str(manifest), {"OK": ["OK"], "defective": ["HS"]})


def test_mapping_rejects_overlap(pcb):
    with pytest.raises(ValueError, match="multiple"):
        read_manifest(str(pcb[0]), str(pcb[1]), {"OK": ["OK"], "defective": ["OK", "HS"]})


def test_split_isolation_saved_state_and_collation(pcb):
    cfg = pcb_config(pcb)
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    assignment = dm.state_dict()
    locations = defaultdict(set)
    for name in ("train", "val", "test"):
        dataset = getattr(dm, f"data_{name}")
        for record in dataset.records:
            locations[record["board_group"]].add(name)
        batch = next(iter(getattr(dm, f"{name}_dataloader")()))
        assert batch["image"].shape[1:] == (3, 16, 16)
        assert batch["target"].dtype == torch.int64
        assert batch["metadata"][0]["mask_path"] is None
        if name != "train":
            paths = [
                r["image_path"] for b in getattr(dm, f"{name}_dataloader")() for r in b["metadata"]
            ]
            assert len(paths) == len(set(paths)) == len(dataset)
    assert len(assignment["assignments"]) == 54
    assert all(len(names) == 1 for names in locations.values())
    other = hydra.utils.instantiate(cfg.data)
    other.setup()
    assert other.state_dict() == assignment
    saved = pcb[0] / "splits.json"
    saved.write_text(json.dumps(assignment))
    cfg.data.split.assignments_path = str(saved)
    other = hydra.utils.instantiate(cfg.data)
    other.setup()
    assert other.state_dict() == assignment
    bad = copy.deepcopy(assignment)
    bad["fingerprint"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        other.load_state_dict(bad)
    # Saved assignments are checked for S and capture isolation too.
    bad = copy.deepcopy(assignment)
    path = next(iter(bad["assignments"]))
    bad["assignments"][path] = "test" if bad["assignments"][path] != "test" else "train"
    with pytest.raises(ValueError, match="leaks"):
        other.load_state_dict(bad)


def test_group_sampler_coverage_and_balance():
    records = []
    for group, labels in enumerate(([0] * 3, [1] * 3, [0] * 2, [1] * 2, [0, 1, 0, 1])):
        records.extend({"board_group": str(group), "target": label} for label in labels)
    sampler = GroupBatchSampler(records, 6, 0.5, 1, 42, "error")
    batches = list(sampler)
    assert len(sampler) == len(batches)
    assert Counter(i for batch in batches for i in batch) == Counter(range(len(records)))
    for gid in sampler.groups:
        assert sum(any(records[i]["board_group"] == gid for i in b) for b in batches) == 1
    assert any(len({records[i]["board_group"] for i in b}) > 1 for b in batches)
    assert any(len(b) < 6 for b in batches)
    assert batches == list(sampler)
    sampler.set_epoch(2)
    expected = list(sampler)
    sampler.set_epoch(2)
    assert list(sampler) == expected
    with pytest.raises(ValueError, match="exceeds"):
        GroupBatchSampler(records, 2, 0.5, 1, 42, "error")
    singletons = GroupBatchSampler(records, 2, 0.5, 1, 42, "singleton")
    assert sorted(i for b in singletons for i in b) == list(range(len(records)))
    impossible = GroupBatchSampler(
        [{"board_group": "one", "target": 0}] * 3, 8, 1.0, 1, 0, "error"
    )
    assert list(impossible) == [[0, 1, 2]]


@pytest.mark.parametrize(
    "override,match",
    [
        ({"preprocessing.channels": 2}, "channels"),
        ({"preprocessing.std": [0, 1, 1]}, "Normalization"),
        ({"split.proportions": [0.5, 0.5, 0.5]}, "proportions"),
        ({"sampling.defective_class": "unknown"}, "defective_class"),
    ],
)
def test_invalid_configuration(pcb, override, match):
    cfg = pcb_config(pcb)
    for key, value in override.items():
        OmegaConf.update(cfg.data, key, value)
    with pytest.raises(Exception, match=match):
        hydra.utils.instantiate(cfg.data)


def test_loss_weighting_and_mapping_guard():
    model = ClassificationLitModule(
        torch.nn.Linear(2, 2),
        partial(torch.optim.Adam),
        None,
        {"OK": ["OK"], "defective": ["HS"]},
        "defective",
    )
    model.train_loss(torch.tensor(1.0), weight=2)
    model.train_loss(torch.tensor(4.0), weight=1)
    assert model.train_loss.compute().item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="mapping"):
        model.on_load_checkpoint({"class_mapping": {"other": ["x"]}})


@pytest.mark.slow
def test_pcb_train_resume_eval_export(pcb):
    cfg = pcb_config(pcb)
    HydraConfig().set_config(cfg)
    metrics, objects = train(cfg)
    dm = objects["datamodule"]
    trainer = objects["trainer"]
    best = trainer.checkpoint_callback.best_model_path
    assert best and (pcb[0] / "train/checkpoints/last.ckpt").is_file()
    results = pcb[0] / "train/results"
    predictions = json.loads((results / "test_predictions.json").read_text())
    assert len(predictions) == len(dm.data_test)
    assert all(r["extra"] == "retained" and len(r["probabilities"]) == 2 for r in predictions)
    assert "test/defective_recall" in metrics
    sampling = json.loads((results / "sampling_epoch_000.json").read_text())
    assert sampling["omitted_images"] == sampling["repeated_images"] == 0
    metadata = extract_model_metadata(dm, objects["model"], cfg.data, cfg.model)
    assert metadata["input_shape"] == [3, 16, 16]
    assert metadata["labels"] == ["OK", "defective"]
    network = objects["model"].net.cpu().eval()
    x = torch.zeros(2, *metadata["input_shape"])
    exported = torch.jit.trace(network, x)
    portable = pcb[0] / "portable.pt"
    torch.jit.save(exported, str(portable))
    torch.testing.assert_close(torch.jit.load(str(portable))(x), network(x))
    evaluation = pcb_config(pcb, "eval")
    evaluation.ckpt_path = best
    evaluation.data.split.assignments_path = str(results / "splits.json")
    HydraConfig().set_config(evaluation)
    evaluated, _ = evaluate(evaluation)
    assert evaluated["test/loss"].item() == pytest.approx(metrics["test/loss"].item())
    evaluation.evaluation_stage = "validate"
    validation, _ = evaluate(evaluation)
    assert "val/loss" in validation
    cfg.ckpt_path = str(pcb[0] / "train/checkpoints/last.ckpt")
    cfg.trainer.max_epochs = 2
    HydraConfig().set_config(cfg)
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step > trainer.global_step


def test_pcb_rejects_distributed(pcb):
    dm = hydra.utils.instantiate(pcb_config(pcb).data)
    from types import SimpleNamespace

    dm.trainer = SimpleNamespace(world_size=2)
    with pytest.raises(ValueError, match="single device"):
        dm.setup()


def test_binary_metrics_use_final_counts(monkeypatch):
    model = ClassificationLitModule(
        torch.nn.Linear(2, 2),
        partial(torch.optim.Adam),
        None,
        {"OK": ["OK"], "defective": ["HS"]},
        "defective",
    )
    values = {}
    monkeypatch.setattr(model, "log", lambda key, value: values.update({key: value.item()}))
    model.test_confusion(torch.tensor([1, 1, 0, 1, 0]), torch.tensor([1, 0, 1, 1, 0]))
    model._log_defective_metrics("test")
    assert values["test/defective_precision"] == pytest.approx(2 / 3)
    assert values["test/defective_recall"] == pytest.approx(2 / 3)
    assert values["test/defective_f1"] == pytest.approx(2 / 3)


def test_grayscale_and_checkpoint_mapping_order(pcb):
    cfg = pcb_config(pcb)
    cfg.data.preprocessing.channels = 1
    cfg.data.preprocessing.mean = [0.5]
    cfg.data.preprocessing.std = [0.5]
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    model = hydra.utils.instantiate(cfg.model)
    batch = next(iter(dm.train_dataloader()))
    assert model(batch["image"]).shape == (len(batch["target"]), 2)
    reversed_mapping = dict(reversed(list(dm.class_mapping.items())))
    with pytest.raises(ValueError, match="mapping"):
        model.on_load_checkpoint({"class_mapping": reversed_mapping})


def test_pcb_validation_config(pcb):
    cfg = pcb_config(pcb, "validate")
    assert cfg.evaluation_stage == "validate"
    assert cfg.data.name == "PCB"
    assert cfg.registry.model_name == "PCB"


def test_publication_uses_pcb_shape_and_registry(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.utils import model_registry as registry

    class Logger:
        def __init__(self):
            self.experiment = Mock(tracking_uri="http://example.invalid")
            self.run_id = "run"

    logger = Logger()
    monkeypatch.setattr(registry, "MLFlowLogger", Logger)
    monkeypatch.setattr(registry.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(registry.mlflow, "set_registry_uri", Mock())
    monkeypatch.setattr(registry.mlflow, "start_run", lambda **kwargs: nullcontext())
    log_model = Mock(return_value=SimpleNamespace(registered_model_version="2"))
    monkeypatch.setattr(registry.mlflow.pytorch, "log_model", log_model)
    promote = Mock()
    monkeypatch.setattr(registry, "promote_if_better", promote)
    trainer = SimpleNamespace(
        is_global_zero=True, loggers=[logger], callback_metrics={"test/loss": torch.tensor(0.25)}
    )
    model = SimpleNamespace(
        net=torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        )
    )
    cfg = OmegaConf.create(
        dict(enabled=True, metric="test/loss", model_name="PCB", alias="best_model")
    )
    assert (
        registry.publish_evaluated_model(
            trainer, model, cfg, "best.ckpt", {"input_shape": [3, 8, 8]}
        )
        == "2"
    )
    exported = log_model.call_args.args[0]
    assert exported(torch.zeros(2, 3, 8, 8)).shape == (2, 2)
    assert log_model.call_args.kwargs["registered_model_name"] == "PCB"
    promote.assert_called_once_with(logger.experiment, "PCB", "best_model", "2", "test/loss", 0.25)


@pytest.mark.parametrize("content", ["{broken", "[]", '{"groups": [null]}'])
def test_malformed_json_manifest(pcb, content):
    pcb[1].write_text(content)
    with pytest.raises(ValueError, match="manifest|group"):
        read_manifest(str(pcb[0]), str(pcb[1]), {"OK": ["OK"], "defective": ["HS"]})


def test_explicit_class_count_conflict(pcb):
    cfg = pcb_config(pcb)
    cfg.model.net.n_classes = 7
    with pytest.raises(Exception, match="class count conflicts"):
        hydra.utils.instantiate(cfg.model)

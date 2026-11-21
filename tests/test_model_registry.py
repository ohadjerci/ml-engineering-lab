"""Selection must never replace the alias with a worse or unmeasured model."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from mlflow.exceptions import MlflowException
from omegaconf import OmegaConf

from src.utils.model_registry import extract_model_metadata, promote_if_better


@pytest.fixture
def tracking(tmp_path):
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    experiment = client.create_experiment(
        "fixture", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return client, experiment


def completed_fixture_run(client, experiment, directory, fingerprint="a" * 64, evaluation="eval"):
    from src.data.components import file_hash

    checkpoint = directory / "model.ckpt"
    checkpoint.write_bytes(b"model artifact")
    run_id = client.create_run(experiment).info.run_id
    client.log_artifact(run_id, str(checkpoint), "checkpoints")
    receipt = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "evaluation": evaluation,
        "checkpoint": "checkpoints/model.ckpt",
        "model_uri": f"runs:/{run_id}/checkpoints/model.ckpt",
        "metrics": {"train/loss": 0.5, "test/loss": 0.6},
        "artifacts": [{"path": "checkpoints/model.ckpt", "sha256": file_hash(checkpoint)}],
    }
    client.log_dict(run_id, receipt, "provenance/completion.json")
    client.set_tag(run_id, "training.fingerprint", fingerprint)
    client.set_tag(run_id, "reproducibility.complete", "true")
    client.set_terminated(run_id, "FINISHED")
    return run_id, receipt


@pytest.mark.parametrize("failure", [None, "missing", "corrupt", "failed", "incomplete"])
def test_reuse_requires_successful_intact_model_artifacts(tracking, tmp_path, failure):
    from src.utils.logging_utils import find_reusable

    client, experiment = tracking
    run_id, _ = completed_fixture_run(client, experiment, tmp_path)
    artifact = tmp_path / "artifacts" / run_id / "artifacts/checkpoints/model.ckpt"
    if failure == "missing":
        artifact.unlink()
    elif failure == "corrupt":
        artifact.write_bytes(b"corrupt")
    elif failure == "failed":
        client.set_terminated(run_id, "FAILED")
    elif failure == "incomplete":
        client.set_tag(run_id, "reproducibility.complete", "false")
    result = find_reusable(client, experiment, "a" * 64)
    assert (result is not None) is (failure is None)


def test_fingerprint_settings_initialization_and_metadata_commits(cfg_train, tmp_path):
    from hydra.core.hydra_config import HydraConfig

    from src.utils.logging_utils import experiment_identity, initialization_identity

    HydraConfig().set_config(cfg_train)

    source = {"revision": "old", "files": {"src/train.py": "code", "uv.lock": "dependencies"}}
    dataset = {"identity": "data", "dataset_name": "fixture", "metadata_revision": "old"}
    model = torch.nn.Linear(2, 2)
    initialized = initialization_identity(model, None)
    baseline = experiment_identity(cfg_train, dataset, source, initialized, {})
    source["revision"] = dataset["metadata_revision"] = "metadata-only-new-revision"
    cfg_train.paths.output_dir = "another-output"
    cfg_train.versioning.remote.url = "s3://another-location"
    assert experiment_identity(cfg_train, dataset, source, initialized, {}) == baseline
    cfg_train.model.optimizer.lr *= 2
    assert experiment_identity(cfg_train, dataset, source, initialized, {})[0] != baseline[0]
    cfg_train.model.optimizer.lr /= 2
    with torch.no_grad():
        model.weight.add_(1)
    assert (
        experiment_identity(cfg_train, dataset, source, initialization_identity(model, None), {})[
            0
        ]
        != baseline[0]
    )
    checkpoint = tmp_path / "mutable.ckpt"
    checkpoint.write_bytes(b"weights one")
    first = initialization_identity(model, checkpoint)
    checkpoint.write_bytes(b"weights two")
    assert initialization_identity(model, checkpoint) != first


def test_equivalent_run_skips_training(cfg_train, tracking, tmp_path, monkeypatch):
    import importlib

    from hydra.core.hydra_config import HydraConfig

    from src.data.prepare_dataset import DatasetVersion
    from src.utils.logging_utils import experiment_identity, initialization_identity
    from tests.test_datamodules import FixtureDataModule

    training = importlib.import_module("src.train")
    HydraConfig().set_config(cfg_train)
    client, experiment = tracking
    cfg_train.versioning.enabled = True
    cfg_train.versioning.coordination_dir = str(tmp_path / "locks")
    cfg_train.data = {
        "_target_": "tests.test_datamodules.FixtureDataModule",
        "root_dir": str(tmp_path),
        "name": "fixture",
    }
    cfg_train.model = {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2}
    cfg_train.logger = {
        "mlflow": {
            "_target_": "lightning.pytorch.loggers.mlflow.MLFlowLogger",
            "tracking_uri": client.tracking_uri,
            "experiment_name": "fixture",
        }
    }
    source = {"files": {"src/train.py": "same", "uv.lock": "same"}, "revision": "one"}
    version = DatasetVersion(tmp_path, {}, {"identity": "same-data", "dataset_name": "fixture"})
    training.L.seed_everything(cfg_train.seed, workers=True)
    fingerprint, evaluation, _ = experiment_identity(
        cfg_train,
        version.provenance,
        source,
        initialization_identity(torch.nn.Linear(2, 2), None),
        {},
    )
    original, _ = completed_fixture_run(client, experiment, tmp_path, fingerprint, evaluation)
    monkeypatch.setattr(training, "source_provenance", lambda *a: source)
    monkeypatch.setattr(training, "prepare_dataset", lambda *a: version)
    monkeypatch.setattr(training, "runtime_information", lambda: {})
    fitting = Mock()
    monkeypatch.setattr(training, "_train", fitting)
    metrics, result = training.train(cfg_train)
    assert result["reused"] and result["run_id"] == original
    assert metrics["train/loss"] == 0.5
    fitting.assert_not_called()
    assert len(client.search_runs([experiment])) == 1
    # Keep a direct assertion that the shared fixture supplies the snapshot interface.
    assert callable(FixtureDataModule.export_snapshot)


def test_secret_redaction():
    from src.utils.logging_utils import sanitized

    value = sanitized(
        {
            "password": "hidden",
            "nested": {"api_token": "hidden"},
            "uri": "https://user:password@host/path?token=hidden",
        }
    )
    assert "hidden" not in str(value)
    assert value["uri"] == "https://host/path"


def test_mlflow_dataset_inputs_keep_full_content_identity(cfg_train, tracking, tmp_path):
    from hydra.core.hydra_config import HydraConfig
    from lightning.pytorch.loggers import MLFlowLogger

    from src.data.prepare_dataset import DatasetVersion
    from src.utils.logging_utils import log_provenance

    HydraConfig().set_config(cfg_train)
    client, _ = tracking
    identity = "b" * 64
    provenance = {
        "schema_version": 1,
        "dataset_name": "another-dataset",
        "identity": identity,
        "metadata_revision": "c" * 40,
        "pointers": [],
        "splits": {name: {"identity": name, "samples": 1} for name in ("train", "val", "test")},
    }
    source = {"repository": "", "revision": "d" * 40, "identity": "source", "files": {}}
    logger = MLFlowLogger(tracking_uri=client.tracking_uri, experiment_name="fixture")
    log_provenance(
        logger,
        cfg_train,
        DatasetVersion(tmp_path, {}, provenance),
        source,
        {},
        "e" * 64,
        "evaluation",
        {},
        None,
    )
    run = client.get_run(logger.run_id)
    assert run.data.tags["dataset.identity"] == identity
    assert len(run.inputs.dataset_inputs) == 3
    for item in run.inputs.dataset_inputs:
        assert item.dataset.name.startswith("another-dataset/")
        assert identity in item.dataset.source
        assert item.dataset.digest == identity[:32]


def test_extract_metadata_prefers_runtime_values():
    class Normalize:
        mean = (0.25,)
        std = (0.75,)

    datamodule = SimpleNamespace(
        name="RuntimeDataset",
        transforms_test=SimpleNamespace(transforms=[Normalize()]),
        data_test=[(torch.zeros(1, 12, 16), 0)],
        classes=["zero", "one"],
        num_classes=2,
    )
    model = SimpleNamespace(net=torch.nn.Linear(1, 1))
    data_cfg = OmegaConf.create(
        {"name": "ConfiguredDataset", "mean": [0.0], "input_shape": [3, 28, 28]}
    )
    model_cfg = OmegaConf.create({"_target_": "configured.Model", "net": {"width": 8}})

    metadata = extract_model_metadata(datamodule, model, data_cfg, model_cfg)

    assert metadata["dataset"] == "RuntimeDataset"
    assert metadata["preprocessing"] == ["Normalize"]
    assert metadata["mean"] == [0.25]
    assert metadata["std"] == [0.75]
    assert metadata["input_shape"] == [1, 12, 16]
    assert metadata["image_size"] == [12, 16]
    assert metadata["labels"] == ["zero", "one"]
    assert metadata["num_classes"] == 2
    assert metadata["model_architecture"] == "Linear"
    assert metadata["model_config"] == {"width": 8}


def test_extract_metadata_uses_config_fallback_and_omits_unknown_values():
    metadata = extract_model_metadata(
        SimpleNamespace(),
        SimpleNamespace(),
        OmegaConf.create(
            {
                "name": "ConfiguredDataset",
                "input_shape": [3, 20, 30],
                "image_size": [20, 30],
                "labels": ["cat", "dog"],
                "num_classes": 2,
                "preprocessing": ["Resize", "ToTensor"],
                "mean": [0.5],
                "std": [0.25],
            }
        ),
        OmegaConf.create({"_target_": "models.Example", "net": {"width": 16}}),
    )

    assert metadata["dataset"] == "ConfiguredDataset"
    assert metadata["input_shape"] == [3, 20, 30]
    assert metadata["image_size"] == [20, 30]
    assert metadata["labels"] == ["cat", "dog"]
    assert metadata["num_classes"] == 2
    assert metadata["preprocessing"] == ["Resize", "ToTensor"]
    assert metadata["mean"] == [0.5]
    assert metadata["std"] == [0.25]
    assert metadata["model_architecture"] == "models.Example"
    assert "mnist" not in str(metadata).lower()
    assert "image_size" not in extract_model_metadata(
        SimpleNamespace(),
        SimpleNamespace(),
        OmegaConf.create({}),
        OmegaConf.create({}),
    )


@pytest.mark.parametrize("candidate,expected", [(0.9, True), (1.0, False), (1.1, False)])
def test_promote_only_strict_improvement(candidate, expected):
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"best_model": "1"})
    client.get_model_version.return_value = SimpleNamespace(
        tags={"selection_metric": "test/loss", "selection_score": "1.0"}
    )
    assert (
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", candidate) is expected
    )
    assert client.set_registered_model_alias.called is expected


def test_first_version_is_promoted():
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={})
    assert promote_if_better(client, "MNIST", "best_model", "1", "test/loss", 1.0)
    client.set_registered_model_alias.assert_called_once_with("MNIST", "best_model", "1")


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_score_is_rejected(score):
    client = Mock()
    with pytest.raises(ValueError, match="Non-finite"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", score)
    client.set_registered_model_alias.assert_not_called()


def test_incompatible_selection_metadata_is_rejected():
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"best_model": "1"})
    client.get_model_version.return_value = SimpleNamespace(tags={})
    with pytest.raises(ValueError, match="incompatible"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", 1.0)
    client.set_registered_model_alias.assert_not_called()


def test_connection_errors_do_not_look_like_missing_aliases():
    client = Mock()
    client.get_registered_model.side_effect = MlflowException("unavailable")
    with pytest.raises(MlflowException, match="unavailable"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", 1.0)
    client.set_registered_model_alias.assert_not_called()

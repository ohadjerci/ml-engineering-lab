import hydra
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


@pytest.mark.parametrize("dataset", ["mnist", "pcb"])
@pytest.mark.parametrize("remote", ["local", "ssh", "s3"])
def test_independent_dataset_config(dataset, remote):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="data/prepare_dataset",
            overrides=[f"data={dataset}", f"data/versioning@versioning={remote}"],
        )
    assert cfg.data._target_.startswith("src.data.")
    assert cfg.versioning.enabled and cfg.versioning.upload and cfg.versioning.git_commit
    assert cfg.versioning.git_push is False

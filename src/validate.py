"""Validate an explicit checkpoint without evaluating the test split or publishing it."""

import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils  # noqa: E402
from src.eval import evaluate  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="validate.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()

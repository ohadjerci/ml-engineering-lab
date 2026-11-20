from pathlib import Path
from typing import Sequence, cast

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for key in cfg:
        # Hydra's configuration groups use string keys.
        field = cast(str, key)
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    from src.utils.logging_utils import sanitized

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            content = OmegaConf.to_container(config_group, resolve=resolve)
            branch_content = OmegaConf.to_yaml(OmegaConf.create(sanitized(content)))
        else:
            branch_content = str(sanitized(config_group))

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config."""
    if not cfg.get("tags"):
        # Hydra exposes a structured DictConfig at runtime, typed as the schema dataclass.
        job_cfg = cast(DictConfig, HydraConfig.get().job)
        if "id" in job_cfg:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags_input = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags_input.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)

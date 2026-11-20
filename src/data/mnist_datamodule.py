from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.data.components import SPLITS, safe_path, verify_snapshot


class SnapshotMNIST(Dataset):
    def __init__(self, root, records, train_transform, test_transform, classes):
        self.root = root
        self.records = records
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.classes = classes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        transform = self.train_transform if record["origin"] == "train" else self.test_transform
        with Image.open(self.root / record["image_path"]) as image:
            return transform(image.convert("L")), record["target"]


class MNISTDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:     def prepare_data(self):         # things to do on 1
    GPU/TPU (not on every GPU/TPU in DDP)         # download data, pre-process, split, save to
    disk, etc...     def setup(self, stage):         # things to do on every process in DDP
    # load data, set variables, etc...     def train_dataloader(self):         # return train
    dataloader     def val_dataloader(self):         # return validation dataloader     def
    test_dataloader(self):         # return test dataloader     def teardown(self):         #
    called on every process in DDP         # clean up after fit or test

    This allows you to share a full dataset without explaining how to download, split, transform
    and process the data.

    Read the docs:

    https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        name: str = "default",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        snapshot_path: str | None = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.transforms_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        if self.hparams["snapshot_path"]:
            return
        MNIST(self.hparams["data_dir"], train=True, download=True)
        MNIST(self.hparams["data_dir"], train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if self.hparams["snapshot_path"] and self.data_train is None:
            root = Path(self.hparams["snapshot_path"])
            manifest = verify_snapshot(root)
            by_id = {record["id"]: record for record in manifest["records"]}
            for name in SPLITS:
                setattr(
                    self,
                    f"data_{name}",
                    SnapshotMNIST(
                        root,
                        [by_id[sid] for sid in manifest["splits"][name]],
                        self.transforms_train,
                        self.transforms_test,
                        [c["name"] for c in manifest["class_mapping"]],
                    ),
                )
            return
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.hparams["data_dir"], train=True, transform=self.transforms_train)
            testset = MNIST(self.hparams["data_dir"], train=False, transform=self.transforms_test)
            dataset: ConcatDataset[Any] = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams["train_val_test_split"],
                generator=torch.Generator().manual_seed(42),
            )

    def export_snapshot(self, destination: Path) -> dict[str, Any]:
        self.prepare_data()
        self.setup()
        if self.hparams["snapshot_path"]:
            raise ValueError("Export source data, not an already bound snapshot")
        subset = cast(Any, self.data_train)
        records = []
        for origin, dataset in zip(("train", "test"), subset.dataset.datasets):
            for index, (pixels, target) in enumerate(zip(dataset.data, dataset.targets)):
                relative = f"images/{origin}/{index:06d}.png"
                path = safe_path(destination, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels.numpy()).save(path)
                records.append(
                    {
                        "id": f"{origin}/{index}",
                        "image_path": relative,
                        "target": int(target),
                        "origin": origin,
                    }
                )
        return {
            "name": self.hparams["name"],
            "annotation_format": "image-classification-v1",
            "class_mapping": [
                {"name": name, "index": i}
                for i, name in enumerate(subset.dataset.datasets[0].classes)
            ],
            "records": records,
            "splits": {
                name: [records[i]["id"] for i in getattr(self, f"data_{name}").indices]
                for name in SPLITS
            },
        }

    def train_dataloader(self):
        return DataLoader(
            # Lightning calls setup before requesting dataloaders.
            dataset=cast(Dataset[Any], self.data_train),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=cast(Dataset[Any], self.data_val),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=cast(Dataset[Any], self.data_test),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = MNISTDataModule()

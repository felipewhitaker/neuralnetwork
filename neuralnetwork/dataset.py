from pathlib import Path
from typing import Callable, Optional

import numpy as np

import pytorch_lightning as pl

import torch
import xarray as xr
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class toNumpy(object):
    def __call__(self, data: xr.DataArray):
        return data.values


class Permute(object):
    def __call__(self, data: torch.Tensor):
        # return result to (time, lat, lon)
        return data.permute(1, 2, 0)


class Precipitation(Dataset):
    def __init__(
        self,
        filepath: Path,
        X_name: str,
        y_name: str,
        transforms: Optional[Callable] = None,
    ):
        self.X_name = X_name
        self.y_name = y_name
        self.transforms = (
            transforms if transforms is not None else self.default_transforms()
        )

        data = xr.open_dataset(filepath)
        X = data[self.X_name]
        self.number, self.steps, *_ = X.shape
        self.X = torch.stack(
            [self.transforms(X[m, ...]) for m in range(self.number)], dim=0
        ).float()
        self.y = self.transforms(data[self.y_name]).float()
        data.close()

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        member, step = index % self.number, index % self.steps
        return self.X[member, step, ...], self.y[step, ...]

    def __len__(self) -> int:
        return self.number * self.steps

    @staticmethod
    def default_transforms() -> torch.Tensor:
        return transforms.Compose([toNumpy(), transforms.ToTensor(), Permute()])

    def __repr__(self) -> str:
        return f"Data ({self.X.shape}, {self.y.shape}) returning `({self.X_name},{self.y_name})`"

    @property
    def grid(self) -> tuple[int, int]:
        return self.X.shape[2:]


class PrecipitationDataModule(pl.LightningDataModule):
    def __init__(
        self, train: Dataset, test: Dataset, batch_size: int, num_workers: int = 8
    ) -> None:
        super().__init__()
        self.train = train
        self.test = test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train, self.val = random_split(self.train, [0.8, 0.2])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    transform = transforms.Compose([toNumpy(), transforms.ToTensor()])

    data = xr.DataArray(
        np.random.uniform(0, 1, (10, 10, 10)), dims=["time", "lat", "lon"]
    )
    assert type(transform(data), torch.Tensor)

import numpy as np

import pytorch_lightning as pl

import torch
from torch import nn


class DenseNet(nn.Module):
    def __init__(self, grid: tuple[int, int], hidden_size: int) -> None:
        super().__init__()

        flatten_size = np.prod(grid)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, flatten_size),
            nn.Unflatten(1, grid),
        )

    def forward(self, X):
        return self.model(X)


class DenseNetModule(pl.LightningModule):
    def __init__(self, grid: tuple[int, int], hidden_size: int = 128) -> None:
        super().__init__()

        self.model = DenseNet(grid, hidden_size)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    grid = (10, 10)
    X = torch.rand(1, *grid)
    model = DenseNet(grid)
    y = model(X)
    assert y.shape == X.shape
    print("OK")

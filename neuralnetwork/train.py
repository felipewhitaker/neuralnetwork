from datetime import datetime

import pytorch_lightning as pl
import torch
from dataset import Precipitation, PrecipitationDataModule
from models import Debias, DenseNetModule, Linear
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn

from torchvision import transforms

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    import argparse

    models = {
        "Linear": Linear,
        "Debias": Debias,
        "DenseNetModule": DenseNetModule,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Linear", help=f"model name")
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="hidden size for `DenseNetModule`"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs for `DenseNetModule`"
    )
    args = parser.parse_args()

    # FIXME inplement args.model == "all"
    if args.model not in models:
        # FIXME implement args.model == "all"
        raise ValueError(f"model name must be one of {models.keys()}")
    # "tp" for total precipitation from seasonal forecast
    X_name, y_name = "tp", "prec"

    if args.model == "DenseNetModule":
        pl.seed_everything(args.seed)  # for reproducibility

        # FIXME files should be parameters
        TRAIN_FILE, TEST_FILE = (
            "./data/combined_train_scaled.nc",
            "./data/combined_test_scaled.nc",
        )

        train_dataset = Precipitation(TRAIN_FILE, X_name, y_name)
        test_dataset = Precipitation(TEST_FILE, X_name, y_name)

        # split data and set dataloader
        dataloader = PrecipitationDataModule(
            train_dataset, test_dataset, batch_size=32, num_workers=8
        )

        # train model
        model_checkpoint = ModelCheckpoint(
            dirpath="./checkpoints/", filename="DenseNetModule", monitor="val_loss"
        )
        model = DenseNetModule(train_dataset.grid, hidden_size=args.hidden_size)
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices="auto",
            strategy="auto",
            logger=CSVLogger(
                "./lightning_logs/", name=f"DenseNetModule_{datetime.now():%H%M}"
            ),
            callbacks=[model_checkpoint],
        )
        trainer.fit(
            model,
            dataloader,
        )

        print(model_checkpoint.best_model_path)
    else:
        import pickle
        import numpy as np
        import xarray as xr

        TRAIN_FILE, TEST_FILE = "./data/combined_train.nc", "./data/combined_test.nc"

        train = xr.open_dataset(TRAIN_FILE)
        # test = xr.open_dataset(TEST_FILE)

        # (members, time, lat, lon)
        model = models[args.model]()
        # FIXME average over members
        model.fit(train[X_name].mean(dim = ["number"]), train[y_name])

        print(model)

        with open(f"./checkpoints/{args.model}.pt", "wb") as f:
            pickle.dump(model, f)
        
        # TODO include one hot enconding of day into model

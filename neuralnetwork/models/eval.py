if __name__ == "__main__":
    # FIXME paths are hard coded

    import pickle

    import xarray as xr

    rse = lambda y, y_pred: ((y - y_pred) ** 2).mean() ** 0.5

    data_train = xr.open_dataset("./data/combined_train.nc")
    data_test = xr.open_dataset("./data/combined_test.nc")

    max_, min_ = data_train.max(dim="valid_time"), data_train.min(dim="valid_time")
    # FIXME should receive `max_`, `min_` and `y_col` as arguments
    rescale = (
        lambda x, m: x
        * (max_.sel(number=m).prec.values - min_.sel(number=m).prec.values)
        + min_.sel(number=m).prec.values
    )

    benchmark = (((data_test.tp - data_test.prec) ** 2).mean() ** 0.5).item()

    print(f"Benchmark: {benchmark:.4f} : {0:.2f}%")
    for name in ["Debias", "Linear"]:
        with open(f"./checkpoints/{name}.pt", "rb") as f:
            model = pickle.load(f)
            error = model.predict(data_test.tp).mean(dim="number") - data_test.prec
            rse = ((error**2).mean() ** 0.5).item()
            print(
                "{:9}: {:.4f} : {:.2f}%".format(
                    name, rse, (benchmark - rse) / benchmark * 100
                )
            )
    import numpy as np
    import torch

    from neuralnetwork.dataset import Precipitation
    from neuralnetwork.models import DenseNetModule

    test_data = Precipitation("./data/combined_test.nc", "tp", "prec")

    model = DenseNetModule.load_from_checkpoint(
        "./checkpoints/DenseNetModule-v3.ckpt", grid=(8, 9)
    )  # Dense (128) 2.92

    model.cpu()

    with torch.no_grad():
        preds = [
            rescale(model(test_data.X[m, ...]).numpy(), m)
            for m in range(test_data.number)
        ]
    error = np.mean(np.stack(preds), axis=0) - test_data.y.numpy()
    rse = (error**2).mean() ** 0.5
    print(
        "{:9}: {:.4f} : {:.2f}%".format(
            "Dense (128)", rse, (benchmark - rse) / benchmark * 100
        )
    )

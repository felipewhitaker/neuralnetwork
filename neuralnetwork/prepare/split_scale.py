if __name__ == "__main__":

    import xarray as xr
    import numpy as np

    np.random.seed(42) # reproducibility

    FILEPATH = "./data/combined.nc"
    data = xr.open_dataset(FILEPATH)

    # random split train test of `combined.nc`
    time = data.sizes["valid_time"]
    train_time = np.random.rand(time) < .9
    test_time = ~train_time

    data_train = data.sel(valid_time = train_time)
    data_test = data.sel(valid_time = test_time)

    data_train.to_netcdf("./data/combined_train.nc")
    data_test.to_netcdf("./data/combined_test.nc")

    # MaxMin scaling
    max_, min_ = data_train.max(dim = "valid_time"), data_train.min(dim = "valid_time")
    data_train = (data_train - min_) / (max_ - min_)
    data_test = (data_test - min_) / (max_ - min_)

    data_train.to_netcdf("./data/combined_train_scaled.nc")
    data_test.to_netcdf("./data/combined_test_scaled.nc")

    data.close()



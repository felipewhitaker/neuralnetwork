
if __name__ == "__main__":
    import xarray as xr

    import argparse

    parser = argparse.ArgumentParser(description="Download necessary data")
    parser.add_argument("observation", type=str, help="Path to observation data")
    parser.add_argument("forecast", type=str, help="Path to forecast data")
    parser.add_argument("out", type=str, help="Path to combined data")
    args = parser.parse_args()

    # FIXME paths shouldn't be hard coded
    merge_path = args.observation # "./data/fmerge.nc"
    seasonal_path = args.forecast # "./data/adaptor.mars.external-1688510425.910106-10412-19-09a67beb-95d7-428c-a00f-6182524f2b1b.grib"

    # use `valid_time` as indexer and remove 2022-01-01
    seasonal = xr.open_dataset(seasonal_path, engine="cfgrib", backend_kwargs=dict(time_dims=('valid_time',))).isel(valid_time = slice(None, -1))

    interp_lat, interp_lon = seasonal.latitude, seasonal.longitude
    merge = xr.open_dataset(merge_path) / 100 # FIXME shouldn't be 1_000 to transform `mm` into `m` ?
    interp_merge = merge.interp(latitude=interp_lat, longitude=interp_lon, time = seasonal.valid_time)

    ds = xr.merge([interp_merge.drop("prmsl"), seasonal])
    ds["day"] = ds.valid_time.dt.day

    ds.to_netcdf(args.out) # ("./data/combined.nc")
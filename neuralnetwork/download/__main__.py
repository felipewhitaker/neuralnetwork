from datetime import date, datetime
from functools import partial

import cdsapi
import xarray as xr
from cfgrib.xarray_to_grib import to_grib

from cptec import _preprocess, clean, filter_combine, MERGE, merge_save, SAMeT

def valid_date_type(arg_date_str):
    """custom argparse *date* type for user dates values given from the command line"""
    # from https://gist.github.com/monkut/e60eea811ef085a6540f
    try:
        return datetime.strptime(arg_date_str, "%Y-%m-%d")
    except ValueError:
        msg = "Given Date ({0}) not valid! Expected format, YYYY-MM-DD!".format(arg_date_str)
        raise argparse.ArgumentTypeError(msg)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download necessary data")

    for name in ["north", "west", "east", "south"]:
        parser.add_argument(
            name,
            type=int,
            help=f"Coordinate of {name}ernmost point of rectangle",
        )
    parser.add_argument(
        "--start-date", type=valid_date_type, help="Start date", default=date(2021, 1, 1)
    )
    parser.add_argument(
        "--end-date", type=valid_date_type, help="End date", default=date(2022, 1, 1)
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean data folder", default=False
    )
    # FIXME add `lat`, `lon` arguments to save only a specific region; and then set `clean` to `False`

    args = parser.parse_args()

    print(args)

    # merge_path = "./data/MERGE/"  # FIXED by `MERGE` class
    files = merge_save(MERGE(), args.start_date, args.end_date)

    north, west, east, south = args.north, args.west, args.east, args.south
    lon_bnds, lat_bnds = (south, north), (west, east)

    # FIXME should use `xr.open_mfdataset` with preprocess instead of
    # reading the entire file and then filtering
    
    # partial_preprocess = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    # filter_combine(files, "merge", preprocess=partial_preprocess)

    files = "./data/MERGE/*.grib2"
    ds = xr.open_mfdataset(
        files,
        engine="cfgrib",
        concat_dim="time",
        combine="nested",
    )

    # FIXME hard coded latitudes and longitudes
    filtered_ds = ds.assign_coords(longitude=ds.longitude - 360.0).sel(
        latitude=slice(south, north), longitude=slice(east, west)
    )
    # to_grib(filtered_ds, "data/fmerge.grib2")
    # filtered_ds.to_netcdf("./data/fmerge.nc")
    filtered_ds.to_netcdf("./data/fmerge_2022.nc")

    if args.clean:
        clean(files)

    # FIXME should consider the parameters received above
    # FIXME should be in a separate file
    # c = cdsapi.Client()
    # c.retrieve(
    #     "seasonal-original-single-levels",
    #     {
    #         "originating_centre": "ecmwf",
    #         "system": "5",
    #         "variable": "total_precipitation",
    #         "year": "2021",
    #         "month": [
    #             "01",
    #             "02",
    #             "03",
    #             "04",
    #             "05",
    #             "06",
    #             "07",
    #             "08",
    #             "09",
    #             "10",
    #             "11",
    #             "12",
    #         ],
    #         "day": "01",
    #         "leadtime_hour": [
    #             "24",
    #             "48",
    #             "72",
    #             "96",
    #             "120",
    #             "144",
    #             "168",
    #             "192",
    #             "216",
    #             "240",
    #             "264",
    #             "288",
    #             "312",
    #             "336",
    #             "360",
    #             "384",
    #             "408",
    #             "432",
    #             "456",
    #             "480",
    #             "504",
    #             "528",
    #             "552",
    #             "576",
    #             "600",
    #             "624",
    #             "648",
    #             "672",
    #             "696",
    #             "720",
    #             "744",
    #         ],
    #         "area": [
    #             -16,
    #             -54,
    #             -23,
    #             -46,
    #         ],
    #         "format": "grib",
    #     },
    #     "./data/seas5.grib",
    # )

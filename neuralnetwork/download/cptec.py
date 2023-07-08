import logging
import os
from abc import ABC, abstractproperty
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Callable

from time import sleep
from random import random
from functools import wraps

import requests
import xarray as xr
from cfgrib.xarray_to_grib import to_grib

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="{asctime} [{levelname}] {name} {message}",
    style="{",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
    ],
)

def uniform(a: int = 0, b: int = 1):
    def wrapper():
        return a + (b - a) * random()
    return wrapper

def wait(wait_gen: Callable = uniform(0, 3)):
    def outer(func):
        def inner(*args, **kwargs):
            sleep(wait_gen())
            return func(*args, **kwargs)
        return inner
    return outer

class CPTEC(ABC):
    BASE_URL = "http://ftp.cptec.inpe.br/modelos/tempo"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        return

    @abstractproperty
    def model_name(self):
        raise NotImplementedError

    def generate_url(self, dt: date, **kwargs):
        # **kwargs exist for SAMeT compatibility
        raise NotImplementedError

    @wait()
    def download_content(self, data_url: str, path: str):
        if os.path.isfile(path):
            self.logger.info(f"`{path}` already exists!")
            return path
        self.logger.info(f"Downloading `{data_url}` into `{path}`")
        req = requests.get(data_url, allow_redirects=True)
        req.raise_for_status()
        with open(path, "wb+") as f:
            f.write(req.content)
        return path

    def generate_path(self, data_url: str):
        file_name = data_url.rsplit("/", 1)[-1]
        # FIXME relative path might not exist
        return f"./data/{self.model_name.lower()}/{file_name}"


class SAMeT(CPTEC):
    # FIXME `TEMPS` should be Enum
    TEMPS = ["TMAX", "TMED", "TMIN"]

    @property
    def model_name(self):
        return "SAMeT"

    def validate_temp(self, temp: str):
        # FIXME could be validated via decorator that checks for kws
        if temp in self.TEMPS:
            return temp
        raise ValueError(f"{temp} not recognized. Must be one of {self.TEMPS}")

    def generate_url(self, dt: date, temp: str):
        # http://ftp.cptec.inpe.br/modelos/tempo/SAMeT/DAILY/TMED/2022/01/SAMeT_CPTEC_TMED_20220101.nc
        name = self.model_name
        year, month, pdate = dt.year, dt.month, dt.strftime("%Y%m%d")
        temp = self.validate_temp(temp)
        # FIXME add other temporal resolutions beside `DAILY`
        data_url = f"{self.BASE_URL}/{name}/DAILY/{temp}/{year}/{month:02}/{name}_CPTEC_{temp}_{pdate}.nc"
        return data_url


class MERGE(CPTEC):
    @property
    def model_name(self):
        return "MERGE"

    def generate_url(self, dt: date):
        # http://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY/2022/01/MERGE_CPTEC_20220101.grib2
        name = self.model_name
        year, month, pdate = dt.year, dt.month, dt.strftime("%Y%m%d")
        # FIXME currently only GPM is available
        # FIXME add other temporal resolutions beside `DAILY`
        data_url = f"{self.BASE_URL}/{name}/GPM/DAILY/{year}/{month:02}/{name}_CPTEC_{pdate}.grib2"
        return data_url


def merge_save(cptec: CPTEC, start_date: date, end_date: date):
    dt = start_date
    saved_files = []
    while dt < end_date:
        cptec_url = cptec.generate_url(dt)
        file_path = cptec.download_content(cptec_url, cptec.generate_path(cptec_url))
        saved_files.append(file_path)
        dt += timedelta(days=1)
    return saved_files


def filter_combine(files: Iterable[Path], filename: str):#, preprocess: Callable):
    # FIXME coords are dataset dependent, and should be passed as kwargs
    # FIXME MERGE longitude is 240:340 instead of -60:60
    ds = xr.open_mfdataset(
        files,
        concat_dim="time",
        combine="nested",
        # preprocess=preprocess,
    )
    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    to_grib(ds, f"./data/{filename}.grib2")


def clean(files: Iterable[Path]):
    for i, file in enumerate(files):
        os.remove(file)
        print(f"Removed {file}, {i + 1}/{len(files)}", end = "\r")
    return


def _preprocess(x, lon_bnds, lat_bnds):
    # FIXME `longitude` and `latitude` are dataset dependent
    return x.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))

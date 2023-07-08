from typing import Any

import numpy as np
import xarray as xr

# FIXME this classes should be used only over `xr.Dataset` sharing
# the same `time` dim, and `lat` and `lon` dims should be the same


class Constant:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        # FIXME when time steps are given, the constant should be
        # the last one (persistence model)
        return X


class Debias:
    def __init__(self) -> None:
        self.dim = "valid_time"  # FIXME should be a parameter

    def fit(self, X, y):
        self.bias = y.mean(dim=self.dim) - X.mean(dim=self.dim)
        return self

    def predict(self, X):
        return X + self.bias
    
    def __repr__(self) -> str:
        return f"Debias({self.bias.mean()})"


class Linear:
    def __init__(self, grid: bool = True) -> None:
        # when `grid == False`, estimates two parameters (linear trend and bias),
        # while `grid == True` estimates both parameters for each grid entry
        self.grid = grid

    def _estimate_grid(self, X, y):
        # FIXME estimates linear regression `(self.degree = 1)`
        # over `time` dim for both `X` and `y`, which should both
        # share the same dim names (e.g. `lat` `lon` `time`)

        def linear_trend(X, y):
            return np.polyfit(X, y, 1)

        if self.grid:
            slopes = xr.apply_ufunc(
                linear_trend,
                X,
                y,
                vectorize=True,
                input_core_dims=[["valid_time"], ["valid_time"]],
                output_core_dims=[["degree"]],
                dask="parallelized",
                dask_gufunc_kwargs={
                    "allow_rechunk": True,
                    "output_sizes": {"degree": 2},  # self.degree + 1
                },
            )
            return slopes.compute().transpose("degree", "latitude", "longitude")
        return linear_trend(X.stack(dim=[...]), y.stack(dim=[...]))

    def _model(self, X, slopes):
        # FIXME applies the linear regression model over `time` dim
        # for `X` and `slopes`, which should both share the same dim
        # names (e.g. `lat` `lon` `time`)

        # FIXME should use `xr.apply_ufunc` with `np.polyval` instead
        b1, b0 = slopes
        return X * b1 + b0

    def fit(self, X, y):
        self.slopes = self._estimate_grid(X, y)
        return self

    def predict(self, X):
        return self._model(X, self.slopes)
    
    def __repr__(self) -> str:
        if self.grid:
            slopes = self.slopes.mean(dim=['latitude', 'longitude'])
        else:
            slopes = self.slopes
        return f"Linear({slopes})"
        


if __name__ == "__main__":
    # FIXME create tests files

    grid = (2, 10, 10)
    time, lat, lon = [np.arange(g) for g in grid]

    X = xr.DataArray(
        np.random.rand(*grid),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )

    true_b1, true_b0 = 2, 1
    y = X * true_b1 + true_b0

    model = Constant()
    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.allclose(y_pred, X), f"{model.__class__.__name__} failed"

    model = Linear(grid=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    b1, b0 = model.slopes
    for i, (b, true_b) in enumerate(((b0, true_b0), (b1, true_b1))):
        assert np.allclose(
            b, true_b
        ), f"{model.__class__.__name__} failed (b{i}): {b.mean().item()} != {true_b}"
    # FIXME consider a gaussian error ` + np.random.normal(0, error_std, grid)`
    true_b1, true_b0 = (
        np.random.uniform(0, 1, grid[1:]),
        np.random.uniform(0, 1, grid[1:]) + 1,
    )
    y = X * true_b1 + true_b0

    model = Debias()
    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.allclose(y_pred, X + model.bias), f"{model.__class__.__name__} failed"
    # assert np.allclose(
    #     model.bias, true_b0 - X.mean(dim="time")
    # ), f"{model.__class__.__name__} failed"

    model = Linear()
    model.fit(X, y)
    y_pred = model.predict(X)

    b1, b0 = model.slopes
    for i, (b, true_b) in enumerate(((b0, true_b0), (b1, true_b1))):
        assert np.allclose(
            b, true_b
        ), f"{model.__class__.__name__} failed (b{i}): {b.mean().item()} != {true_b.mean()}"

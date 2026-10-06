"""Forecast models. Each model gives quantile forecasts for feature rows.

* ``seasonal_naive``: the same weekday one week earlier (or more, for long horizons).
* ``weekday_mean``: the mean of the last four values of the same weekday.
* ``ridge``: a scikit-learn pipeline (drop constant, impute, scale, ridge regression).
* ``gbm``: one gradient-boosting model with quantile loss per quantile.
* ``lightgbm``: the same with LightGBM (optional extra, imported only when you use it).

All preprocessing is inside a pipeline. The pipeline is fit on the training rows only.
A new model object is made for each backtest fold and for the final forecast.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import FEATURES


class DropConstant(BaseEstimator, TransformerMixin):
    """Drop columns that have fewer than two distinct non-missing values in the training rows."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = np.array(
            [len(np.unique(col[~np.isnan(col)])) > 1 for col in X.T], dtype=bool
        )
        if not self.keep_.any():
            self.keep_[0] = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def sort_quantiles(pred: np.ndarray) -> np.ndarray:
    """Remove quantile crossing and negative demand: sort each row and clip at zero."""
    return np.clip(np.sort(pred, axis=1), 0, None)


class Forecaster:
    name = "base"

    def __init__(self, quantiles: tuple[float, ...], seed: int = 42, threads: int = 1):
        self.quantiles = tuple(sorted(quantiles))
        self.seed = seed
        self.threads = threads

    def fit(self, rows: pd.DataFrame) -> "Forecaster":
        raise NotImplementedError

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Return an array with one column per quantile, in ascending quantile order."""
        raise NotImplementedError


class _ResidualQuantiles(Forecaster):
    """Point forecast plus empirical residual quantiles per horizon."""

    def _point(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _fit_residuals(self, rows: pd.DataFrame, point: np.ndarray) -> None:
        res = rows["y"].to_numpy() - point
        ok = ~np.isnan(res)
        self.res_q_: dict[int, np.ndarray] = {}
        all_q = np.quantile(res[ok], self.quantiles) if ok.any() else np.zeros(len(self.quantiles))
        for h in np.unique(rows["horizon"]):
            m = ok & (rows["horizon"].to_numpy() == h)
            self.res_q_[int(h)] = np.quantile(res[m], self.quantiles) if m.sum() >= 20 else all_q
        self.res_all_ = all_q

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        point = self._point(rows)
        offs = np.vstack([self.res_q_.get(int(h), self.res_all_) for h in rows["horizon"]])
        return sort_quantiles(point[:, None] + offs)


class SeasonalNaive(_ResidualQuantiles):
    name = "seasonal_naive"

    def _point(self, rows):
        return rows["same_weekday_last"].fillna(rows["last_obs"]).fillna(0).to_numpy(dtype=float)

    def fit(self, rows):
        self._fit_residuals(rows, self._point(rows))
        return self


class WeekdayMean(_ResidualQuantiles):
    name = "weekday_mean"

    def _point(self, rows):
        return rows["same_weekday_mean_4"].fillna(rows["mean_7"]).fillna(0).to_numpy(dtype=float)

    def fit(self, rows):
        self._fit_residuals(rows, self._point(rows))
        return self


class RidgeForecaster(_ResidualQuantiles):
    """Linear model in a pipeline. Residual quantiles come from a time-ordered holdout."""

    name = "ridge"

    def _pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("drop_constant", DropConstant()),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

    def fit(self, rows):
        rows = rows.sort_values("target_date")
        cut = int(len(rows) * 0.8)
        head, tail = rows.iloc[:cut], rows.iloc[cut:]
        calib = self._pipeline().fit(head[list(FEATURES)], head["y"])
        self._fit_residuals(tail, calib.predict(tail[list(FEATURES)]))
        self.pipeline_ = self._pipeline().fit(rows[list(FEATURES)], rows["y"])
        return self

    def _point(self, rows):
        return self.pipeline_.predict(rows[list(FEATURES)])


class GBMQuantile(Forecaster):
    """One HistGradientBoostingRegressor with quantile loss for each quantile."""

    name = "gbm"

    def _estimator(self, q: float):
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            learning_rate=0.06,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            random_state=self.seed,
        )

    def fit(self, rows):
        X, y = rows[list(FEATURES)], rows["y"].to_numpy()
        # Many OpenMP threads on a small table are slower than one thread.
        with threadpool_limits(limits=self.threads):
            self.pipelines_ = [
                Pipeline([("drop_constant", DropConstant()), ("model", self._estimator(q))]).fit(X, y)
                for q in self.quantiles
            ]
        return self

    def predict(self, rows):
        X = rows[list(FEATURES)]
        with threadpool_limits(limits=self.threads):
            return sort_quantiles(np.column_stack([p.predict(X) for p in self.pipelines_]))


class LightGBMQuantile(GBMQuantile):
    name = "lightgbm"

    def _estimator(self, q: float):
        try:
            import lightgbm  # noqa: PLC0415  (optional extra)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('the lightgbm model needs: pip install "hotfood-forecast[lightgbm]"') from exc
        return lightgbm.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=200, learning_rate=0.05,
            num_leaves=15, min_child_samples=30, random_state=self.seed, n_jobs=self.threads, verbose=-1,
        )


MODELS: dict[str, Callable[..., Forecaster]] = {
    "seasonal_naive": SeasonalNaive,
    "weekday_mean": WeekdayMean,
    "ridge": RidgeForecaster,
    "gbm": GBMQuantile,
    "lightgbm": LightGBMQuantile,
}
BASELINES = ("seasonal_naive", "weekday_mean")


def make_model(name: str, quantiles: tuple[float, ...], seed: int = 42, threads: int = 1) -> Forecaster:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; use one of {sorted(MODELS)}")
    return MODELS[name](quantiles=quantiles, seed=seed, threads=threads)

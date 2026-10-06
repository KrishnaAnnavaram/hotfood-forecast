"""Past-only features for a direct multi-horizon forecast.

One row is one (origin, horizon) pair for one category. The origin is the last day with
known sales. The target date is ``origin + horizon``. The features obey two rules:

* Demand features use only days up to and including the origin.
* Calendar and price features describe the target date. They are known in advance.

There is no recursive loop, so a forecast never feeds its own predictions back as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import HolidayCalendar
from .data import SalesData
from .schema import COL_CATEGORY

LOOKBACK = 28  # days of history that the demand features read

DEMAND_FEATURES = ("last_obs", "mean_7", "mean_28", "std_28", "same_weekday_last", "same_weekday_mean_4")
CALENDAR_FEATURES = ("dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "month_sin", "month_cos", "is_holiday")
FEATURES: tuple[str, ...] = ("horizon", *DEMAND_FEATURES, *CALENDAR_FEATURES, "price")


def _shift(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return ``a[idx]`` with NaN where ``idx`` is out of range."""
    out = np.full(idx.shape, np.nan)
    ok = (idx >= 0) & (idx < len(a))
    out[ok] = a[idx[ok]]
    return out


def _demand_arrays(y: np.ndarray) -> dict[str, np.ndarray]:
    s = pd.Series(y)
    weekly = pd.concat([s.shift(7 * k) for k in range(4)], axis=1)
    return {
        "last_obs": s.ffill().to_numpy(),
        "mean_7": s.rolling(7, min_periods=1).mean().to_numpy(),
        "mean_28": s.rolling(LOOKBACK, min_periods=1).mean().to_numpy(),
        "std_28": s.rolling(LOOKBACK, min_periods=2).std().to_numpy(),
        "weekly_mean_4": weekly.mean(axis=1, skipna=True).to_numpy(),
    }


def build_rows(
    sales: SalesData,
    category: str,
    origins: np.ndarray,
    horizons: range | list[int],
    calendar: HolidayCalendar,
) -> pd.DataFrame:
    """Build the feature rows for the given origin positions (indexes into ``sales.dates``).

    The ``y`` column holds the observed units of the target date, or NaN if that day is
    closed, missing or after the end of the data.
    """
    horizons = list(horizons)
    origins = np.asarray(origins, dtype=int)
    y = sales.target()[category].to_numpy(dtype=float)
    n = len(y)
    dates = sales.dates
    max_h = max(horizons)
    ext = pd.date_range(dates[0], periods=n + max_h, freq="D")
    hol = calendar.flags(ext).to_numpy()
    price = sales.prices[category].reindex(ext).ffill().to_numpy(dtype=float)
    arr = _demand_arrays(y)

    o = np.repeat(origins, len(horizons))
    h = np.tile(np.asarray(horizons, dtype=int), len(origins))
    t = o + h
    k = np.ceil(h / 7).astype(int)
    base = t - 7 * k  # the latest day before or at the origin with the same weekday as the target

    tdates = ext[t]
    dow = tdates.dayofweek.to_numpy()
    month = tdates.month.to_numpy()
    rows = {
        "origin": dates[o],
        "target_date": tdates,
        COL_CATEGORY: category,
        "horizon": h,
        "last_obs": _shift(arr["last_obs"], o),
        "mean_7": _shift(arr["mean_7"], o),
        "mean_28": _shift(arr["mean_28"], o),
        "std_28": _shift(arr["std_28"], o),
        "same_weekday_last": _shift(y, base),
        "same_weekday_mean_4": _shift(arr["weekly_mean_4"], base),
    }
    for d in range(1, 7):
        rows[f"dow_{d}"] = (dow == d).astype(int)
    rows["month_sin"] = np.sin(2 * np.pi * month / 12)
    rows["month_cos"] = np.cos(2 * np.pi * month / 12)
    rows["is_holiday"] = hol[t]
    rows["price"] = price[t]
    rows["y"] = _shift(y, t)
    return pd.DataFrame(rows)


def training_rows(
    sales: SalesData, category: str, horizons: range | list[int], calendar: HolidayCalendar
) -> pd.DataFrame:
    """All rows whose target date is inside the data and observed. Use them to fit a model."""
    n = len(sales.dates)
    origins = np.arange(LOOKBACK - 1, n - 1)
    rows = build_rows(sales, category, origins, horizons, calendar)
    return rows[rows["y"].notna()].reset_index(drop=True)


def forecast_rows(
    sales: SalesData, category: str, horizons: range | list[int], calendar: HolidayCalendar
) -> pd.DataFrame:
    """Rows for the last day of the data as the origin. Their ``y`` is unknown."""
    origin = np.array([len(sales.dates) - 1])
    return build_rows(sales, category, origin, horizons, calendar)

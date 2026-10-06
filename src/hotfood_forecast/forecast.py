"""Final forecast and order quantities.

The final model is a NEW model fit on all observed rows. It is not a model object from a
backtest fold. The forecast is direct: each horizon has its own feature row, and the
calendar features describe the target date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import HolidayCalendar
from .config import Settings
from .data import SalesData
from .features import forecast_rows, training_rows
from .metrics import qcol
from .models import Forecaster, make_model
from .newsvendor import CategoryCosts, order_quantity
from .schema import CATEGORIES


def fit_final(
    sales: SalesData, model_name: str, settings: Settings, calendar: HolidayCalendar, horizon: int
) -> dict[str, Forecaster]:
    horizons = range(1, horizon + 1)
    return {
        c: make_model(model_name, settings.quantiles, settings.seed, settings.threads).fit(training_rows(sales, c, horizons, calendar))
        for c in CATEGORIES
    }


def forecast(
    sales: SalesData,
    model_name: str,
    settings: Settings,
    calendar: HolidayCalendar,
    costs: dict[str, CategoryCosts],
    horizon: int | None = None,
) -> pd.DataFrame:
    """Quantile forecasts and newsvendor orders for the days after the last date in the data."""
    horizon = horizon or settings.horizon
    models = fit_final(sales, model_name, settings, calendar, horizon)
    parts = []
    for c, model in models.items():
        rows = forecast_rows(sales, c, range(1, horizon + 1), calendar)
        pred = model.predict(rows)
        out = rows[["origin", "target_date", "category", "horizon", "is_holiday"]].copy()
        out[[qcol(q) for q in settings.quantiles]] = np.round(pred, 2)
        out["critical_ratio"] = round(costs[c].critical_ratio, 3)
        out["order"] = order_quantity(pred, settings.quantiles, costs[c].critical_ratio)
        parts.append(out)
    return pd.concat(parts, ignore_index=True).sort_values(["target_date", "category"]).reset_index(drop=True)


def orders_for_date(
    sales: SalesData,
    date: str | pd.Timestamp,
    model_name: str,
    settings: Settings,
    calendar: HolidayCalendar,
    costs: dict[str, CategoryCosts],
) -> pd.DataFrame:
    """Order quantities for one future date. The date must be 1..28 days after the last date."""
    date = pd.Timestamp(date).normalize()
    last = sales.dates[-1]
    horizon = (date - last).days
    if not 1 <= horizon <= 28:
        raise ValueError(f"{date.date()} is {horizon} days after the last date {last.date()}; use 1..28 days")
    fc = forecast(sales, model_name, settings, calendar, costs, horizon=horizon)
    return fc[fc["target_date"] == date].reset_index(drop=True)

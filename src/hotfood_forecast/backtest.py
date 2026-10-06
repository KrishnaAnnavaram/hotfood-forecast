"""Rolling-origin backtest with an expanding training window.

For each origin, every model is made new and fit only on rows whose target date is on or
before the origin. Then it forecasts horizons 1..H from that origin. The forecasts and the
actuals go into one table, and all metrics come from that table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import HolidayCalendar
from .config import Settings
from .data import SalesData
from .features import LOOKBACK, build_rows, training_rows
from .metrics import evaluate, qcol, wape_ci
from .models import make_model
from .newsvendor import CategoryCosts, order_quantity
from .schema import CATEGORIES
from .simulate import simulate

MIN_TRAIN_DAYS = 56


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame  # one row per (model, origin, target_date, category, horizon)
    metrics: pd.DataFrame  # one row per model
    by_category: pd.DataFrame
    by_horizon: pd.DataFrame
    policies: pd.DataFrame  # one row per (model, policy)
    origins: list[pd.Timestamp]


def origin_positions(n_days: int, horizon: int, n_origins: int, step: int) -> np.ndarray:
    """Positions of the forecast origins, oldest first. The last origin leaves ``horizon`` days to score."""
    last = n_days - 1 - horizon
    pos = np.array([last - step * i for i in range(n_origins)])[::-1]
    pos = pos[pos >= LOOKBACK + MIN_TRAIN_DAYS]
    if len(pos) == 0:
        raise ValueError("the data is too short for a backtest with these settings")
    return pos


def run_backtest(
    sales: SalesData,
    models: list[str],
    settings: Settings,
    calendar: HolidayCalendar,
    costs: dict[str, CategoryCosts],
    n_origins: int = 12,
    step: int = 7,
) -> BacktestResult:
    horizons = range(1, settings.horizon + 1)
    pos = origin_positions(len(sales.dates), settings.horizon, n_origins, step)
    qcols = [qcol(q) for q in settings.quantiles]
    parts = []
    for cat in CATEGORIES:
        all_train = training_rows(sales, cat, horizons, calendar)
        ratio = costs[cat].critical_ratio
        for o in pos:
            cutoff = sales.dates[o]
            train = all_train[all_train["target_date"] <= cutoff]
            test = build_rows(sales, cat, np.array([o]), horizons, calendar)
            test = test[test["y"].notna()]
            if test.empty:
                continue
            for name in models:
                model = make_model(name, settings.quantiles, settings.seed, settings.threads).fit(train)
                pred = model.predict(test)
                out = test[["origin", "target_date", "category", "horizon", "y"]].copy()
                out[qcols] = pred
                out["model"] = name
                out["order_newsvendor"] = order_quantity(pred, settings.quantiles, ratio)
                out["order_p50"] = np.ceil(out[qcol(0.5)] - 1e-9).astype(int)
                parts.append(out)
    fc = pd.concat(parts, ignore_index=True)

    rows, cat_rows, h_rows, pol_rows = [], [], [], []
    for name, g in fc.groupby("model", sort=False):
        lo, hi = wape_ci(g, seed=settings.seed)
        rows.append({"model": name, **evaluate(g), "wape_ci_low": lo, "wape_ci_high": hi})
        for cat, gc in g.groupby("category", sort=False):
            cat_rows.append({"model": name, "category": cat, **evaluate(gc)})
        for h, gh in g.groupby("horizon"):
            h_rows.append({"model": name, "horizon": int(h), **evaluate(gh)})
        for policy in ("order_newsvendor", "order_p50"):
            pol_rows.append({"model": name, "policy": policy.removeprefix("order_"), **simulate(g, policy, costs)})
    return BacktestResult(
        forecasts=fc,
        metrics=pd.DataFrame(rows).sort_values("wape").reset_index(drop=True),
        by_category=pd.DataFrame(cat_rows),
        by_horizon=pd.DataFrame(h_rows),
        policies=pd.DataFrame(pol_rows).sort_values("profit", ascending=False).reset_index(drop=True),
        origins=[sales.dates[o] for o in pos],
    )

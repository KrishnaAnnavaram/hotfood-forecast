"""Synthetic daily sales in the store-file layout.

The generator gives a known demand process: weekly and yearly patterns, holiday effects,
price levels with a negative price effect, over-dispersed counts and closure days. The tests
and the offline demo use it, so nothing needs a download.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import USFederalHolidays
from .schema import CATEGORIES, DATE_COL, HOLIDAY_COL, PRICE_COLS

BASE_LEVEL = {"Chicken": 60.0, "FriedSnacks": 35.0, "FriedBurritos": 22.0, "Otherfood": 40.0}
BASE_PRICE = {"Chicken": 2.49, "FriedSnacks": 1.29, "FriedBurritos": 1.99, "Otherfood": 2.19}
# Monday .. Sunday multipliers
WEEKLY = {
    "Chicken": (0.85, 0.85, 0.9, 0.95, 1.2, 1.35, 1.1),
    "FriedSnacks": (1.1, 1.05, 1.05, 1.0, 1.0, 0.9, 0.85),
    "FriedBurritos": (1.15, 1.1, 1.1, 1.05, 0.95, 0.8, 0.75),
    "Otherfood": (0.95, 0.95, 1.0, 1.0, 1.1, 1.05, 0.95),
}
HOLIDAY_EFFECT = {"Chicken": 1.3, "FriedSnacks": 0.7, "FriedBurritos": 0.65, "Otherfood": 1.1}
PRICE_ELASTICITY = -1.2
DISPERSION = 12.0  # gamma shape: smaller value = more noise


def generate(
    n_days: int = 730,
    start: str = "2022-01-03",
    seed: int = 42,
    closure_rate: float = 0.01,
) -> pd.DataFrame:
    """Return a frame with the columns of the store file (dates as M/D/YYYY text)."""
    if n_days < 1:
        raise ValueError("n_days must be positive")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    holiday = USFederalHolidays().flags(dates).to_numpy()
    t = np.arange(n_days)
    yearly = 1 + 0.15 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 80) / 365.25)
    trend = 1 + 0.08 * t / 365.25

    closed = rng.random(n_days) < closure_rate
    closed |= (dates.month == 12) & (dates.day == 25)

    out = pd.DataFrame({DATE_COL: [f"{d.month}/{d.day}/{d.year}" for d in dates]})
    for c in CATEGORIES:
        # price changes three times, which gives four price levels
        cuts = np.sort(rng.choice(np.arange(60, max(61, n_days - 60)), size=3, replace=False)) if n_days > 130 else []
        level = np.zeros(n_days, dtype=int)
        for cut in cuts:
            level[cut:] += 1
        steps = np.array([1.0, 1.05, 0.97, 1.1])
        price = np.round(BASE_PRICE[c] * steps[level], 2)
        weekly = np.asarray(WEEKLY[c])[dates.dayofweek.to_numpy()]
        hol = np.where(holiday == 1, HOLIDAY_EFFECT[c], 1.0)
        price_eff = (price / BASE_PRICE[c]) ** PRICE_ELASTICITY
        mean = BASE_LEVEL[c] * weekly * yearly * trend * hol * price_eff
        lam = rng.gamma(DISPERSION, mean / DISPERSION)
        units = rng.poisson(lam).astype(int)
        units[closed] = 0
        out[c] = units
        out[PRICE_COLS[c]] = price
    out[HOLIDAY_COL] = holiday
    return out


def write(path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(**kwargs).to_csv(path, index=False)
    return path

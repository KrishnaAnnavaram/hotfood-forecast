"""Waste and stock-out simulation for an order policy on backtest days.

For each row: sold = min(order, demand), waste = max(order - demand, 0) and
stock-out = max(demand - order, 0). Recorded sales are a lower bound of demand on days with a
stock-out, so the simulation uses recorded sales as demand. ``Known problems`` in the README
tells more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .newsvendor import CategoryCosts


def simulate(frame: pd.DataFrame, order_col: str, costs: dict[str, CategoryCosts]) -> dict[str, float]:
    """Totals for one policy. ``frame`` needs the columns ``category``, ``y`` and ``order_col``."""
    y = frame["y"].to_numpy(dtype=float)
    q = frame[order_col].to_numpy(dtype=float)
    if (q < 0).any():
        raise ValueError("orders must not be negative")
    price = frame["category"].map(lambda c: costs[c].price).to_numpy(dtype=float)
    cost = frame["category"].map(lambda c: costs[c].unit_cost).to_numpy(dtype=float)
    disp = frame["category"].map(lambda c: costs[c].disposal_cost).to_numpy(dtype=float)
    sold = np.minimum(q, y)
    waste = np.maximum(q - y, 0)
    short = np.maximum(y - q, 0)
    profit = price * sold - cost * q - disp * waste
    demand = y.sum()
    return {
        "ordered": float(q.sum()),
        "sold": float(sold.sum()),
        "waste_units": float(waste.sum()),
        "stockout_units": float(short.sum()),
        "waste_rate": float(waste.sum() / q.sum()) if q.sum() else 0.0,
        "fill_rate": float(sold.sum() / demand) if demand else 1.0,
        "profit": float(profit.sum()),
    }

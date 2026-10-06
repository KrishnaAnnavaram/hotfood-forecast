"""Order quantities from quantile forecasts (the newsvendor rule).

Hot food has a shelf life of one day. A unit that is not sold is waste. For each unit:

* underage cost ``cu`` = price - unit cost (the margin that a stock-out loses)
* overage cost ``co`` = unit cost + disposal cost (the money that a wasted unit loses)

The order that gives the lowest expected cost is the demand quantile at the critical ratio
``cu / (cu + co)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings
from .data import SalesData
from .schema import CATEGORIES

DEFAULT_PRICE = 2.0  # used only when the data has no price column and no override is set


@dataclass(frozen=True)
class CategoryCosts:
    price: float
    unit_cost: float
    disposal_cost: float

    def __post_init__(self) -> None:
        if self.unit_cost < 0 or self.disposal_cost < 0:
            raise ValueError("costs must not be negative")
        if self.price <= self.unit_cost:
            raise ValueError(f"price {self.price} must be larger than the unit cost {self.unit_cost}")

    @property
    def underage(self) -> float:
        return self.price - self.unit_cost

    @property
    def overage(self) -> float:
        return self.unit_cost + self.disposal_cost

    @property
    def critical_ratio(self) -> float:
        return critical_ratio(self.underage, self.overage)


def critical_ratio(cu: float, co: float) -> float:
    if cu <= 0 or co <= 0:
        raise ValueError("underage and overage costs must be positive")
    return cu / (cu + co)


def category_costs(settings: Settings, sales: SalesData | None = None) -> dict[str, CategoryCosts]:
    """Costs per category: explicit settings first, then the last price in the data."""
    out = {}
    for c in CATEGORIES:
        price = settings.unit_price.get(c)
        if price is None and sales is not None:
            p = sales.prices[c].dropna()
            price = float(p.iloc[-1]) if len(p) else None
        price = DEFAULT_PRICE if price is None else price
        cost = settings.unit_cost.get(c, round(price * settings.cost_ratio, 4))
        out[c] = CategoryCosts(price=price, unit_cost=cost, disposal_cost=settings.disposal_cost)
    return out


def order_quantity(qvalues: np.ndarray, quantiles: tuple[float, ...], ratio: float) -> np.ndarray:
    """Interpolate the demand quantile at ``ratio`` for each row and round up to whole units.

    If ``ratio`` is outside the quantile grid, the nearest grid quantile is used.
    """
    qvalues = np.atleast_2d(np.asarray(qvalues, dtype=float))
    qs = np.asarray(quantiles, dtype=float)
    if qvalues.shape[1] != len(qs):
        raise ValueError("one column per quantile is necessary")
    order = np.argsort(qs)
    qs, qvalues = qs[order], np.sort(qvalues[:, order], axis=1)
    level = np.array([np.interp(ratio, qs, row) for row in qvalues])
    return np.ceil(np.clip(level, 0, None) - 1e-9).astype(int)

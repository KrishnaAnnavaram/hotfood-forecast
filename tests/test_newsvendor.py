"""Problem 8 (no order decision and no waste model)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hotfood_forecast.config import Settings
from hotfood_forecast.newsvendor import CategoryCosts, category_costs, critical_ratio, order_quantity
from hotfood_forecast.simulate import simulate


def test_critical_ratio():
    assert critical_ratio(3.0, 1.0) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        critical_ratio(0.0, 1.0)


def test_category_costs_from_price():
    c = CategoryCosts(price=2.5, unit_cost=1.0, disposal_cost=0.1)
    assert c.underage == pytest.approx(1.5)
    assert c.overage == pytest.approx(1.1)
    assert c.critical_ratio == pytest.approx(1.5 / 2.6)
    with pytest.raises(ValueError, match="larger than the unit cost"):
        CategoryCosts(price=1.0, unit_cost=1.0, disposal_cost=0.0)


def test_costs_use_last_price_and_overrides(sales):
    s = replace(Settings(), unit_cost={"Chicken": 0.5})
    costs = category_costs(s, sales)
    assert costs["Chicken"].unit_cost == 0.5
    assert costs["Otherfood"].price == pytest.approx(sales.prices["Otherfood"].iloc[-1])
    assert costs["Otherfood"].unit_cost == pytest.approx(costs["Otherfood"].price * 0.4, abs=1e-3)


def test_order_quantity_interpolates_and_rounds_up():
    q = (0.1, 0.5, 0.9)
    vals = np.array([[10.0, 20.0, 30.0]])
    assert order_quantity(vals, q, 0.5).tolist() == [20]
    assert order_quantity(vals, q, 0.7).tolist() == [25]
    assert order_quantity(vals, q, 0.72).tolist() == [26]  # 25.5 rounds up
    assert order_quantity(vals, q, 0.99).tolist() == [30]  # outside the grid: nearest quantile


def test_higher_margin_orders_more():
    vals = np.array([[10.0, 20.0, 30.0]])
    low = CategoryCosts(price=1.2, unit_cost=1.0, disposal_cost=0.1).critical_ratio
    high = CategoryCosts(price=4.0, unit_cost=1.0, disposal_cost=0.1).critical_ratio
    assert order_quantity(vals, (0.1, 0.5, 0.9), high)[0] > order_quantity(vals, (0.1, 0.5, 0.9), low)[0]


def test_newsvendor_beats_median_on_known_distribution():
    rng = np.random.default_rng(0)
    demand = rng.normal(100, 20, 5000).clip(0).round()
    costs = {"Chicken": CategoryCosts(price=3.0, unit_cost=0.6, disposal_cost=0.0)}
    ratio = costs["Chicken"].critical_ratio  # 0.8
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    true_q = np.quantile(demand, qs)
    f = pd.DataFrame({"category": "Chicken", "y": demand})
    f["nv"] = order_quantity(np.tile(true_q, (len(f), 1)), qs, ratio)
    f["p50"] = np.ceil(true_q[2])
    assert simulate(f, "nv", costs)["profit"] > simulate(f, "p50", costs)["profit"]


def test_simulate_known_values():
    costs = {"Chicken": CategoryCosts(price=2.0, unit_cost=0.5, disposal_cost=0.1)}
    f = pd.DataFrame({"category": ["Chicken", "Chicken"], "y": [10.0, 4.0], "order": [8, 6]})
    r = simulate(f, "order", costs)
    assert r["sold"] == 12 and r["waste_units"] == 2 and r["stockout_units"] == 2
    assert r["profit"] == pytest.approx(2.0 * 12 - 0.5 * 14 - 0.1 * 2)
    assert r["fill_rate"] == pytest.approx(12 / 14)

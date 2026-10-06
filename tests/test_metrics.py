"""Problem 2 (mixed-up evaluation arrays) and problem 3 (MAPE with zero sales)."""
import numpy as np
import pandas as pd
import pytest

from hotfood_forecast.metrics import align, evaluate, pinball, smape, wape_ci


def _frame(y, p50, lo=None, hi=None):
    n = len(y)
    f = pd.DataFrame({
        "target_date": pd.date_range("2024-01-01", periods=n),
        "category": "Chicken",
        "horizon": 1,
        "y": y,
        "q0.5": p50,
    })
    f["q0.1"] = lo if lo is not None else np.array(p50) - 1
    f["q0.9"] = hi if hi is not None else np.array(p50) + 1
    return f


def test_evaluate_known_values():
    m = evaluate(_frame([10.0, 0.0, 20.0], [12.0, 0.0, 14.0]))
    assert m["mae"] == pytest.approx(8 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 0 + 36) / 3))
    assert m["wape"] == pytest.approx(8 / 30)
    assert m["bias"] == pytest.approx((2 + 0 - 6) / 3)
    assert m["coverage"] == pytest.approx(1 / 3)
    assert m["nominal_coverage"] == pytest.approx(0.8)


def test_zero_sales_do_not_break_metrics():
    m = evaluate(_frame([0.0, 0.0, 5.0], [0.0, 1.0, 5.0]))
    assert np.isfinite(m["smape"]) and np.isfinite(m["wape"])
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0


def test_pinball_loss():
    y = np.array([10.0])
    assert pinball(y, np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball(y, np.array([12.0]), 0.9) == pytest.approx(0.2)


def test_align_joins_on_keys_not_positions():
    fc = _frame([0, 0, 0], [1.0, 2.0, 3.0]).drop(columns="y")
    actual = _frame([5.0, 6.0, 7.0], [0, 0, 0])[["target_date", "category", "horizon", "y"]]
    merged = align(actual.iloc[::-1], fc)  # reversed order must not matter
    assert list(merged["y"]) == [5.0, 6.0, 7.0]


def test_align_refuses_missing_or_duplicate_rows():
    fc = _frame([0, 0], [1.0, 2.0]).drop(columns="y")
    actual = _frame([5.0, 6.0], [0, 0])[["target_date", "category", "horizon", "y"]]
    with pytest.raises(ValueError, match="no observed actual"):
        align(actual.iloc[:1], fc)
    with pytest.raises(ValueError, match="duplicate"):
        align(pd.concat([actual, actual]), fc)


def test_evaluate_needs_median_column():
    with pytest.raises(ValueError):
        evaluate(_frame([1.0], [1.0]).drop(columns="q0.5"))


def test_wape_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.poisson(30, 200).astype(float)
    f = _frame(y, y + rng.normal(0, 5, 200))
    lo, hi = wape_ci(f)
    assert lo <= evaluate(f)["wape"] <= hi

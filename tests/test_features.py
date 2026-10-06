"""Problem 1 (same-day leakage), problem 4 (recursive column scramble) and problem 7 (redundant features)."""
import numpy as np
import pandas as pd

from hotfood_forecast.data import SalesData
from hotfood_forecast.features import FEATURES, build_rows, forecast_rows, training_rows


def _with_future_changed(sales: SalesData, origin: int) -> SalesData:
    units = sales.units.copy()
    units.iloc[origin + 1 :] = 999.0
    return SalesData(units, sales.prices, sales.closed, sales.missing, sales.report)


def test_features_do_not_read_after_the_origin(sales, calendar):
    o = 150
    a = build_rows(sales, "Chicken", np.array([o]), range(1, 8), calendar)
    b = build_rows(_with_future_changed(sales, o), "Chicken", np.array([o]), range(1, 8), calendar)
    pd.testing.assert_frame_equal(a[list(FEATURES)], b[list(FEATURES)])
    assert not a["y"].equals(b["y"])  # only the target changes


def test_rolling_mean_excludes_the_target_day(sales, calendar):
    o = 120
    rows = build_rows(sales, "FriedSnacks", np.array([o]), [1], calendar)
    y = sales.target()["FriedSnacks"].to_numpy()
    assert np.isclose(rows["mean_7"].iloc[0], np.nanmean(y[o - 6 : o + 1]))
    assert rows["target_date"].iloc[0] == sales.dates[o + 1]


def test_same_weekday_feature_has_target_weekday_and_is_in_the_past(sales, calendar):
    o = 140
    rows = build_rows(sales, "Otherfood", np.array([o]), range(1, 15), calendar)
    y = sales.target()["Otherfood"].to_numpy()
    for _, r in rows.iterrows():
        h = int(r["horizon"])
        base = o + h - 7 * int(np.ceil(h / 7))
        assert base <= o
        assert sales.dates[base].dayofweek == r["target_date"].dayofweek
        assert np.isclose(r["same_weekday_last"], y[base], equal_nan=True)


def test_calendar_features_advance_with_the_target_date(sales, calendar):
    rows = forecast_rows(sales, "Chicken", range(1, 8), calendar)
    assert list(rows["target_date"]) == list(pd.date_range(sales.dates[-1] + pd.Timedelta(days=1), periods=7))
    dows = rows["target_date"].dt.dayofweek.to_numpy()
    onehot = rows[[f"dow_{d}" for d in range(1, 7)]].to_numpy()
    for dow, row in zip(dows, onehot):
        assert row.sum() == (0 if dow == 0 else 1)
        if dow:
            assert row[dow - 1] == 1
    # every horizon uses the same past, so the demand features do not change with the horizon
    assert rows["mean_7"].nunique() == 1


def test_truncated_data_gives_the_same_training_rows(sales, calendar):
    cut = sales.dates[180]
    full = training_rows(sales, "Chicken", range(1, 8), calendar)
    full = full[full["target_date"] <= cut].reset_index(drop=True)
    trunc = training_rows(sales.until(cut), "Chicken", range(1, 8), calendar)
    pd.testing.assert_frame_equal(full, trunc)


def test_feature_list_has_no_duplicates_or_complements():
    assert len(FEATURES) == len(set(FEATURES))
    assert "dow_0" not in FEATURES  # Monday is the reference level
    assert not {"weekday", "is_weekend"} & set(FEATURES)


def test_training_rows_skip_closed_targets(sales, calendar):
    rows = training_rows(sales, "Chicken", range(1, 4), calendar)
    closed_dates = set(sales.dates[sales.closed.to_numpy()])
    assert closed_dates
    assert not set(rows["target_date"]) & closed_dates
    assert rows["y"].notna().all()

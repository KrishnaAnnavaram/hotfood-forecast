"""Problem 6 (the forecast used the last fold model) and the backtest contract."""
from dataclasses import replace

import pandas as pd
import pytest

from hotfood_forecast import backtest as bt
from hotfood_forecast import forecast as fcmod
from hotfood_forecast.config import Settings
from hotfood_forecast.forecast import forecast, orders_for_date
from hotfood_forecast.newsvendor import category_costs

SETTINGS = replace(Settings(), quantiles=(0.1, 0.5, 0.9), horizon=3)


class _Recorder:
    def __init__(self, real):
        self.real = real
        self.fits = []
        self.models = []  # keep every model alive so id() values cannot be reused after garbage collection

    def __call__(self, *a, **k):
        model = self.real(*a, **k)
        self.models.append(model)
        fit = model.fit

        def recording_fit(rows):
            self.fits.append((id(model), rows["target_date"].max(), len(rows)))
            return fit(rows)

        model.fit = recording_fit
        return model


def test_each_fold_fits_a_new_model_on_past_rows_only(sales, calendar, monkeypatch):
    rec = _Recorder(bt.make_model)
    monkeypatch.setattr(bt, "make_model", rec)
    res = bt.run_backtest(sales, ["weekday_mean"], SETTINGS, calendar, category_costs(SETTINGS, sales),
                          n_origins=4, step=7)
    assert len(rec.fits) == 4 * 4  # 4 categories x 4 origins
    assert len({mid for mid, _, _ in rec.fits}) == len(rec.fits)
    allowed = set(res.origins)
    assert all(max_date <= max(allowed) for _, max_date, _ in rec.fits)
    # every forecast row has a target after its origin and within the horizon
    fc = res.forecasts
    assert ((fc["target_date"] - fc["origin"]).dt.days.between(1, 3)).all()
    assert (fc["origin"].isin(allowed)).all()


def test_backtest_scores_only_observed_days(sales, calendar):
    res = bt.run_backtest(sales, ["seasonal_naive", "ridge"], SETTINGS, calendar,
                          category_costs(SETTINGS, sales), n_origins=5, step=5)
    closed = set(sales.dates[sales.closed.to_numpy()])
    assert not set(res.forecasts["target_date"]) & closed
    assert set(res.metrics["model"]) == {"seasonal_naive", "ridge"}
    assert set(res.policies["policy"]) == {"newsvendor", "p50"}
    assert (res.metrics["wape_ci_low"] <= res.metrics["wape"]).all()


def test_origin_positions_need_enough_history():
    with pytest.raises(ValueError):
        bt.origin_positions(n_days=60, horizon=7, n_origins=3, step=7)
    pos = bt.origin_positions(n_days=300, horizon=7, n_origins=3, step=7)
    assert list(pos) == [278, 285, 292]


def test_final_forecast_refits_on_all_rows(sales, calendar, monkeypatch):
    rec = _Recorder(fcmod.make_model)
    monkeypatch.setattr(fcmod, "make_model", rec)
    fc = forecast(sales, "ridge", SETTINGS, calendar, category_costs(SETTINGS, sales))
    assert len(rec.fits) == 4
    last_observed = sales.dates[sales.observed.to_numpy()][-1]
    assert all(max_date == last_observed for _, max_date, _ in rec.fits)
    assert len(fc) == 4 * 3
    assert (fc["order"] >= 0).all()


def test_orders_for_date_checks_the_range(sales, calendar):
    costs = category_costs(SETTINGS, sales)
    day = sales.dates[-1] + pd.Timedelta(days=2)
    out = orders_for_date(sales, day, "weekday_mean", SETTINGS, calendar, costs)
    assert len(out) == 4 and (out["target_date"] == day).all()
    with pytest.raises(ValueError, match="days after the last date"):
        orders_for_date(sales, sales.dates[-1], "weekday_mean", SETTINGS, calendar, costs)

import numpy as np
import pandas as pd
import pytest

from hotfood_forecast import synthetic
from hotfood_forecast.data import DataValidationError, load_sales, validate_frame
from hotfood_forecast.schema import CATEGORIES


def test_synthetic_is_deterministic():
    a = synthetic.generate(n_days=100, seed=3)
    b = synthetic.generate(n_days=100, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(synthetic.generate(n_days=100, seed=4))


def test_valid_frame_has_complete_calendar(sales):
    assert sales.report.ok
    assert len(sales.dates) == 240
    assert (np.diff(sales.dates.values).astype("timedelta64[D]").astype(int) == 1).all()


def test_closure_days_are_flagged_and_excluded_from_target(raw):
    df = raw.copy()
    for c in CATEGORIES:
        df.loc[10, c] = 0
    s = validate_frame(df)
    assert s.closed.iloc[10]
    assert s.target().iloc[10].isna().all()
    assert s.report.info["closed_days"] >= 1


def test_calendar_gap_is_flagged_not_filled(raw):
    s = validate_frame(raw.drop(index=[20, 21]))
    assert s.missing.iloc[20] and s.missing.iloc[21]
    assert s.units.iloc[20].isna().all()
    assert any("no usable row" in w for w in s.report.warnings)


def test_us_and_iso_dates(raw):
    iso = raw.copy()
    iso["Date"] = pd.to_datetime(iso["Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    assert (validate_frame(iso).dates == validate_frame(raw).dates).all()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns=["Chicken"]), "missing required columns"),
        (lambda d: pd.concat([d, d.iloc[[5]]]), "duplicate dates"),
        (lambda d: d.assign(Chicken=-d["Chicken"] - 1), "negative units"),
        (lambda d: d.assign(Date=["not a date"] + list(d["Date"].iloc[1:])), "unparseable dates"),
        (lambda d: d.iloc[:30], "observed days"),
    ],
)
def test_validation_errors(raw, mutate, message):
    with pytest.raises(DataValidationError, match=message):
        validate_frame(mutate(raw.copy()))


def test_until_keeps_only_past(sales):
    cut = sales.dates[100]
    s = sales.until(cut)
    assert s.dates[-1] == cut and len(s.dates) == 101


def test_load_sales_missing_file_tells_how_to_get_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="hotfood synth"):
        load_sales(tmp_path / "nope.csv")


def test_holiday_flag_mismatch_is_reported(tmp_path, raw, calendar):
    df = raw.copy()
    df["Is Holiday"] = 1 - df["Is Holiday"]
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    s = load_sales(p, holiday_flags_fn=calendar.flags)
    assert any("holiday flag differs" in w for w in s.report.warnings)

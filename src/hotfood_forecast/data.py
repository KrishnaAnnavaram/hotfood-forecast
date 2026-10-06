"""Load the daily sales file and validate it.

The loader gives a ``SalesData`` object with a complete daily calendar. Closure days and
calendar gaps are flags. They are never silently filled with a median or with zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CATEGORIES, CLOSED_COL, DATE_COL, HOLIDAY_COL, PRICE_COLS, REQUIRED_COLS

MIN_OBSERVED_DAYS = 56  # 8 weeks: the features look back 28 days, the backtest needs more


class DataValidationError(ValueError):
    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__("; ".join(report.errors))


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def lines(self) -> list[str]:
        out = [f"{k}: {v}" for k, v in self.info.items()]
        out += [f"WARNING: {w}" for w in self.warnings]
        out += [f"ERROR: {e}" for e in self.errors]
        return out


@dataclass
class SalesData:
    """Daily sales on a complete calendar (one row per day from the first to the last date)."""

    units: pd.DataFrame  # columns = CATEGORIES, NaN on calendar gaps
    prices: pd.DataFrame  # columns = CATEGORIES, forward-filled, NaN if the file has no price
    closed: pd.Series  # bool, True on closure days
    missing: pd.Series  # bool, True on days that are not in the file
    report: ValidationReport

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.units.index)

    @property
    def observed(self) -> pd.Series:
        """True on days with a usable demand observation (open and present in the file)."""
        return ~(self.closed | self.missing)

    def target(self) -> pd.DataFrame:
        """Units on observed days, NaN on closure days and calendar gaps."""
        return self.units.where(self.observed, np.nan)

    def until(self, last_date: pd.Timestamp) -> "SalesData":
        """Return only the days up to and including ``last_date``."""
        keep = self.dates <= pd.Timestamp(last_date)
        return SalesData(self.units[keep], self.prices[keep], self.closed[keep], self.missing[keep], self.report)


def _parse_dates(raw: pd.Series) -> pd.Series:
    text = raw.astype(str).str.strip()
    us = pd.to_datetime(text, format="%m/%d/%Y", errors="coerce")
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    return us.fillna(iso)


def _robust_outliers(x: pd.Series, z: float = 8.0) -> int:
    x = x.dropna()
    if len(x) < 10:
        return 0
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if mad == 0:
        return 0
    return int(((x - med).abs() / mad > z).sum())


def validate_frame(df: pd.DataFrame, holiday_flags: pd.Series | None = None) -> SalesData:
    """Validate a raw daily frame in the store-file layout. Raise ``DataValidationError`` on errors."""
    rep = ValidationReport()
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"missing required columns: {missing_cols}")
        raise DataValidationError(rep)

    df = df.copy()
    dates = _parse_dates(df[DATE_COL])
    bad = df.loc[dates.isna(), DATE_COL].astype(str).head(5).tolist()
    if bad:
        rep.errors.append(f"{int(dates.isna().sum())} unparseable dates, for example {bad}")
    df[DATE_COL] = dates
    df = df[dates.notna()]
    dup = df[DATE_COL][df[DATE_COL].duplicated()].dt.date.astype(str).head(5).tolist()
    if dup:
        rep.errors.append(f"duplicate dates, for example {dup}")

    for c in CATEGORIES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if (df[c] < 0).any():
            rep.errors.append(f"negative units in {c}")
        frac = df[c].dropna() % 1
        if (frac != 0).any():
            rep.warnings.append(f"{c} has non-integer units")
    if rep.errors:
        raise DataValidationError(rep)

    df = df.set_index(DATE_COL).sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    in_file = pd.Series(full.isin(df.index), index=full)
    df = df.reindex(full)

    units = df[list(CATEGORIES)].astype(float)
    if CLOSED_COL in df.columns:
        closed = df[CLOSED_COL].fillna(0).astype(bool)
    else:
        closed = (units.fillna(-1) == 0).all(axis=1)
    missing = ~in_file | (units.isna().any(axis=1) & ~closed)
    closed = closed & in_file

    prices = pd.DataFrame(index=full)
    for c, pc in PRICE_COLS.items():
        if pc in df.columns:
            p = pd.to_numeric(df[pc], errors="coerce")
            if (p < 0).any():
                rep.errors.append(f"negative price in {pc}")
            prices[c] = p.ffill().bfill()
        else:
            prices[c] = np.nan
    if rep.errors:
        raise DataValidationError(rep)

    n_obs = int((~(closed | missing)).sum())
    if n_obs < MIN_OBSERVED_DAYS:
        rep.errors.append(f"only {n_obs} observed days; at least {MIN_OBSERVED_DAYS} are necessary")
        raise DataValidationError(rep)

    if int(missing.sum()):
        rep.warnings.append(f"{int(missing.sum())} calendar days have no usable row; they are excluded")
    outliers = {c: _robust_outliers(units[c].where(~closed)) for c in CATEGORIES}
    if any(outliers.values()):
        rep.warnings.append(f"possible outliers (median-based z > 8), kept as they are: {outliers}")
    if HOLIDAY_COL in df.columns and holiday_flags is not None:
        file_flag = df.loc[in_file.values, HOLIDAY_COL].fillna(0).astype(int)
        cal = holiday_flags.reindex(file_flag.index).fillna(0).astype(int)
        diff = int((file_flag != cal).sum())
        if diff:
            rep.warnings.append(
                f"the file holiday flag differs from the calendar on {diff} days; the model uses the calendar"
            )

    rep.info.update(
        first_date=str(full.min().date()),
        last_date=str(full.max().date()),
        calendar_days=len(full),
        observed_days=n_obs,
        closed_days=int(closed.sum()),
        missing_days=int(missing.sum()),
        has_prices=bool(prices.notna().all().all()),
    )
    return SalesData(units=units, prices=prices, closed=closed.astype(bool), missing=missing.astype(bool), report=rep)


def load_sales(path: str | Path, holiday_flags_fn=None) -> SalesData:
    """Read a CSV in the store-file layout and validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `hotfood synth --out {path}` for synthetic data, or see data/README.md"
        )
    df = pd.read_csv(path)
    flags = None
    if holiday_flags_fn is not None and DATE_COL in df.columns:
        d = _parse_dates(df[DATE_COL]).dropna()
        if len(d):
            flags = holiday_flags_fn(pd.date_range(d.min(), d.max(), freq="D"))
    return validate_frame(df, flags)

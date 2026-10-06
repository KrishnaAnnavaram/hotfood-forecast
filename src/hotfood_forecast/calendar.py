"""Holiday calendars.

A forecast needs the holiday flag of a FUTURE date, so the flag must come from a calendar,
not from the sales file. The same calendar gives the flag for history and for the future,
so training and forecasting use one definition.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


class HolidayCalendar(Protocol):
    name: str

    def flags(self, dates: pd.DatetimeIndex) -> pd.Series: ...


class USFederalHolidays:
    """US federal holidays, computed by rule (no network)."""

    name = "us_federal"

    def flags(self, dates: pd.DatetimeIndex) -> pd.Series:
        dates = pd.DatetimeIndex(dates).normalize()
        if len(dates) == 0:
            return pd.Series([], dtype=int, index=dates)
        hol = USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max())
        return pd.Series(dates.isin(hol).astype(int), index=dates)


class NoHolidays:
    name = "none"

    def flags(self, dates: pd.DatetimeIndex) -> pd.Series:
        dates = pd.DatetimeIndex(dates)
        return pd.Series(0, index=dates, dtype=int)


def get_calendar(name: str) -> HolidayCalendar:
    calendars = {"us_federal": USFederalHolidays, "none": NoHolidays}
    if name not in calendars:
        raise ValueError(f"unknown holiday calendar {name!r}; use one of {sorted(calendars)}")
    return calendars[name]()

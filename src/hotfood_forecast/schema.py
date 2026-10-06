"""Column names of the daily sales file and of the internal tables.

The store file has one row per day. The four category totals are the forecast targets.
The internal "long" table has one row per (date, category).
"""
from __future__ import annotations

DATE_COL = "Date"
HOLIDAY_COL = "Is Holiday"
CLOSED_COL = "Closed"  # optional column in the input file

CATEGORIES: tuple[str, ...] = ("Chicken", "FriedSnacks", "FriedBurritos", "Otherfood")

# The store file uses an irregular name for the last price column. Keep the mapping explicit.
PRICE_COLS: dict[str, str] = {
    "Chicken": "Price_Chicken",
    "FriedSnacks": "Price_FriedSnacks",
    "FriedBurritos": "Price_FriedBurritos",
    "Otherfood": "Price_Other_Food",
}

REQUIRED_COLS: tuple[str, ...] = (DATE_COL, *CATEGORIES)
OPTIONAL_COLS: tuple[str, ...] = (HOLIDAY_COL, CLOSED_COL, *PRICE_COLS.values())

# Long-table columns
COL_DATE = "date"
COL_CATEGORY = "category"
COL_UNITS = "units"
COL_PRICE = "price"
COL_CLOSED = "closed"
COL_MISSING = "missing"

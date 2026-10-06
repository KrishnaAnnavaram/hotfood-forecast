# Data

Git does not track the files in this folder, only this README. No sales data is in the repository.

## Synthetic data (default, no download)

```bash
hotfood synth --out data/synthetic_sales.csv --days 730 --seed 42
```

The generator (`src/hotfood_forecast/synthetic.py`) writes a file in the store layout below. It has weekly and
yearly patterns, US federal holiday effects, four price levels per category, over-dispersed counts and about
1 % closure days (plus 25 December). The tests and `hotfood demo` use it.

## Store data (not included)

| Item | Value |
|---|---|
| Source | Daily sales history of one convenience-store hot-food counter (private store data) |
| URL | None. The data is not public. |
| License and terms | Get written permission from the store owner. Do not commit the file. |
| Period of the earlier analysis | 2017-03-01 to 2024-08-20, 2,730 daily rows, 32 columns |
| Expected file | `data/sales.csv` (or set `HOTFOOD_DATA`) |

A public retail-demand dataset (for example the M5 forecasting data on Kaggle, which has its own competition
terms) can also feed the pipeline after you aggregate it to the columns below.

## Columns

| Column | Required | Type | Meaning |
|---|---|---|---|
| `Date` | Yes | `M/D/YYYY` or `YYYY-MM-DD` | One row per day. Duplicate dates are an error. |
| `Chicken` | Yes | integer ≥ 0 | Units sold in the category. A forecast target. |
| `FriedSnacks` | Yes | integer ≥ 0 | Units sold. A forecast target. |
| `FriedBurritos` | Yes | integer ≥ 0 | Units sold. A forecast target. |
| `Otherfood` | Yes | integer ≥ 0 | Units sold. A forecast target. |
| `Price_Chicken`, `Price_FriedSnacks`, `Price_FriedBurritos`, `Price_Other_Food` | No | number ≥ 0 | Unit price per category. The last price sets the newsvendor costs. |
| `Is Holiday` | No | 0 or 1 | Compared with the holiday calendar. The model uses the calendar. |
| `Closed` | No | 0 or 1 | Closure flag. If it is absent, a day with zero units in all four categories is a closure day. |

Other columns (item-level counts such as `Thighs` or `Wings`) are ignored.

"""Forecast metrics on ONE aligned table.

``evaluate`` takes a table with the actual units in column ``y`` and one column per quantile.
Actuals and forecasts are in the same rows, keyed by target date, category and horizon, so a
metric can never compare arrays from different models, scales or dates.

MAPE is not used: closure days and slow days have zero sales, and MAPE divides by the actual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["target_date", "category", "horizon"]


def qcol(q: float) -> str:
    return f"q{q:g}"


def quantile_cols(frame: pd.DataFrame) -> list[tuple[float, str]]:
    out = []
    for c in frame.columns:
        if isinstance(c, str) and c.startswith("q"):
            try:
                out.append((float(c[1:]), c))
            except ValueError:
                continue
    return sorted(out)


def align(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Join actuals (``KEY`` + ``y``) and forecasts (``KEY`` + quantile columns) on the key.

    Raise if a forecast row has no actual or if a key is duplicated.
    """
    for name, f in (("actual", actual), ("forecast", forecast)):
        if f.duplicated(KEY).any():
            raise ValueError(f"duplicate keys in the {name} table")
    merged = forecast.merge(actual[KEY + ["y"]], on=KEY, how="left", validate="one_to_one")
    if merged["y"].isna().any():
        raise ValueError(f"{int(merged['y'].isna().sum())} forecast rows have no observed actual")
    return merged


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def smape(y: np.ndarray, p: np.ndarray) -> float:
    """Symmetric MAPE in percent. A pair with y = p = 0 counts as a zero error."""
    den = np.abs(y) + np.abs(p)
    ratio = np.where(den == 0, 0.0, 2 * np.abs(y - p) / np.where(den == 0, 1, den))
    return float(100 * ratio.mean())


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    """Point and quantile metrics for one aligned table. The point forecast is the 0.5 quantile."""
    qs = quantile_cols(frame)
    if not qs or 0.5 not in [q for q, _ in qs]:
        raise ValueError("the table needs a q0.5 column")
    y = frame["y"].to_numpy(dtype=float)
    p = frame[qcol(0.5)].to_numpy(dtype=float)
    err = y - p
    total = np.abs(y).sum()
    lo, hi = frame[qs[0][1]].to_numpy(), frame[qs[-1][1]].to_numpy()
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "wape": float(np.abs(err).sum() / total) if total else float("nan"),
        "smape": smape(y, p),
        "bias": float((p - y).mean()),
        "pinball": float(np.mean([pinball(y, frame[c].to_numpy(), q) for q, c in qs])),
        "coverage": float(((y >= lo) & (y <= hi)).mean()),
        "nominal_coverage": float(qs[-1][0] - qs[0][0]),
    }


def wape_ci(frame: pd.DataFrame, n_boot: int = 300, seed: int = 0, alpha: float = 0.1) -> tuple[float, float]:
    """Bootstrap interval for WAPE. It resamples whole target dates, so days stay together."""
    g = frame.assign(ae=(frame["y"] - frame[qcol(0.5)]).abs(), ay=frame["y"].abs())
    daily = g.groupby("target_date")[["ae", "ay"]].sum().to_numpy()
    if len(daily) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    s = daily[idx].sum(axis=1)
    vals = s[:, 0] / np.where(s[:, 1] == 0, np.nan, s[:, 1])
    return (float(np.nanquantile(vals, alpha / 2)), float(np.nanquantile(vals, 1 - alpha / 2)))

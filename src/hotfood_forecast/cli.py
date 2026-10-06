"""Command line: ``hotfood <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import __version__, synthetic
from .backtest import run_backtest
from .calendar import get_calendar
from .config import Settings
from .data import DataValidationError, load_sales
from .forecast import forecast, orders_for_date
from .models import BASELINES, MODELS
from .newsvendor import category_costs

DEFAULT_MODELS = "seasonal_naive,weekday_mean,ridge,gbm"


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _load(args, settings: Settings):
    cal = get_calendar(settings.holidays)
    sales = load_sales(args.data or settings.data_path, holiday_flags_fn=cal.flags)
    return sales, cal


def cmd_synth(args, settings):
    path = synthetic.write(args.out, n_days=args.days, seed=args.seed if args.seed is not None else settings.seed)
    print(f"wrote {args.days} synthetic days to {path}")


def cmd_validate(args, settings):
    sales, _ = _load(args, settings)
    print("\n".join(sales.report.lines()))
    print("result: OK")


def cmd_backtest(args, settings):
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; use {sorted(MODELS)}")
    if not any(m in BASELINES for m in names):
        print("note: no baseline in --models; add seasonal_naive to compare against a trivial forecast")
    sales, cal = _load(args, settings)
    costs = category_costs(settings, sales)
    res = run_backtest(sales, names, settings, cal, costs, n_origins=args.origins, step=args.step)
    print(f"origins: {len(res.origins)} ({res.origins[0].date()} .. {res.origins[-1].date()}), "
          f"horizon 1..{settings.horizon}, closed and missing days excluded")
    print("\n== accuracy (all categories, all horizons) ==")
    _print(res.metrics[["model", "n", "mae", "rmse", "wape", "wape_ci_low", "wape_ci_high", "smape",
                        "pinball", "coverage", "nominal_coverage", "bias"]])
    print("\n== order policies (waste and stock-outs on the backtest days) ==")
    _print(res.policies)
    out = Path(args.out) if args.out else settings.output_dir / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    res.forecasts.to_csv(out / "forecasts.csv", index=False)
    res.metrics.to_csv(out / "metrics.csv", index=False)
    res.by_category.to_csv(out / "metrics_by_category.csv", index=False)
    res.by_horizon.to_csv(out / "metrics_by_horizon.csv", index=False)
    res.policies.to_csv(out / "policies.csv", index=False)
    summary = {"origins": [str(o.date()) for o in res.origins], "settings": {
        "horizon": settings.horizon, "quantiles": settings.quantiles, "seed": settings.seed,
        "holidays": settings.holidays}, "metrics": res.metrics.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote reports to {out}")


def cmd_forecast(args, settings):
    sales, cal = _load(args, settings)
    costs = category_costs(settings, sales)
    fc = forecast(sales, args.model or settings.model, settings, cal, costs, horizon=args.horizon)
    fc["target_date"] = fc["target_date"].dt.date
    fc = fc.drop(columns=["origin"])
    _print(fc)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fc.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


def cmd_order(args, settings):
    sales, cal = _load(args, settings)
    costs = category_costs(settings, sales)
    fc = orders_for_date(sales, args.date, args.model or settings.model, settings, cal, costs)
    for c, cc in costs.items():
        print(f"{c}: price {cc.price:.2f}, unit cost {cc.unit_cost:.2f}, disposal {cc.disposal_cost:.2f}, "
              f"critical ratio {cc.critical_ratio:.3f}")
    fc["target_date"] = fc["target_date"].dt.date
    _print(fc.drop(columns=["origin"]))


def cmd_demo(args, settings):
    out_dir = Path(args.out_dir) if args.out_dir else settings.output_dir / "demo"
    path = out_dir / "synthetic_sales.csv"
    synthetic.write(path, n_days=730, seed=settings.seed)
    print(f"wrote synthetic data to {path}\n")
    ns = argparse.Namespace(data=path, models=DEFAULT_MODELS, origins=8, step=14, out=str(out_dir / "backtest"))
    cmd_backtest(ns, settings)
    print("\n== 7-day forecast and orders (gbm) ==")
    cmd_forecast(argparse.Namespace(data=path, model="gbm", horizon=7, out=None), settings)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hotfood", description="Hot-food demand forecast and order quantities.")
    p.add_argument("--version", action="version", version=f"hotfood-forecast {__version__}")
    p.add_argument("--horizon", type=int, help="forecast horizon in days (default HOTFOOD_HORIZON or 7)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write a synthetic sales file")
    s.add_argument("--out", default="data/synthetic_sales.csv")
    s.add_argument("--days", type=int, default=730)
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("validate", help="validate a sales file and show the report")
    s.add_argument("--data")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("backtest", help="rolling-origin backtest of models and order policies")
    s.add_argument("--data")
    s.add_argument("--models", default=DEFAULT_MODELS)
    s.add_argument("--origins", type=int, default=12)
    s.add_argument("--step", type=int, default=7)
    s.add_argument("--out", help="folder for the CSV and JSON reports (default <HOTFOOD_OUTPUT_DIR>/backtest)")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("forecast", help="quantile forecast and orders for the next days")
    s.add_argument("--data")
    s.add_argument("--model", choices=sorted(MODELS))
    s.add_argument("--out")
    s.set_defaults(func=cmd_forecast)

    s = sub.add_parser("order", help="order quantities for one future date")
    s.add_argument("--date", required=True)
    s.add_argument("--data")
    s.add_argument("--model", choices=sorted(MODELS))
    s.set_defaults(func=cmd_order)

    s = sub.add_parser("demo", help="offline demo on synthetic data")
    s.add_argument("--out-dir", help="default <HOTFOOD_OUTPUT_DIR>/demo")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.horizon:
            settings = replace(settings, horizon=args.horizon)
        if not hasattr(args, "horizon") or args.horizon is None:
            args.horizon = settings.horizon
        args.func(args, settings)
    except (DataValidationError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, DataValidationError):
            print("\n".join(exc.report.lines()), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

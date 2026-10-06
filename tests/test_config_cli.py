"""Configuration and the command line (problem 9: a reproducible run from start to end)."""
import pandas as pd
import pytest

from hotfood_forecast.calendar import get_calendar
from hotfood_forecast.cli import main
from hotfood_forecast.config import Settings


def test_settings_from_env():
    s = Settings.from_env({
        "HOTFOOD_HORIZON": "3", "HOTFOOD_QUANTILES": "0.9,0.5,0.1", "HOTFOOD_UNIT_COST": "Chicken=0.8",
        "HOTFOOD_SEED": "9", "HOTFOOD_THREADS": "2",
    })
    assert s.horizon == 3 and s.quantiles == (0.1, 0.5, 0.9) and s.unit_cost == {"Chicken": 0.8}
    assert s.seed == 9 and s.threads == 2


@pytest.mark.parametrize("env", [
    {"HOTFOOD_HORIZON": "0"},
    {"HOTFOOD_QUANTILES": "0.1,0.9"},
    {"HOTFOOD_UNIT_COST": "Pizza=1"},
    {"HOTFOOD_COST_RATIO": "1.5"},
])
def test_bad_settings_are_rejected(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_holiday_calendars():
    days = pd.date_range("2024-07-03", "2024-07-05")
    assert get_calendar("us_federal").flags(days).tolist() == [0, 1, 0]
    assert get_calendar("none").flags(days).sum() == 0
    with pytest.raises(ValueError):
        get_calendar("lunar")


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOTFOOD_QUANTILES", "0.1,0.5,0.9")
    data = tmp_path / "s.csv"
    assert main(["synth", "--out", str(data), "--days", "200", "--seed", "1"]) == 0
    assert main(["validate", "--data", str(data)]) == 0
    assert "result: OK" in capsys.readouterr().out
    out_dir = tmp_path / "bt"
    assert main(["--horizon", "3", "backtest", "--data", str(data), "--models", "seasonal_naive,ridge",
                 "--origins", "3", "--out", str(out_dir)]) == 0
    assert (out_dir / "metrics.csv").exists() and (out_dir / "summary.json").exists()
    fc = tmp_path / "fc.csv"
    assert main(["--horizon", "2", "forecast", "--data", str(data), "--model", "ridge", "--out", str(fc)]) == 0
    assert len(pd.read_csv(fc)) == 8
    last = pd.to_datetime(pd.read_csv(data)["Date"], format="%m/%d/%Y").max()
    day = (last + pd.Timedelta(days=1)).date().isoformat()
    assert main(["order", "--data", str(data), "--date", day, "--model", "weekday_mean"]) == 0
    assert "critical ratio" in capsys.readouterr().out


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["validate", "--data", str(tmp_path / "missing.csv")]) == 2
    assert "hotfood synth" in capsys.readouterr().err


def test_dotenv_is_read_and_environment_wins(tmp_path, monkeypatch):
    from hotfood_forecast.config import read_dotenv

    (tmp_path / ".env").write_text("# comment\nHOTFOOD_HORIZON=5\nHOTFOOD_SEED=\n", encoding="utf-8")
    assert read_dotenv(tmp_path / ".env") == {"HOTFOOD_HORIZON": "5"}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HOTFOOD_HORIZON", raising=False)
    assert Settings.from_env().horizon == 5
    monkeypatch.setenv("HOTFOOD_HORIZON", "2")
    assert Settings.from_env().horizon == 2

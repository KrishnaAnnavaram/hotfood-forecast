import pytest

from hotfood_forecast import synthetic
from hotfood_forecast.calendar import get_calendar
from hotfood_forecast.data import validate_frame


@pytest.fixture(scope="session")
def calendar():
    return get_calendar("us_federal")


@pytest.fixture(scope="session")
def raw():
    return synthetic.generate(n_days=240, start="2023-01-02", seed=7, closure_rate=0.02)


@pytest.fixture(scope="session")
def sales(raw, calendar):
    return validate_frame(raw)

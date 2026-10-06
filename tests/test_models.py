"""Problem 5 (preprocessing fit on all rows) and baselines from problem 7."""
import numpy as np
import pytest

from hotfood_forecast.features import FEATURES, training_rows
from hotfood_forecast.models import BASELINES, DropConstant, make_model, sort_quantiles

Q = (0.1, 0.5, 0.9)


@pytest.fixture(scope="module")
def rows(sales, calendar):
    return training_rows(sales, "Chicken", range(1, 8), calendar)


def test_drop_constant_uses_training_rows_only():
    X_train = np.array([[1.0, 5.0, np.nan], [1.0, 6.0, np.nan], [1.0, 7.0, np.nan]])
    t = DropConstant().fit(X_train)
    assert t.keep_.tolist() == [False, True, False]
    assert t.transform(np.array([[9.0, 1.0, 3.0]])).tolist() == [[1.0]]


def test_sort_quantiles_removes_crossing_and_negatives():
    out = sort_quantiles(np.array([[5.0, 3.0, -1.0]]))
    assert out.tolist() == [[0.0, 3.0, 5.0]]


@pytest.mark.parametrize("name", ["seasonal_naive", "weekday_mean", "ridge", "gbm"])
def test_models_give_ordered_nonnegative_quantiles(name, rows):
    m = make_model(name, Q, seed=1).fit(rows)
    pred = m.predict(rows.tail(50))
    assert pred.shape == (50, 3)
    assert (pred >= 0).all()
    assert (np.diff(pred, axis=1) >= 0).all()


def test_ridge_scaler_is_fit_on_training_rows_only(rows):
    train = rows[rows["target_date"] <= rows["target_date"].quantile(0.5)]
    m = make_model("ridge", Q).fit(train)
    pipe = m.pipeline_
    kept = np.array(FEATURES)[pipe.named_steps["drop_constant"].keep_]
    imputed = pipe.named_steps["impute"].transform(pipe.named_steps["drop_constant"].transform(train[list(FEATURES)]))
    assert np.allclose(pipe.named_steps["scale"].mean_, imputed.mean(axis=0))
    assert len(kept) == len(pipe.named_steps["scale"].mean_)


def test_gbm_is_deterministic_with_a_seed(rows):
    a = make_model("gbm", Q, seed=5).fit(rows).predict(rows.tail(20))
    b = make_model("gbm", Q, seed=5).fit(rows).predict(rows.tail(20))
    assert np.array_equal(a, b)


def test_baselines_are_registered():
    assert set(BASELINES) == {"seasonal_naive", "weekday_mean"}


def test_unknown_model_is_an_error():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("lstm", Q)


def test_lightgbm_optional(rows):
    pytest.importorskip("lightgbm")
    pred = make_model("lightgbm", Q).fit(rows).predict(rows.tail(10))
    assert pred.shape == (10, 3)

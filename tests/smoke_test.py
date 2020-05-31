from fireTS.models import NARX, DirectAutoRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

has_xgboost = True
try:
    from xgboost import XGBRegressor
except ImportError:
    has_xgboost = False


def test_NARX():
    x = np.random.randn(100, 1)
    y = np.random.randn(100)
    mdl = NARX(RandomForestRegressor(), auto_order=2, exog_order=[2])
    mdl.fit(x, y)
    ypred = mdl.predict(x, y, step=3)
    assert len(ypred) == 100

    x = np.random.randn(100, 1)
    y = np.random.randn(100)
    mdl = NARX(RandomForestRegressor(), auto_order=1, exog_order=[1])
    mdl.fit(x, y)
    ypred = mdl.predict(x, y, step=3)
    assert len(ypred) == 100

def test_forecast():
    x = np.random.randn(100, 1)
    y = np.random.randn(100)
    mdl = NARX(LinearRegression(), auto_order=2, exog_order=[2])
    mdl.fit(x, y)
    y_forecast = mdl.forecast(x, y, step=10, X_future=np.random.randn(9, 1))
    assert len(y_forecast) == 10

def test_direct():
    x = np.random.randn(100, 1)
    y = np.random.randn(100)
    mdl = DirectAutoRegressor(
        LinearRegression(), auto_order=2, exog_order=[1], pred_step=3)
    mdl.fit(x, y)
    ypred = mdl.predict(x, y)
    assert len(ypred) == 100


def test_readme_examples():
    # Random training data
    x = np.random.randn(100, 2)
    y = np.random.randn(100)

    # Build a non-linear autoregression model with exogenous inputs
    # using Random Forest regression as the base model
    mdl1 = NARX(
        RandomForestRegressor(n_estimators=10),
        auto_order=2,
        exog_order=[2, 2],
        exog_delay=[1, 1])
    mdl1.fit(x, y)
    ypred1 = mdl1.predict(x, y, step=3)
    assert len(ypred1) == 100

    if has_xgboost:
        # Build a general autoregression model and make multi-step prediction
        # directly using XGBRegressor as the base model
        mdl2 = DirectAutoRegressor(
            XGBRegressor(n_estimators=10),
            auto_order=2,
            exog_order=[2, 2],
            exog_delay=[1, 1],
            pred_step=3)
        mdl2.fit(x, y)
        ypred2 = mdl2.predict(x, y)
        assert len(ypred2) == 100

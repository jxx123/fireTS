from fireTS.models import NARX, DirectAutoRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


def test_forecast_and_predict_consistency():
    np.random.seed(0)
    x = np.random.randn(10, 1)
    y = np.random.randn(10)
    mdl = NARX(LinearRegression(), auto_order=2, exog_order=[2])
    mdl.fit(x, y)

    # 1-step
    ypred = mdl.predict(x, y, step=1)
    yforecast = mdl.forecast(x[:-1, :], y[:-1], step=1)
    np.testing.assert_almost_equal(ypred[-1], yforecast[-1])

    # 2-step
    ypred = mdl.predict(x, y, step=2)
    X_future = x[-2:-1, :]
    #  print(X_future)
    yforecast = mdl.forecast(x[:-2, :], y[:-2], step=2, X_future=X_future)
    np.testing.assert_almost_equal(ypred[-1], yforecast[-1])

    # 3-step
    ypred = mdl.predict(x, y, step=3)
    X_future = x[-3:-1, :]
    yforecast = mdl.forecast(x[:-3, :], y[:-3], step=3, X_future=X_future)
    np.testing.assert_almost_equal(ypred[-1], yforecast[-1])


if __name__ == '__main__':
    test_forecast_and_predict_consistency()

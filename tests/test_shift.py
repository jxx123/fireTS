from fireTS.utils import shift, create_lag_features
from fireTS.core import NARX
from sklearn.linear_model import LinearRegression
import numpy as np
from nose.tools import *


def test_shift():
    x = np.array([1., 2., 3., 4.])
    x1 = shift(x, 2)
    np.testing.assert_array_equal(x1, np.array([np.nan, np.nan, 1, 2]))

    x2 = shift(x, -2)
    np.testing.assert_array_equal(x2, np.array([3, 4, np.nan, np.nan]))

    x3 = shift(x, 6)
    np.testing.assert_array_equal(x3,
                                  np.array([np.nan, np.nan, np.nan, np.nan]))

    x4 = shift(x, -6)
    np.testing.assert_array_equal(x4,
                                  np.array([np.nan, np.nan, np.nan, np.nan]))

    x5 = shift(x, 0)
    np.testing.assert_array_equal(x5, x)

    x6 = shift(x, 4)
    np.testing.assert_array_equal(x6,
                                  np.array([np.nan, np.nan, np.nan, np.nan]))

    x7 = shift(x, 4)
    np.testing.assert_array_equal(x7,
                                  np.array([np.nan, np.nan, np.nan, np.nan]))

    y = np.array([[1., 2., 3.], [4., 5., 6.]])
    y1 = shift(y, -1)
    np.testing.assert_array_equal(y1,
                                  np.array([[4., 5., 6.],
                                            [np.nan, np.nan, np.nan]]))

    y = np.array([[1., 2., 3.], [4., 5., 6.]])
    y1 = shift(y, 1)
    np.testing.assert_array_equal(y1,
                                  np.array([[np.nan, np.nan, np.nan],
                                            [1., 2., 3.]]))


def test_create_lag_features():
    x = np.array([1., 2., 3., 4.])

    lag1 = create_lag_features(x, 2)
    exp1 = np.array([[1., 2., 3., 4.], [np.nan, 1., 2., 3.]]).T
    np.testing.assert_array_equal(lag1, exp1)

    lag2 = create_lag_features(x, 3)
    exp2 = np.array([[1., 2., 3., 4.], [np.nan, 1., 2., 3.],
                     [np.nan, np.nan, 1., 2.]]).T
    np.testing.assert_array_equal(lag2, exp2)


def test_preprocess_data():
    estimator = NARX(
        LinearRegression(), auto_order=2, exog_order=[2, 3], exog_delay=[1, 2])

    X = np.array([[1., 3.], [2., 7.], [4., 6.], [3., 8.], [5., 5.], [2.5, 4.5],
                  [3., 3.8]])
    y = np.array([1., 5., 7., 4., 6., 3., 2.])
    features, target = estimator._preprocess_data(y, X=X)

    y_exp = np.array([3., 2.])
    X_exp = np.array([[6., 4., 3., 4., 6., 7., 3.],
                      [3., 6., 5., 3., 8., 6., 7.]])

    np.testing.assert_array_equal(target, y_exp)
    np.testing.assert_array_equal(features, X_exp)

    estimator.fit(X, y)


def test_update_lag_features():
    regressor = NARX(
        LinearRegression(), auto_order=2, exog_order=[2, 3], exog_delay=[1, 2])

    Xdata = np.array([[4., 7., 3., 4., 6., 7.,
                       3.], [6., 4., 5., 3., 8., 6., 7.],
                      [3., 6., 2.5, 5., 5., 8., 6.]])
    ypred = np.array([6., 3., 2.])

    X_updated = regressor._update_lag_features(Xdata, ypred)

    X_exp = np.array([[6., 4., 5., 3., 8., 6.,
                       7.], [3., 6., 2.5, 5., 5., 8., 6.],
                      [2., 3., np.nan, 2.5, np.nan, 5., 8.]])
    np.testing.assert_array_equal(X_updated, X_exp)


if __name__ == "__main__":
    # test_preprocess_data()
    test_update_lag_features()

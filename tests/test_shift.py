from fireTS.utils import shift, LagFeatureProcessor, MetaLagFeatureProcessor
from fireTS.models import NARX
from sklearn.linear_model import LinearRegression
import numpy as np


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

    p = LagFeatureProcessor(x, 2, 0)
    lag1 = p.generate_lag_features()
    exp1 = np.array([[1., 2., 3., 4.], [np.nan, 1., 2., 3.]]).T
    np.testing.assert_array_equal(lag1, exp1)

    p = LagFeatureProcessor(x, 3, 0)
    lag2 = p.generate_lag_features()
    exp2 = np.array([[1., 2., 3., 4.], [np.nan, 1., 2., 3.],
                     [np.nan, np.nan, 1., 2.]]).T
    np.testing.assert_array_equal(lag2, exp2)


def test_preprocess_data():
    estimator = NARX(
        LinearRegression(), auto_order=2, exog_order=[2, 3], exog_delay=[1, 2])

    X = np.array([[1., 3.], [2., 7.], [4., 6.], [3., 8.], [5., 5.], [2.5, 4.5],
                  [3., 3.8]])
    y = np.array([1., 5., 7., 4., 6., 3., 2.])
    features, target = estimator._preprocess_data(X, y)

    y_exp = np.array([3., 2.])
    X_exp = np.array([[6., 4., 3., 4., 6., 7., 3.],
                      [3., 6., 5., 3., 8., 6., 7.]])

    np.testing.assert_array_equal(target, y_exp)
    np.testing.assert_array_equal(features, X_exp)

    estimator.fit(X, y)


def test_update_lag_features():
    X = np.array([[1., 3.], [2., 7.], [4., 6.], [3., 8.], [5., 5.], [2.5, 4.5],
                  [3., 3.8]])
    y = np.array([1., 5., 7., 4., 6., 3., 2.])
    ypred = np.array([5., 7., 4., 6., 3., 2., 1.])

    data = [y]
    data.extend(X.T)

    p = MetaLagFeatureProcessor(data, [2, 2, 3], [0, 1, 2])
    data_new = [ypred]
    data_new.extend([None] * 2)

    X_updated = p.update(data_new)

    X_exp = np.array([[5., 1., 1., np.nan, np.nan, np.nan, np.nan], [
        7., 5., 2., 1., 3., np.nan, np.nan
    ], [4., 7., 4., 2., 7., 3., np.nan], [6., 4., 3., 4., 6., 7.,
                                          3.], [3., 6., 5., 3., 8., 6., 7.],
                      [2., 3., 2.5, 5., 5., 8.,
                       6.], [1., 2., 3., 2.5, 4.5, 5., 8.]])
    np.testing.assert_array_equal(X_updated, X_exp)


if __name__ == "__main__":
    # test_preprocess_data()
    # test_update_lag_features()
    test_update_lag_features()

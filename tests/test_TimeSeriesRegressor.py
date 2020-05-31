import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from fireTS.models import NARX
from fireTS.utils import shift
import numpy as np
import pytest
import copy


@pytest.mark.parametrize('na', [1, 2, 3, 8])
@pytest.mark.parametrize('nb', [[6, 6], [1, 1], [2, 3]])
@pytest.mark.parametrize('nk', [[1, 1], [1, 2], [4, 5]])
def test_TimeSeriesRegressor_create_features(na, nb, nk):
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(100, 2))
    y = pd.Series(np.random.randn(100))
    mdl = NARX(LinearRegression(), auto_order=na, exog_order=nb, exog_delay=nk)

    Xfeatures_act, ytarget_act = mdl._preprocess_data(X.values, y.values)

    Xfeatures_exp, ytarget_exp = helper_preprocess(X, y, na, nb, nk)

    np.testing.assert_array_equal(Xfeatures_act, Xfeatures_exp)
    np.testing.assert_array_equal(ytarget_act, ytarget_exp)


def helper_preprocess(X,
                      y,
                      auto_order,
                      exog_order,
                      exog_delay,
                      pred_step=1,
                      removeNA=True):
    target = y.shift(-pred_step)
    predictor = pd.DataFrame()
    for lag in range(auto_order):
        predictor[str(y.name) + '_lag%d' % lag] = y.shift(lag)

    for i, (nb, nk) in enumerate(zip(exog_order, exog_delay)):
        for lag in range(nb):
            predictor[str(X.columns[i])
                      + '_lag%d' % (lag + nk)] = X.iloc[:, i].shift(lag + nk)

    if removeNA:
        mask = target.isna() | predictor.isna().any(axis=1)
        return predictor.loc[~mask, :].values, target[~mask].values
    else:
        return predictor.values, target.values


def test_TimeSeriesRegressor_predict():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(100, 2))
    y = pd.Series(np.random.randn(100))
    na = 3
    nb = [3, 3]
    nk = [1, 1]
    step = 2
    mdl = NARX(LinearRegression(), auto_order=na, exog_order=nb, exog_delay=nk)

    mdl.fit(X, y)
    ypred_act = mdl.predict(X, y, step=step)
    mdl.score(X, y, step=step, method="r2")
    mdl.score(X, y, step=step, method="mse")

    # -------- manual computation ---------------
    kernel_mdl = LinearRegression()
    Xfeatures_exp, ytarget_exp = helper_preprocess(
        X, y, na, nb, nk, removeNA=False)
    mask = np.isnan(ytarget_exp) | np.isnan(Xfeatures_exp).any(axis=1)
    kernel_mdl.fit(Xfeatures_exp[~mask, :], ytarget_exp[~mask])

    ypred_exp1 = np.empty(X.shape[0]) * np.nan
    ypred_exp1[~mask] = kernel_mdl.predict(Xfeatures_exp[~mask, :])

    X1 = copy.deepcopy(Xfeatures_exp)
    X2 = copy.deepcopy(Xfeatures_exp)
    # Xfeatures_updated = mdl._update_lag_features(X1, ypred_exp1)

    X2[:, 1:3] = X2[:, 0:2]
    X2[:, 0] = ypred_exp1

    X2[:, 4:6] = X2[:, 3:5]
    X2[:, 3] = shift(X2[:, 3], -1)

    X2[:, 7:9] = X2[:, 6:8]
    X2[:, 6] = shift(X2[:, 6], -1)
    mask = ~np.isnan(X2).any(axis=1)

    ypred_exp2 = np.empty(X2.shape[0]) * np.nan
    ypred_exp2[mask] = kernel_mdl.predict(X2[mask, :])
    ypred_exp2 = np.concatenate([np.empty(2) * np.nan, ypred_exp2])[0:len(y)]

    # print(X2)
    # print(ypred_act)
    np.testing.assert_array_almost_equal(ypred_act, ypred_exp2)


def test_TimeSeriesRegressor_grid_search():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(100, 2))
    y = pd.Series(np.random.randn(100))
    na = 3
    nb = [3, 3]
    nk = [1, 1]
    mdl = NARX(Ridge(), auto_order=na, exog_order=nb, exog_delay=nk)

    para_grid = {'alpha': [0, 0.1, 0.3]}
    mdl.grid_search(X, y, para_grid)


if __name__ == "__main__":
    test_TimeSeriesRegressor_predict()
    # test_TimeSeriesRegressor_create_features(2, [2, 3], [1, 1])
    # pass

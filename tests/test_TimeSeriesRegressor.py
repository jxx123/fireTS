import pandas as pd
from sklearn.linear_model import LinearRegression
from fireTS.core import TimeSeriesRegressor
from fireTS.utils import shift
import numpy as np
import pytest
import copy


@pytest.mark.parametrize('na', [1, 2, 3, 8])
@pytest.mark.parametrize('nb', [[6, 6], [1, 1], [2, 3]])
@pytest.mark.parametrize('nk', [[1, 1], [1, 2], [4, 5]])
def test_TimeSeriesRegressor_create_features(na, nb, nk):
    df = pd.read_csv(
        '~/Documents/DiabetesProject/Glucose prediction/OhioData/OhioT1DM-training/clean3/train/559_train.csv'
    )
    df['insulin'] = df['basal'] + df['bolus'] / 5
    df = df.set_index('ts')
    df.index = pd.to_datetime(df.index)
    selected_features = ['insulin', 'meal']
    X = df[selected_features]
    y = df['bg']
    mdl = TimeSeriesRegressor(
        LinearRegression(), output_order=na, input_order=nb, input_delay=nk)

    Xfeatures_act, ytarget_act = mdl._preprocess_data(X.values, y.values)

    Xfeatures_exp, ytarget_exp = helper_preprocess(X, y, na, nb, nk)

    np.testing.assert_array_equal(Xfeatures_act, Xfeatures_exp)
    np.testing.assert_array_equal(ytarget_act, ytarget_exp)


def helper_preprocess(X, y, auto_order, exog_order, exog_delay, pred_step=1):
    target = y.shift(-(pred_step - 1))
    predictor = pd.DataFrame()
    for lag in range(auto_order):
        predictor[y.name + '_lag%d' % (lag + 1)] = y.shift(lag + 1)

    for i, (nb, nk) in enumerate(zip(exog_order, exog_delay)):
        for lag in range(nb):
            predictor[X.columns[i]
                      + '_lag%d' % (lag + nk)] = X.iloc[:, i].shift(lag + nk)

    mask = target.isna() | predictor.isna().any(axis=1)

    return predictor.loc[~mask, :].values, target[~mask].values


def test_TimeSeriesRegressor_predict():
    df = pd.read_csv(
        '~/Documents/DiabetesProject/Glucose prediction/OhioData/OhioT1DM-training/clean3/train/559_train.csv'
    )
    df['insulin'] = df['basal'] + df['bolus'] / 5
    df = df.set_index('ts')
    df.index = pd.to_datetime(df.index)
    selected_features = ['insulin', 'meal']
    X = df[selected_features]
    y = df['bg']
    na = 3
    nb = [3, 3]
    nk = [1, 1]
    step = 2
    mdl = TimeSeriesRegressor(
        LinearRegression(), output_order=na, input_order=nb, input_delay=nk)

    mdl.fit(X, y)
    ypred_act = mdl.predict(X, y, step=step)

    # -------- manual computation ---------------
    kernel_mdl = LinearRegression()
    Xfeatures_exp, ytarget_exp = helper_preprocess(X, y, na, nb, nk)
    kernel_mdl.fit(Xfeatures_exp, ytarget_exp)
    ypred_exp1 = kernel_mdl.predict(Xfeatures_exp)

    X1 = copy.deepcopy(Xfeatures_exp)
    X2 = copy.deepcopy(Xfeatures_exp)
    Xfeatures_updated = mdl._update_lag_features(X1, ypred_exp1)

    X2[:, 1:3] = X2[:, 0:2]
    X2[:, 0] = ypred_exp1

    X2[:, 4:6] = X2[:, 3:5]
    X2[:, 3] = shift(X2[:, 3], -1)

    X2[:, 7:9] = X2[:, 6:8]
    X2[:, 6] = shift(X2[:, 6], -1)
    mask = ~np.isnan(X2).any(axis=1)

    X2 = X2[mask, :]

    np.testing.assert_array_equal(Xfeatures_updated, X2)

    ypred_exp2 = kernel_mdl.predict(X2)

    np.testing.assert_array_almost_equal(ypred_act.values, ypred_exp2)


if __name__ == "__main__":
    test_TimeSeriesRegressor_predict()

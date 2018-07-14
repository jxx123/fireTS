import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics.regression import r2_score
from fireTS.utils import create_lag_features, shift
from sklearn.model_selection import GridSearchCV
import pandas as pd


class TimeSeriesEstimator(BaseEstimator):
    """
    Base Class for Time Series Estimators. Could be inherited for
    classification or regression.
    """

    def __init__(self,
                 base_estimator,
                 output_order=0,
                 output_ahead=0,
                 input_order=1,
                 input_delay=None,
                 **base_params):
        self.base_estimator = base_estimator.set_params(**base_params)
        self.output_order = output_order
        self.input_order = input_order
        if input_delay is None:
            input_delay = [1] * len(self.input_order)
        self.input_delay = input_delay
        self.output_ahead = output_ahead
        self._X_last = None
        self._Xdata_last = None
        self._ydata_last = None

    def set_params(self, **params):
        for param, value in params.items():
            if param in self.get_params():
                super(TimeSeriesEstimator, self).set_params(**{param: value})
            else:
                self.base_estimator.set_params(**{param: value})
        return self

    def fit(self, X, y, **params):
        X, y = check_X_y(X, y, y_numeric=True)
        Xdata, ydata = self._preprocess_data(X, y)
        self._Xdata_last = Xdata[-1, :]
        self._ydata_last = ydata[-1]
        self._X_last = X[-1, :]
        self.base_estimator.fit(Xdata, ydata, **params)

    def _preprocess_data(self, X, y):
        """
        Generate lag features based on input_order, input_delay, output_order,
        and output_ahead

        Inputs
        ------
        X: feature matrix, shape: N * p, where N is the number of samples, and
            p is the number of features
        y: target vector, shape: N * 1

        Outputs
        -------
        lag_features: lag feature matrix, in form of [y_lag_features,
            X1_lag_features, ..., Xp_lag_features]
            shape:
            (N - max(max(input_delay + input_order) - 1, output_order) + 1 - output_ahead) *
            (output_order + sum(input_order))
        target: target vector shifted by output_ahead, i.e. y(t + output_ahead)
            shape:
            (N - max(max(input_delay + input_order), output_order) + 1 - output_ahead) * 1
        """
        X_lag = self._create_input_lag_features(X)

        y_shift = shift(y, 1)
        y_lag = create_lag_features(y_shift, self.output_order)

        lag_features = np.concatenate([y_lag, X_lag], axis=1)
        target = shift(y, -self.output_ahead)

        all_data = np.concatenate(
            [target.reshape(-1, 1), lag_features], axis=1)
        mask = np.isnan(all_data).any(axis=1)

        return lag_features[~mask], target[~mask]

    def _create_input_lag_features(self, X):
        X_lag = []
        for Xcol, delay, order in zip(X.T, self.input_delay, self.input_order):
            X_shift = shift(Xcol, delay)
            X_lag.append(create_lag_features(X_shift, order))
        X_lag = np.concatenate(X_lag, axis=1)
        return X_lag

    def grid_search(self, X, y, para_grid, **params):
        grid = GridSearchCV(self.base_estimator, para_grid, **params)
        X, y = check_X_y(X, y, y_numeric=True)
        Xdata, ydata = self._preprocess_data(X, y)
        grid.fit(Xdata, ydata)
        self.set_params(**grid.best_params_)


class TimeSeriesRegressor(TimeSeriesEstimator, RegressorMixin):
    def score(self, X, y, step=1):
        ypred = self.predict(X, y, step=step)
        return r2_score(y[ypred.index], ypred)

    def forecast(self, X):
        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            index = list(range(X.shape[0]))

        X = check_array(X)
        ypred = []
        Xdata = self._update_lag_features(
            self._Xdata_last.reshape(1, -1),
            self._ydata_last,
            Xfuture=self._X_last)
        ypred_tmp = self.base_estimator.predict(Xdata)
        ypred.append(ypred_tmp)

        for Xfuture in X:
            Xdata = self._update_lag_features(
                Xdata, ypred_tmp, Xfuture=Xfuture)
            ypred_tmp = self.base_estimator.predict(Xdata)
            ypred.append(ypred_tmp)

        ypred = np.array(ypred[:-1]).flatten()
        ypred_series = pd.Series(ypred, index=index)
        return ypred_series

    def predict(self, X, y, step=1):
        if isinstance(y, pd.Series):
            index = y.index
        else:
            index = list(range(y.size))

        X, y = check_X_y(X, y, y_numeric=True)
        Xdata, ydata = self._preprocess_data(X, y)

        for k in range(step):
            yhat = self.base_estimator.predict(Xdata)
            if k == step - 1:
                break
            Xdata = self._update_lag_features(Xdata, yhat)

        input_lag = np.array(self.input_order) + np.array(self.input_delay)

        # start index takes account of lag features, and the step to predict
        start = max(max(input_lag) - 1,
                    self.output_order) + self.output_ahead + step - 1

        yhat_series = pd.Series(yhat, index=index[start:])
        return yhat_series

    def _update_lag_features(self, Xdata, yhat, Xfuture=None):
        # update y_lag_features
        Xdata[:, 1:self.output_order] = Xdata[:, 0:self.output_order - 1]
        Xdata[:, 0] = yhat

        # update x_lag_features
        for i, n in enumerate(self.input_order):
            start = self.output_order + sum(self.input_order[:i])
            Xdata[:, start + 1:start + n] = Xdata[:, start:start + n - 1]
            Xdata[:, start] = shift(Xdata[:, start], -1)
            if Xfuture is not None:
                Xdata[-1, start] = Xfuture[i]

        mask = np.isnan(Xdata).any(axis=1)
        return Xdata[~mask]

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics.regression import r2_score, mean_squared_error
from fireTS.utils import create_lag_features, shift
from sklearn.model_selection import GridSearchCV
import pandas as pd
import copy


class TimeSeriesRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, **base_params):
        """
        TimeSeriesRegressor creates a time series model based on a
        general-purpose regression model defined in base_estimator.
        base_estimator must be a model which implements the scikit-learn APIs.
        """
        self.base_estimator = base_estimator.set_params(**base_params)

    def set_params(self, **params):
        for param, value in params.items():
            if param in self.get_params():
                super(TimeSeriesRegressor, self).set_params(**{param: value})
            else:
                self.base_estimator.set_params(**{param: value})
        return self


class NARX(TimeSeriesRegressor, RegressorMixin):
    def __init__(self,
                 base_estimator,
                 auto_order=0,
                 pred_step=1,
                 exog_order=1,
                 exog_delay=None,
                 **base_params):
        """
        NARX stands for Nonlinear AutoRegressive eXogenous model
        (https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model).
        This model generalizes the NARX model into the following form:

        y(t + pred_step) = f(y(t), ..., y(t - auto_order + 1), x_1(t -
        exog_delay[0]), ..., x_1(t - exog_delay[0] - exog_order[0] + 1),
        ..., x_m(t - exog_delay[m-1]), ..., x_m(t - exog_delay[m-1] -
        exog_order[m-1] + 1))
        """
        super(NARX, self).__init__(base_estimator, **base_params)
        self.auto_order = auto_order
        self.exog_order = exog_order
        if exog_delay is None:
            exog_delay = [0] * len(self.exog_order)
        self.exog_delay = exog_delay
        self.pred_step = pred_step

    def fit(self, X, y, **params):
        """
        Create lag features and fit the base_estimator.

        Parameters
        ----------
            X : array-like, shape = (n_samples, n_features)
                Exogenous input time series

            y : array-like, shape = (n_samples)
        """
        X, y = check_X_y(X, y, y_numeric=True)
        Xdata, ydata = self._preprocess_data(y, X=X)
        self.base_estimator.fit(Xdata, ydata, **params)

    def _preprocess_data(self, y, X=None, removeNA=True):
        """
        Generate lag features based on exog_order, exog_delay, auto_order,
        and pred_step

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
            (N - max(max(exog_delay + exog_order) - 1, auto_order) + 1 - pred_step) *
            (auto_order + sum(exog_order))
        target: target vector shifted by pred_step, i.e. y(t + pred_step)
            shape:
            (N - max(max(exog_delay + exog_order), auto_order) + 1 - pred_step) * 1
        """
        y_lag = create_lag_features(y, self.auto_order)
        if X is None:
            lag_features = copy.copy(y_lag)
        else:
            X_lag = self._create_input_lag_features(X)
            lag_features = np.concatenate([y_lag, X_lag], axis=1)

        target = shift(y, -self.pred_step)

        if removeNA:
            all_data = np.concatenate(
                [target.reshape(-1, 1), lag_features], axis=1)
            mask = np.isnan(all_data).any(axis=1)
            lag_features, target = lag_features[~mask], target[~mask]

        return lag_features, target

    def _create_input_lag_features(self, X):
        X_lag = []
        for Xcol, delay, order in zip(X.T, self.exog_delay, self.exog_order):
            X_shift = shift(Xcol, delay)
            X_lag.append(create_lag_features(X_shift, order))
        X_lag = np.concatenate(X_lag, axis=1)
        return X_lag

    def grid_search(self, X, y, para_grid, **params):
        grid = GridSearchCV(self.base_estimator, para_grid, **params)
        X, y = check_X_y(X, y, y_numeric=True)
        Xdata, ydata = self._preprocess_data(y, X=X)
        grid.fit(Xdata, ydata)
        self.set_params(**grid.best_params_)

    def score(self, X, y, step=1, method="r2"):
        ypred = self.predict(X, y, step=step)
        mask = np.isnan(y) | np.isnan(ypred)
        if method == "r2":
            return r2_score(y[~mask], ypred[~mask])
        elif method == "mse":
            return mean_squared_error(y[~mask], ypred[~mask])

    # TODO: add forecast method
    def predict(self, X, y, step=None):
        if step is None:
            step = 1

        if self.pred_step > 1 and step > 1:
            raise ValueError(
                "Cannot perform multi-step prediction when pred_step is larger than 1."
            )

        Xchk, ychk = check_X_y(X, y, y_numeric=True)
        Xdata, _ = self._preprocess_data(ychk, X=Xchk, removeNA=False)

        for k in range(step):
            yhat = self._predictNA(Xdata)
            if k == step - 1:
                break
            Xdata = self._update_lag_features(Xdata, yhat)

        pred_step = max(self.pred_step, step)
        ypred = np.concatenate([np.empty(pred_step) * np.nan, yhat])[0:len(y)]

        if isinstance(y, pd.Series):
            ypred = pd.Series(ypred, index=y.index)

        return ypred

    def _predictNA(self, Xdata):
        ypred = np.empty(Xdata.shape[0]) * np.nan
        mask = np.isnan(Xdata).any(axis=1)
        X2pred = Xdata[~mask]
        ypred[~mask] = self.base_estimator.predict(X2pred)
        return ypred

    def _update_lag_features(self, Xdata, yhat):
        # update y_lag_features
        Xdata[:, 1:self.auto_order] = Xdata[:, 0:self.auto_order - 1]
        Xdata[:, 0] = yhat

        # update x_lag_features
        for i, n in enumerate(self.exog_order):
            start = self.auto_order + sum(self.exog_order[:i])
            Xdata[:, start + 1:start + n] = Xdata[:, start:start + n - 1]
            Xdata[:, start] = shift(Xdata[:, start], -1)
        return Xdata

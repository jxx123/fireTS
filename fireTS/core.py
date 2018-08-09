import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from fireTS.utils import shift, MetaLagFeatureProcessor
from sklearn.model_selection import GridSearchCV


class TimeSeriesRegressor(BaseEstimator, RegressorMixin):
    """
    TimeSeriesRegressor creates a time series model based on a
    general-purpose regression model defined in base_estimator.
    base_estimator must be a model which implements the scikit-learn APIs.
    """

    def __init__(self, base_estimator, **base_params):
        self.base_estimator = base_estimator.set_params(**base_params)

    def set_params(self, **params):
        for param, value in params.items():
            if param in self.get_params():
                super(TimeSeriesRegressor, self).set_params(**{param: value})
            else:
                self.base_estimator.set_params(**{param: value})
        return self


class GeneralAutoRegressor(TimeSeriesRegressor, RegressorMixin):
    r"""
    The general auto regression model can be written in the following form:

    .. math::
        y(t + k) &=& f(y(t), ..., y(t-p+1), \\
                 & & x_1(t - d_1), ..., x_1(t-d_1-q_1+1), \\
                 & & ..., x_m(t - d_1), ..., x_m(t - d_m - q_m + 1)) + e(t)
        :label: gar

    :param object base_estimator: an estimator object that implements the
                                  scikit-learn API (fit, and predict). The
                                  estimator will be used to fit the function
                                  :math:`f` in equation :eq:`gar`.
    :param int auto_order: the autoregression order :math:`p` in equation
                           :eq:`gar`.
    :param list exog_order: the exogenous input order, a list of integers
                            representing the order for each exogenous input,
                            i.e. :math:`[q_1, q_2, ..., q_m]` in equation
                            :eq:`gar`.
    :param list exog_delay: the delays of the exogenous inputs, a list of
                            integers representing the delay of each exogenous
                            input, i.e. :math:`[d_1, d_2, ..., d_m]` in
                            equation :eq:`gar`. By default, all the delays are
                            set to 0.
    :param int pred_step: the prediction step :math:`k` in equation :eq:`gar`.
                          By default, it is set to 1.
    :param dict base_params: other keyword arguments for base_estimator.
    """

    def __init__(self,
                 base_estimator,
                 auto_order,
                 exog_order,
                 exog_delay=None,
                 pred_step=1,
                 **base_params):
        super(GeneralAutoRegressor, self).__init__(base_estimator,
                                                   **base_params)
        self.auto_order = auto_order
        self.exog_order = exog_order
        if exog_delay is None:
            exog_delay = [0] * len(exog_order)
        if len(exog_delay) != len(exog_order):
            raise ValueError(
                'The length of exog_delay must be the same as the length of exog_order.'
            )
        self.exog_delay = exog_delay
        self.pred_step = pred_step

    def fit(self, X, y, **params):
        """
        Create lag features and fit the base_estimator.

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        """
        X, y = check_X_y(X, y, y_numeric=True)
        if len(self.exog_order) != X.shape[1]:
            raise ValueError(
                'The number of columns of X must be the same as the length of exog_order.'
            )
        features, target = self._preprocess_data(X, y)
        self.base_estimator.fit(features, target, **params)

    def _preprocess_data(self, X, y):
        p = self._get_lag_feature_processor(X, y)
        features = p.generate_lag_features()
        target = shift(y, -self.pred_step)

        # Remove NaN
        all_data = np.concatenate([target.reshape(-1, 1), features], axis=1)
        mask = np.isnan(all_data).any(axis=1)
        features, target = features[~mask], target[~mask]

        return features, target

    def _get_lag_feature_processor(self, X, y):
        data = [y]
        orders = [self.auto_order]
        delays = [0]
        data.extend(X.T)
        orders.extend(self.exog_order)
        delays.extend(self.exog_delay)
        return MetaLagFeatureProcessor(data, orders, delays)

    def grid_search(self, X, y, para_grid, **params):
        """
        Perform grid search on the base_estimator. The function first generates
        the lag features and predicting targets, and then calls
        ``GridSearchCV`` in scikit-learn package.

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        :param dict para_grid: use the same format in ``GridSearchCV`` in
                               scikit-learn package.
        :param dict params: other keyword arguments that can be passed into
                            ``GridSearchCV`` in scikit-learn package.
        """
        grid = GridSearchCV(self.base_estimator, para_grid, **params)
        X, y = check_X_y(X, y, y_numeric=True)
        features, target = self._preprocess_data(X, y)
        grid.fit(features, target)
        self.set_params(**grid.best_params_)

    def _predictNA(self, Xdata):
        ypred = np.empty(Xdata.shape[0]) * np.nan
        mask = np.isnan(Xdata).any(axis=1)
        X2pred = Xdata[~mask]
        ypred[~mask] = self.base_estimator.predict(X2pred)
        return ypred

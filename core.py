from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.linear_model import LinearRegression
import numpy as np


class TimeSeriesEstimator(BaseEstimator):
    def __init__(self, base_estimator, output_order=0, input_order=1,
                 input_delay=None, n_ahead=1, **base_params):
        self.base_estimator = base_estimator.set_params(**base_params)
        self.output_order = output_order
        self.input_order = input_order
        if input_delay is None:
            input_delay = np.ones(np.size(input_order))
        self.input_delay = input_delay
        self.n_ahead = n_ahead

    def fit(self):
        pass


class TimeSeriesRegressor(RegressorMixin, TimeSeriesEstimator):
    def __init__(self):
        pass

import numpy as np
from collections import deque


def shift(darray, k, axis=0):
    """
    Utility function to shift a numpy array

    Inputs
    ------
    darray: a numpy array
        the array to be shifted.
    k: integer
        number of shift
    axis: non-negative integer
        axis to perform shift operation

    Outputs
    -------
    shifted numpy array, fill the unknown values with nan
    """
    if k == 0:
        return darray
    elif k < 0:
        shift_array = np.roll(darray, k, axis=axis)
        shift_array[k:] = np.nan
        return shift_array
    else:
        shift_array = np.roll(darray, k, axis=axis)
        shift_array[:k] = np.nan
        return shift_array


class OutputLagFeatureProcessor:
    def __init__(self, data, order):
        self._feature_queue = deque([shift(data, l) for l in range(order)])

    def generate_lag_features(self):
        return np.array(self._feature_queue).T

    def update(self, data_new):
        # TODO: this is not memory efficient, need to do this in a
        # better way in the future
        self._feature_queue.appendleft(data_new)
        self._feature_queue.pop()
        return np.array(self._feature_queue).T


class InputLagFeatureProcessor:
    def __init__(self, data, order, delay):
        self._data = data
        self._lags = np.array(range(delay, delay + order))

    def generate_lag_features(self):
        features = [shift(self._data, l) for l in self._lags]
        return np.array(features).T

    def update(self):
        self._lags = self._lags - 1
        return self.generate_lag_features()


class MetaLagFeatureProcessor(object):
    def __init__(self, X, y, auto_order, exog_order, exog_delay):
        self._lag_feature_processors = [
            OutputLagFeatureProcessor(y, auto_order)
        ]
        self._lag_feature_processors.extend([
            InputLagFeatureProcessor(data, order, delay)
            for data, order, delay in zip(X.T, exog_order, exog_delay)
        ])

    def generate_lag_features(self):
        lag_feature_list = [
            p.generate_lag_features() for p in self._lag_feature_processors
        ]
        lag_features = np.concatenate(lag_feature_list, axis=1)
        return lag_features

    def update(self, data_new):
        lag_feature_list = [
            p.update(data_new) if i == 0 else p.update()
            for i, p in enumerate(self._lag_feature_processors)
        ]
        lag_features = np.concatenate(lag_feature_list, axis=1)
        return lag_features

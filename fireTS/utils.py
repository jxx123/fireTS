import numpy as np


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


class LagFeatureProcessor(object):
    def __init__(self, data, order, delay):
        self._data = data
        self._lags = np.array(range(delay, delay + order))

    def generate_lag_features(self, data_new=None):
        features = [shift(self._data, l) for l in self._lags]
        if data_new is not None:
            features[0] = data_new
        return np.array(features).T

    def update(self, data_new=None):
        self._lags = self._lags - 1
        return self.generate_lag_features(data_new=data_new)


class MetaLagFeatureProcessor(object):
    def __init__(self, data, orders, delays):
        self._lag_feature_processors = [
            LagFeatureProcessor(d, order, delay)
            for d, order, delay in zip(data, orders, delays)
        ]

    def generate_lag_features(self):
        lag_feature_list = [
            p.generate_lag_features() for p in self._lag_feature_processors
        ]
        lag_features = np.concatenate(lag_feature_list, axis=1)
        return lag_features

    def update(self, data_new):
        lag_feature_list = [
            p.update(data_new=d)
            for d, p in zip(data_new, self._lag_feature_processors)
        ]
        lag_features = np.concatenate(lag_feature_list, axis=1)
        return lag_features

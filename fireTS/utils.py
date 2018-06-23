import numpy as np


def create_lag_features(data, num_lag):
    """
    Utility function to create lag features

    Inputs
    ------
    data: 1-d numpy array
        data to derive lag features
    num_lag: positive integer
        number of lag features to create

    Outputs
    -------
    lag_features: a 2-d numpy array, with number of rows = input length, number
    of columns = order
    """
    lag_features = []
    for k in range(num_lag):
        lag_features.append(shift(data, k))
    return np.array(lag_features).T


def shift(darray, k, axis=0):
    """
    Utility function to shift a numpy array

    Inputs
    ------
    darray: a numpy array
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

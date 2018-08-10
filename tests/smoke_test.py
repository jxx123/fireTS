from fireTS.models import NARX
from sklearn.linear_model import LinearRegression
import numpy as np


def test_NARX():
    x = np.random.randn(100, 1)
    y = np.random.randn(100)
    mdl = NARX(LinearRegression(), auto_order=2, exog_order=[1])
    mdl.fit(x, y)
    ypred = mdl.predict(x, y, step=3)


if __name__ == '__main__':
    test_NARX()

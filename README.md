# fireTS #
[![Documentation Status](https://readthedocs.org/projects/firets/badge/?version=latest)](https://firets.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/firets)](https://pepy.tech/project/firets)
[![Downloads](https://pepy.tech/badge/firets/month)](https://pepy.tech/project/firets)
[![Downloads](https://pepy.tech/badge/firets/week)](https://pepy.tech/project/firets)

[Documentation](https://firets.readthedocs.io/en/latest/)

# UPDATES
- 5/31/2020 `forecast` method is AVAILABLE now in `NARX` models!!! (`DirectAutoRegressor` is not suitable to do forecast, so there is no forecast method for it.) Here is a quick start example. Check "examples/Basic usage of NARX and DirectAutoregressor.ipynb" for more details.
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from fireTS.models import NARX

x = np.random.randn(100, 1)
y = np.random.randn(100)
mdl = NARX(LinearRegression(), auto_order=2, exog_order=[2])
mdl.fit(x, y)
y_forecast = mdl.forecast(x, y, step=10, X_future=np.random.randn((9, 1)))
```

# Introduction
`fireTS` is a sklean style package for multi-variate time-series prediction. Here is a simple code snippet to showcase the awesome features provided by `fireTS` package.
```python
from fireTS.models import NARX, DirectAutoRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

# Random training data
x = np.random.randn(100, 2)
y = np.random.randn(100)

# Build a non-linear autoregression model with exogenous inputs
# using Random Forest regression as the base model
mdl1 = NARX(
    RandomForestRegressor(n_estimators=10),
    auto_order=2,
    exog_order=[2, 2],
    exog_delay=[1, 1])
mdl1.fit(x, y)
ypred1 = mdl1.predict(x, y, step=3)

# Build a general autoregression model and make multi-step prediction directly
# using XGBRegressor as the base model
mdl2 = DirectAutoRegressor(
    XGBRegressor(n_estimators=10),
    auto_order=2,
    exog_order=[2, 2],
    exog_delay=[1, 1],
    pred_step=3)
mdl2.fit(x, y)
ypred2 = mdl2.predict(x, y)
```
- `sklearn` style API. The package provides `fit` and `predict` methods, which is very similar to `sklearn` package. 
- Plug-and-go. You are able to plug in any machine learning regression algorithms provided in `sklearn` package and build a time-series forecasting model.
- Create the lag features for you by specifying the autoregression order `auto_order`, the exogenous input order `exog_order`, and the exogenous input delay `exog_delay`.
- Support multi-step prediction. The package can make multi-step prediction in two different ways: recursive way and direct way. `NARX` model is to build a one-step-ahead-predictive model, and the model will be used recursively to make multi-step prediction (future exogenous input information is needed). `DirectAutoRegressor` makes multi-step prediction directly (no future exogenous input information is needed) by specifying the prediction step in the constructor.
- Support grid search to tune the hyper-parameters of the base model (cannot do grid search on the orders and delays of the time series model for now). 

I developed this package when writing [this paper](http://ceur-ws.org/Vol-2148/paper16.pdf). It is really handy to generate lag features and leverage various regression algorithms provided by sklearn to build non-linear multi-variate time series models. The API can also be used to build deep neural network models to make time-series prediction. [The paper](http://ceur-ws.org/Vol-2148/paper16.pdf) used this package to build LSTM models and make multi-step predictions.

The documentation can be found [here](https://firets.readthedocs.io/en/latest/). The documentation provides the mathematical equations of each model. It is highly recommended to read the documentation before using the model.

### Nonlinear AutoRegression with eXogenous (NARX) model
`fireTS.models.NARX` model is trying to train a one-step-ahead-prediction model
and make multi-step prediction recursively given the future exogenous inputs.

Given the output time series to predict `y(t)` and exogenous inputs `X(t)` The model will generate target and features as follows:

| Target | Features |
| ------------- |:-------------:|
| y(t + 1) | y(t), y(t - 1), ..., y(t - p + 1), X(t - d), X(t - d - 1), ..., X(t - d - q + 1) |

where p is the autogression order `auto_order`, q is the exogenous input order `exog_order`, d is the exogenous delay `exog_delay`.

NARX model can make any step ahead prediction given the future exogenous inputs. To make multi-step prediction, set the `step` in the `predict` method.

### Direct Autoregressor
`fireTS.models.DirectAutoRegressor` model is trying to train a 
multi-step-head-prediction model directly. No future exogenous inputs are
required to make the multi-step prediction.

Given the output time series to predict `y(t)` and exogenous inputs `X(t)` The model will generate target and features as follows:

| Target | Features |
| ------------- |:-------------:|
| y(t + k) | y(t), y(t - 1), ..., y(t - p + 1), X(t - d), X(t - d - 1), ..., X(t - d - q + 1) |

where p is the autogression order `auto_order`, q is the exogenous input order `exog_order`, d is the exogenous delay `exog_delay`, k is the prediction step `pred_step`.

Direct autoregressor does not require future exogenous input information to make multi-step prediction. Its `predict` method cannot specify prediction step.

## Installation ##
**NOTE**: Only python3 is supported.

It is highly recommended to use `pip` to install `fireTS`, follow this
 [link](https://pip.pypa.io/en/stable/installing/) to install pip.
 
After pip is installed, 
```
pip install fireTS
```

To get the latest development version, 
```
git clone https://github.com/jxx123/fireTS.git
cd fireTS
pip install -e .
```

## Quick Start ##
- Use `RandomForestRegressor` as base model to build a `NARX` model
```python
from fireTS.models import NARX
from sklearn.ensemble import RandomForestRegressor
import numpy as np

x = np.random.randn(100, 1)
y = np.random.randn(100)
mdl = NARX(RandomForestRegressor(), auto_order=2, exog_order=[2], exog_delay=[1])
mdl.fit(x, y)
ypred = mdl.predict(x, y, step=3)
```
- Use `RandomForestRegressor` as base model to build a `DirectAutoRegressor` model
```python
from fireTS.models import DirectAutoRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

x = np.random.randn(100, 1)
y = np.random.randn(100)
mdl = DirectAutoRegressor(RandomForestRegressor(), 
                          auto_order=2, 
                          exog_order=[2], 
                          exog_delay=[1], 
                          pred_step=3)
mdl.fit(x, y)
ypred = mdl.predict(x, y)
```
- Usage of grid search
```python
from fireTS.models import NARX
from sklearn.ensemble import RandomForestRegressor
import numpy as np

x = np.random.randn(100, 1)
y = np.random.randn(100)

# DirectAutoRegressor can do grid search as well
mdl = NARX(RandomForestRegressor(), auto_order=2, exog_order=[2], exog_delay=[1])

# Grid search
para_grid = {'n_estimators': [10, 30, 100]}
mdl.grid_search(x, y, para_grid, verbose=2)

# Best hyper-parameters are set after grid search, print the model to see the difference
print(mdl)

# Fit the model and make the prediction
mdl.fit(x, y)
ypred = mdl.predict(x, y, step=3)
```
The examples folder provides more realistic examples. The [example1](https://github.com/jxx123/fireTS/blob/master/examples/Basic%20usage%20of%20NARX%20and%20DirectAutoregressor.ipynb) and [example2](https://github.com/jxx123/fireTS/blob/master/examples/Use%20Grid%20Search%20to%20tune%20the%20hyper-parameter%20of%20base%20model.ipynb) use the data simulated by [simglucose pakage](https://github.com/jxx123/simglucose) to fit time series model and make multi-step prediction.

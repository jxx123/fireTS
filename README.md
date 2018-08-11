# fireTS #
[![Documentation Status](https://readthedocs.org/projects/firets/badge/?version=latest)](https://firets.readthedocs.io/en/latest/?badge=latest)

`fireTS` is a sklean style package for multi-variate time-series prediction. I
developed this package when writing [this
paper](http://ceur-ws.org/Vol-2148/paper16.pdf). It is really handy to generate lag features and leverage various regression algorithms provided by sklearn to build non-linear multi-variate time series models. The API can also be used to build deep neural network models to make time-series prediction. [The paper](http://ceur-ws.org/Vol-2148/paper16.pdf) used this package to build LSTM models and make multi-step predictions.

The documentation can be found [here](https://firets.readthedocs.io/en/latest/). The documentation provides the mathematical equations of each model. It is highly recommended to read the documentation before using the model.

### Nonlinear AutoRegression with eXogenous (NARX) model
`fireTS.models.NARX` model is trying to train a one-step-ahead-prediction model
and make multi-step prediction recursively given the future exogenous inputs.

Given the output time series to predict `y(t)` and exogenous inputs `X(t)` The model will generate target and features as follows:

| Target | Features |
| ------------- |:-------------:|
| y(t+1) | y(t), y(t - 1), ..., y(t - p + 1), X(t - d), X(t - d - 1), ..., X(t - d - q + 1) |

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

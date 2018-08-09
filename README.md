# fireTS #

`fireTS` is a sklean style package for multi-variate time-series prediction. I
developed this package when writing [this
paper](http://ceur-ws.org/Vol-2148/paper16.pdf). The paper introduced two
methods to perform multi-step prediction: recursive method and direct method.
`fireTS.models.NARX` model is trying to train a one-step-ahead-prediction model
and make multi-step prediction recursively given the future exogenous inputs.
`fireTS.models.DirectAutoRegressor` model is trying to train a
multi-step-head-prediction model directly. No future exogenous inputs are
required to make the multi-step prediction.

## Installation ##
It is highly recommended to use `pip` to install `fireTS`, follow this
 [link](https://pip.pypa.io/en/stable/installing/) to install pip.
 
After pip is installed, 
```
pip install firets
```

To get the latest development version, 
```
git clone 
```


* Dependencies
 - numpy
 - sklearn

## Quick Start ##

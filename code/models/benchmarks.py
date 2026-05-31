"""
- defines statistical benchmarks mean, ar1, naive
- get_history slices the target series strictly before origin (1-quarter publication lag)
- provides historical_mean, ar1, naive_method, drift_method, each returning (preds, {})
- all forecasts target origin+1 to origin+horizon (preds[0] = forecast for origin+1)
"""

# imports, config & utilities
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from config import TARGET, DATA_PATH

def get_history(origin):
    '''
    - returns the target series strictly before origin
    - equivalent to series up to including origin and applying the 1-quarter publication lag
    '''
    data = pd.read_csv(DATA_PATH)
    data['t'] = pd.to_datetime(data['t'])
    data = data[['t', TARGET]]
    return data.loc[data['t'] < origin, TARGET].reset_index(drop=True)


# models

def mean_method(horizon, origin):
    '''
    constant forecast equal to the mean of values < origin
    '''
    series = get_history(origin)
    return [series.mean()] * horizon, {}


def ar1(horizon, origin):
    '''
    - AR(1) fit on the target history < origin (conditional MLE via AutoReg)
    - series ends at origin - 1; predict(start=len(series)) would target origin itself
    - shifts the prediction window so preds target origin+1..origin+horizon
    '''
    series = get_history(origin)
    fit = AutoReg(series, lags=1, old_names=False).fit()
    preds = fit.predict(start=len(series) + 1, end=len(series) + horizon).tolist()
    return preds, {}


def naive_method(horizon, origin):
    '''
    repeats the last observed value of the target series
    '''
    series = get_history(origin)
    return [series.iloc[-1]] * horizon, {}
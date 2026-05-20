import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint


def compute_beta(series_a, series_b):
    model = OLS(series_a.values, add_constant(series_b.values)).fit()
    return float(model.params[1])


def compute_spread(series_a, series_b, beta):
    return series_a - beta * series_b


def smooth_spread(spread, span=8):
    return spread.ewm(span=span, adjust=False).mean()


def rolling_stats(spread, window):
    mu    = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    return mu, sigma


def cointegration_score(series_a, series_b):
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, _, _ = coint(series_a.values, series_b.values)
        score = float(t_stat)
        if np.isfinite(score):
            return score   # more negative -> stronger cointegration
        return 0.0
    except Exception:
        return 0.0

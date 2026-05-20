import pandas as pd
from src import config


def load_prices(path=None):
    path = path or config.DATA_PATH
    df = pd.read_csv(path, parse_dates=["Date"])
    prices = df.pivot(index="Date", columns="Symbol", values="Adj Close")
    prices.index = pd.DatetimeIndex(prices.index)
    prices = prices.sort_index()
    # forward-fill short gaps (holidays/halts), drop tickers with >20% missing
    prices = prices.ffill().dropna(axis=1, thresh=int(len(prices) * 0.8))
    universe = prices.columns.tolist()
    return prices, universe

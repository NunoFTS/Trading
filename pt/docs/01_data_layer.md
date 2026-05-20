# Data Layer — Implementation Reference

## Source File
- **Path:** `data/sp500_stocks.csv`
- **Format:** Long format — one row per (date, ticker)
- **Columns:**

| Column | Type | Notes |
|--------|------|-------|
| `Date` | string/date | Parse as datetime, set as index |
| `Symbol` | string | Ticker symbol, e.g. `AAPL` |
| `Adj Close` | float | **Use this for all price calculations** |
| `Close` | float | Unadjusted close — not used |
| `High` | float | Not used in Stage 1 |
| `Low` | float | Not used in Stage 1 |
| `Open` | float | Not used in Stage 1 |
| `Volume` | float | Not used in Stage 1 |

---

## Module: `src/data_loader.py`

### Responsibility
Load raw CSV → clean wide-format DataFrame indexed by date, with tickers as columns.

### Output Shape
```
DataFrame:
  Index: DatetimeIndex (trading days)
  Columns: ticker symbols (e.g. AAPL, MSFT, ...)
  Values: Adj Close prices (float64)
```

### Steps
1. Read CSV with `pd.read_csv`, parse `Date` column as datetime.
2. Set `Date` as index, sort ascending.
3. Pivot: `df.pivot(columns='Symbol', values='Adj Close')`.
4. Drop tickers with more than 20% missing values (too sparse to use).
5. Forward-fill remaining NaNs (max 5 days) to handle holidays/halts.
6. Drop any remaining NaN rows at the start (before first valid data).
7. Return the clean wide DataFrame.

### Train/Test Split
```python
split_ratio = 0.70
n = len(prices)
train = prices.iloc[:int(n * split_ratio)]
test  = prices.iloc[int(n * split_ratio):]
```
No date hardcoding — always split by index position based on available data.

### Universe Array
After loading, create a sorted array of valid tickers:
```python
universe = prices.columns.tolist()  # e.g. ['AAPL', 'ABBV', 'ABT', ...]
```
GA chromosomes store integer indices into this array (Gene 1 and Gene 2).

---

## Module: `src/utils.py`

### Responsibility
Shared mathematical helpers used across multiple modules.

### Functions

#### `ema(series, span=8)`
- Exponential moving average with `span` parameter.
- Use `pandas.Series.ewm(span=span, adjust=False).mean()`.
- Applied to the raw spread before threshold comparison.

#### `sma(series, window=30)`
- Simple moving average.
- Use `pandas.Series.rolling(window=window).mean()`.
- Used for momentum detection.

#### `rolling_zscore(series, window)`
- Compute rolling z-score of a series.
- `mu = series.rolling(window).mean()`
- `sigma = series.rolling(window).std()`
- Returns `(series - mu) / sigma`, NaNs at the start are expected.

#### `rolling_stats(series, window)`
- Returns `(rolling_mean, rolling_std)` as a tuple of two Series.
- Used in spread.py and strategy.py.

---

## Data Quality Notes
- The CSV is large (>50MB). Load once at startup; pass the DataFrame around — do not reload.
- Some tickers have gaps (delisted stocks, new additions). The 20% threshold handles this.
- Forward-fill only fills short gaps (holidays, trading halts). Do not fill large structural gaps.
- Always verify the date range after loading: `prices.index.min()` and `prices.index.max()`.

---

## Verification Checklist
After implementing, confirm:
- [ ] `prices.shape` — should be (trading_days, ~400–500 tickers)
- [ ] `prices.isnull().sum().sum()` — should be 0 after cleaning
- [ ] `prices.index` — should be DatetimeIndex, ascending, business days only
- [ ] `universe` — sorted list of strings, matches `prices.columns`
- [ ] Train/test split is non-overlapping and covers full date range

# Spread & Cointegration — Implementation Reference

## Core Concept
The spread is the core trading signal. For a pair (A, B):

$$S_t = A_t - \beta \cdot B_t$$

Where:
- $A_t$, $B_t$ are Adj Close price series (aligned on the same dates)
- $\beta$ is the **hedge ratio** — estimated so that $S$ is as stationary as possible
- A stationary spread is the prerequisite for pairs trading to work

---

## Module: `src/spread.py`

### Function: `compute_beta(series_a, series_b)`
Estimates the hedge ratio β using Ordinary Least Squares regression.

**Method:** Regress A on B (no intercept needed but include it for robustness):
```
A = β·B + ε
```
Use `statsmodels.OLS` or `numpy.polyfit(B, A, deg=1)`.

**Returns:** scalar float β

**Note:** β is computed on the **training window only**. It is fixed for the life of the trade. Never recompute β mid-trade.

---

### Function: `compute_spread(series_a, series_b, beta)`
Computes the raw spread series.

```python
spread = series_a - beta * series_b
```

**Returns:** `pd.Series` with same DatetimeIndex as inputs.

---

### Function: `rolling_spread_stats(spread, window)`
Computes rolling mean and std of the spread.

```python
mu    = spread.rolling(window).mean()
sigma = spread.rolling(window).std()
```

**Returns:** tuple `(mu, sigma)` — both `pd.Series`.

**Important:** The first `window` rows will be NaN — this is expected and must be handled in the strategy layer (skip those rows).

---

### Function: `smooth_spread(spread, ema_span=8)`
Applies EMA smoothing to the raw spread before generating signals.

```python
smoothed = spread.ewm(span=ema_span, adjust=False).mean()
```

**Why:** Reduces noise and prevents false threshold crossings from short-term volatility.

**Returns:** `pd.Series` — same index as input.

---

### Function: `cointegration_score(series_a, series_b)`
Runs the Engle-Granger cointegration test and returns a scalar score for use as a GA fitness objective.

**Method:**
1. Use `statsmodels.tsa.stattools.coint(series_a, series_b)`
2. Returns `(t_stat, p_value, critical_values)`
3. The **score** for the GA is the ADF **t-statistic** (more negative = more stationary = better cointegration)
4. Negate for minimisation in the GA: `score = -abs(t_stat)` — but confirm sign convention used in GA objective (see `docs/05_fitness_functions.md`)

**Returns:** float — the cointegration score (lower = more cointegrated)

**Caution:** This is expensive. Only call it once per chromosome evaluation, not on every backtest step.

---

## Z-Score Normalisation
The z-score of the smoothed spread is used for threshold comparisons:

$$z_t = \frac{S^{EMA}_t - \mu^{roll}_t}{\sigma^{roll}_t}$$

Where $S^{EMA}$ is the EMA-smoothed spread, and $\mu^{roll}$, $\sigma^{roll}$ are computed over the chromosome's rolling window.

Thresholds are expressed as multiples of σ:
- Entry long: $z_t > +\theta_{open\_long}$
- Entry short: $z_t < -\theta_{open\_short}$
- Close long: $z_t < +\theta_{close\_long}$
- Close short: $z_t > -\theta_{close\_short}$

Where $\theta$ values come from the chromosome (Genes 3–6).

---

## Engle-Granger Test — Background
- Tests whether the residual of the regression `A = β·B + ε` is stationary (ADF test on ε)
- p-value < 0.05 suggests cointegration at 95% confidence
- In the GA, we use the test statistic directly as a continuous fitness signal rather than a binary pass/fail

---

## Hedge Ratio Stability
- β should be relatively stable over time for the strategy to hold
- In Stage 1: β is computed once per chromosome evaluation over the full training window
- In later stages: rolling β estimation may be considered

---

## Verification Checklist
- [ ] `compute_beta` returns a positive number for typical correlated stock pairs
- [ ] `compute_spread` returns a series with mean near zero for cointegrated pairs
- [ ] `smooth_spread` visually reduces noise when plotted vs raw spread
- [ ] `cointegration_score` returns a more negative value for a known-good pair vs a random pair
- [ ] Rolling stats have NaNs in first `window` rows only

# Fitness Functions — Implementation Reference

## Two Objectives (NSGA-II Minimises Both)

| # | Name | Optimisation Direction | What It Measures |
|---|------|------------------------|-----------------|
| 1 | Cointegration | Minimise | Statistical stability of the pair (lower residual = better) |
| 2 | Negative ROI | Minimise | Past profitability (negate ROI so pymoo can minimise it) |

NSGA-II handles the trade-off between these two objectives. A pair with perfect cointegration but poor ROI, and a pair with high ROI but weaker cointegration, can both survive on the Pareto front.

---

## Objective 1: Cointegration Score

### What it is
The Engle-Granger cointegration test regresses A on B and tests whether the residuals are stationary using the Augmented Dickey-Fuller (ADF) test.

The **t-statistic** from the ADF test is the fitness value:
- More negative t-stat → stronger rejection of unit root → more stationary residuals → better cointegration
- Typical values: -5.0 is strong, -2.0 is weak

### Fitness value
```python
from statsmodels.tsa.stattools import coint

t_stat, p_value, _ = coint(series_a, series_b)
obj_cointegration = -t_stat   # negate so that minimising means more negative t-stat
```

Wait — re-check sign convention: `coint` returns a **negative** t-stat for cointegrated pairs (ADF on residuals). More negative = better cointegration. Since pymoo minimises, and we want to minimise (find lowest = most negative), we just pass `t_stat` directly as-is:

```python
# t_stat is already negative for cointegrated pairs
# More cointegrated → more negative t_stat
# pymoo minimises → seeks most negative → correct behaviour
obj_cointegration = t_stat   # already in correct sign for minimisation
```

**Confirm this sign when implementing.** Run on a known-good pair and verify the t_stat is negative.

---

## Objective 2: Historical ROI

### What it is
The return on investment from running the backtest on training data.

```python
roi = sum(trade['pnl'] for trade in trades) / initial_capital
```

Where `initial_capital` is a fixed reference value (e.g., 100.0 or the initial spread magnitude).

### Fitness value
Since pymoo minimises, negate ROI:
```python
obj_roi = -roi    # maximising ROI = minimising -ROI
```

If no trades were executed (zero trades): `obj_roi = 0.0` (neutral, not penalised further — the cointegration score alone governs selection).

---

## Magnitude Penalisation (Q)

### Purpose
Biases the GA toward pairs that are "ready to trade" — pairs whose spread magnitude is currently in a tradeable range, not too close to the mean and not too far away.

### Definition
At the start of each evaluation window, compute the z-score of the spread's current value:

$$z_{current} = \frac{S_{current} - \mu_{roll}}{\sigma_{roll}}$$

**Penalise if** $|z_{current}| \notin [1, 3]$:
- $|z| < 1$: Spread too close to mean → no trading opportunity
- $|z| > 3$: Spread too far from mean → likely a structural break, not a reversion opportunity

### Penalty application
```python
z_current = abs((spread.iloc[-1] - mu) / sigma)

if z_current < 1.0 or z_current > 3.0:
    penalty = Q   # Q is a large positive constant, e.g. 10.0
else:
    penalty = 0.0

obj_roi = -roi + penalty
```

### Q value
Use `Q = 10.0` as a starting point. This should be large enough to move penalised solutions off the Pareto front but not so large it creates numerical instability. Tune if needed.

---

## Module: `src/fitness.py`

### Function: `evaluate_chromosome(x, prices, universe)`
Single entry point called by the pymoo problem's `_evaluate` method.

**Steps:**
1. Decode chromosome x → ticker_a, ticker_b, params dict
2. Guard: if ticker_a == ticker_b, return `(0.0, 0.0)` (worst fitness — all valid solutions dominate this)
3. Extract price series A, B from `prices`
4. Compute β and spread
5. Compute rolling stats with Gene 7 window
6. Compute smoothed spread (EMA)
7. Compute z-score series
8. Compute cointegration score → `obj1`
9. Compute magnitude penalty (check current z at end of training window)
10. Run backtest → get trades → compute ROI
11. Apply penalty → `obj2 = -roi + penalty`
12. Return `(obj1, obj2)`

**Performance note:** This function is called `pop_size × generations` times (up to 250 × 80 = 20,000 times). Keep it fast. The bottleneck is usually the `coint` call. Consider caching cointegration results by (ticker_a, ticker_b) pair if the same pair appears in multiple generations.

---

## Caching Strategy (Optional Optimisation)
```python
_coint_cache = {}

def cointegration_score_cached(series_a, series_b, key):
    if key not in _coint_cache:
        _coint_cache[key] = cointegration_score(series_a, series_b)
    return _coint_cache[key]
```
Where `key = (min(idx_a, idx_b), max(idx_a, idx_b))` — order-independent.
Cache is cleared each time a new training window is used (5-day recalibration).

---

## Verification Checklist
- [ ] `obj_cointegration` is more negative for a known-good pair than a random pair
- [ ] `obj_roi` is negative for profitable strategies (confirms negation is correct)
- [ ] Magnitude penalty `Q` visibly pushes penalised solutions away from Pareto front
- [ ] No crashes when zero trades occur (empty trade list handled gracefully)
- [ ] Chromosome with `stock_a == stock_b` returns dominated fitness, never on Pareto front

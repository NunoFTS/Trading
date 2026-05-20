# Portfolio Management & Output — Implementation Reference

## Overview
After the GA finishes evolving, the Pareto front is extracted and converted into a tradeable portfolio. Capital is distributed equally across selected pairs, and the system recalibrates every 5 trading days. The final output is a plot of trades on the training spread.

---

## Module: `src/portfolio.py`

### Step 1: Pareto Front Extraction

The Pareto front (Rank 0 in NSGA-II) contains the best trade-off solutions. These become the portfolio candidates.

```python
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

sorter = NonDominatedSorting()
fronts = sorter.do(fitness_matrix)   # fitness_matrix shape: (n_individuals, 2)
pareto_indices = fronts[0]           # indices of Rank-0 solutions
pareto_chromosomes = population[pareto_indices]
```

**Or** use the result object directly if using `pymoo.optimize.minimize`:
```python
pareto_chromosomes = result.X   # pymoo returns only Pareto-front solutions by default
pareto_fitness     = result.F
```

### Step 2: Capital Allocation

Capital is split equally among all pairs on the Pareto front, up to L pairs.

```python
CAPITAL_SPLIT_L = 75       # from config.py
total_capital   = 100_000  # or whatever starting capital is

n_pairs         = min(len(pareto_chromosomes), CAPITAL_SPLIT_L)
capital_per_pair = total_capital / CAPITAL_SPLIT_L
```

**Note:** `L = 75` is a fixed denominator, not the actual number of Pareto-front solutions. It normalises the position size regardless of how many pairs are currently active.

### Step 3: 5-Day Recalibration Loop

The system re-runs the GA on a rolling basis. Every 5 trading days:
1. Advance the training window by 5 days.
2. Re-run `evolve()` on the new window.
3. Extract new Pareto front.
4. Any open trades from the previous cycle continue to their natural exit (or are carried forward if still valid).

```python
RECALIBRATION_DAYS = 5

for start in range(0, len(train_prices), RECALIBRATION_DAYS):
    window_prices = train_prices.iloc[start:start + window_size]
    result = evolve(window_prices, universe)
    portfolio = extract_portfolio(result)
    # ... run strategy forward for next 5 days
```

**Stage 1 simplification:** For the initial implementation, run the GA once on the full training set and use those results to generate the trade plot. The 5-day rolling loop adds complexity that can be layered in once the core works.

---

## Module: `src/plot.py`

### The Only Required Output for Stage 1

**Function: `plot_trades(spread_series, z_score_series, trades, title="")`**

**What it shows:**
- Spread z-score over time (line)
- Entry/exit threshold lines (horizontal dashed lines at ±open_long, ±open_short)
- Long entries: green upward triangles
- Long exits: green downward triangles
- Short entries: red downward triangles
- Short exits: red upward triangles
- Trade durations shaded in the background (green = long, red = short)

**Suggested layout:**
```
Upper panel: Z-score of smoothed spread
  - z-score line (blue)
  - ±open threshold dashed lines (orange)
  - ±close threshold dashed lines (green)
  - Entry markers (triangles)
  - Exit markers (triangles)
Lower panel (optional): individual stock prices A and B
```

**Minimal implementation (just upper panel):**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_trades(z_score, trades, open_long, open_short, close_long, close_short, title=""):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(z_score.index, z_score.values, color='steelblue', lw=1, label='Z-score')
    ax.axhline(+open_long,  color='orange', ls='--', lw=0.8, label=f'+open_long ({open_long:.2f})')
    ax.axhline(-open_short, color='orange', ls='--', lw=0.8)
    ax.axhline(+close_long, color='green',  ls='--', lw=0.8, label=f'+close ({close_long:.2f})')
    ax.axhline(-close_short,color='green',  ls='--', lw=0.8)
    ax.axhline(0, color='black', lw=0.5, ls='-')

    for trade in trades:
        color = 'green' if trade['direction'] == 'long' else 'red'
        marker_up = '^' if trade['direction'] == 'long' else 'v'
        marker_dn = 'v' if trade['direction'] == 'long' else '^'
        ax.scatter(trade['entry_date'], trade['entry_z'], color=color, marker=marker_up, s=80, zorder=5)
        ax.scatter(trade['exit_date'],  trade['exit_z'],  color=color, marker=marker_dn, s=80, zorder=5)

    ax.set_title(title or 'Pairs Trading — Training Trades')
    ax.set_xlabel('Date')
    ax.set_ylabel('Z-Score')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('output/trades.png', dpi=150)
    plt.show()
```

---

## `main.py` — End-to-End Runner

```python
# main.py
from src.data_loader import load_prices
from src.ga import evolve
from src.fitness import evaluate_chromosome
from src.plot import plot_trades
from src.spread import compute_beta, compute_spread, smooth_spread, rolling_spread_stats
from src.backtest import run_backtest
from src import config

prices, universe = load_prices('data/sp500_stocks.csv')
split = int(len(prices) * 0.70)
train = prices.iloc[:split]
test  = prices.iloc[split:]

result = evolve(train, universe)

# Take best chromosome from Pareto front (e.g. best ROI)
best_idx = result.F[:, 1].argmin()   # most negative ROI = highest ROI
best_chrom = result.X[best_idx]

# Decode
idx_a = int(round(best_chrom[0]))
idx_b = int(round(best_chrom[1]))
ticker_a, ticker_b = universe[idx_a], universe[idx_b]
params = { ... }  # decode remaining genes

A, B = train[ticker_a], train[ticker_b]
beta = compute_beta(A, B)
spread = compute_spread(A, B, beta)
smoothed = smooth_spread(spread)
mu, sigma = rolling_spread_stats(smoothed, params['window'])
z_score = (smoothed - mu) / sigma

trades = run_backtest(smoothed, z_score, A, B, params,
                      A.rolling(30).mean(), B.rolling(30).mean())

plot_trades(z_score, trades, params['open_long'], params['open_short'],
            params['close_long'], params['close_short'],
            title=f'{ticker_a} / {ticker_b} — Training Trades')
```

---

## Output Directory
Save plots to `output/` directory. Create if it doesn't exist:
```python
import os
os.makedirs('output', exist_ok=True)
```

---

## Verification Checklist
- [ ] Pareto front is non-empty after GA run
- [ ] Capital per pair is `total_capital / 75` regardless of Pareto front size
- [ ] Trade plot shows at least a few trades for a good pair
- [ ] Entry markers are above the open threshold line; exit markers near close threshold
- [ ] Plot is saved to `output/trades.png`
- [ ] `main.py` runs end-to-end without errors

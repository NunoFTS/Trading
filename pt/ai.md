# AI Programming Rules — Pairs Trading System

## Project Identity
- **Goal:** Implement a Genetic Algorithm-optimised Pairs Trading system in 4 progressive stages (SU → MU → XMU → GXMU).
- **Current scope:** Stage 1 — Simple Unified system (SU) only.
- **Language:** Python 3.11+
- **Only required output right now:** A plot of trades on training data.

---

## Code Rules

### General
- Keep every module under ~200 lines. Single responsibility per file.
- No global state. Pass data explicitly between functions.
- No premature abstraction. If something is used once, don't wrap it in a helper.
- No docstrings or comments unless the logic is non-obvious math.
- No error handling for scenarios that can't happen at runtime boundaries.
- Do not add features beyond what the current stage requires.

### Naming
- Files: `snake_case.py`
- Functions: `snake_case`
- Classes: `PascalCase` (avoid unless truly stateful)
- Constants (config values): `UPPER_SNAKE_CASE` in a `config.py`
- Variables: descriptive, no single letters except loop counters and math (e.g., `beta`, `mu`, `sigma` are fine)

### Data Conventions
- Price data: always use **Adj Close** column.
- DataFrames: always `DatetimeIndex`, tickers as columns.
- Never mutate a DataFrame in place — always assign to a new variable or use `.copy()`.
- All spread/price series must be aligned on the same date index before any math.

### Math/Finance Conventions
- Spread: `S = A − β·B` where β is the OLS hedge ratio from Engle-Granger cointegration.
- Z-score of spread: `z = (S − rolling_mean) / rolling_std`
- EMA smoothing: 8-day span applied to raw spread before computing z-score thresholds.
- Momentum: defined by 30-day Simple Moving Average slope (price > SMA → bullish).
- Rolling window: configurable per chromosome, range 200–400 days.

### GA / NSGA-II Rules
- Use `pymoo` library for NSGA-II. Do not reimplement the algorithm.
- Always encode stock identifiers as integer indices into the universe array, not ticker strings.
- Gene bounds must be enforced in the problem definition (not post-hoc clipping).
- Pareto front extraction uses `pymoo`'s built-in `NonDominatedSorting`.

### Trading Logic Rules
- Never enter a trade on the first threshold crossing — always wait for the **second** (Simple Reversion Test).
- Momentum filter is checked **before** every entry attempt. If violated, skip entirely.
- Adaptive exit: every time the spread crosses the close threshold, shift it by `exit_step · σ` toward zero.
- Stop-loss gene (Gene 9) is present in the chromosome but **not triggered** in Stage 1.
- All trade PnL is computed on the spread, not individual stock returns.

### Backtesting Rules
- No lookahead bias: rolling stats use only past data at each timestep.
- Trade log entries: `(entry_date, exit_date, direction, pnl, entry_z, exit_z)`.
- Historical ROI = `sum(pnl) / initial_capital`.

---

## Architecture — File Map

```
x:\pt\
  src/
    config.py        ← all hyperparameters and constants
    data_loader.py   ← CSV → clean wide DataFrame
    utils.py         ← EMA, SMA, rolling z-score helpers
    spread.py        ← beta, spread, cointegration score
    strategy.py      ← entry/exit/momentum/reversion logic
    backtest.py      ← run_backtest() → trade log + ROI
    fitness.py       ← cointegration fitness, ROI fitness, magnitude penalty
    ga.py            ← pymoo problem definition + evolve()
    portfolio.py     ← Pareto extraction, capital allocation, recalibration
    plot.py          ← plot_trades() — the only required output
  tests/
    test_spread.py       ← 24 tests: compute_beta/spread/smooth/rolling_stats/coint
    test_backtest.py     ← 44 tests: state machine, guard, expiry, force-close, PnL
    test_fitness.py      ← 20 tests: decode() clamping, evaluate_chromosome() edge cases
    test_data_loader.py  ← 17 tests: load, ffill, sparse drop, sorting
    test_integration.py  ← 15 tests: end-to-end pipeline + logging
  main.py            ← end-to-end runner
  data/
    sp500_stocks.csv
  docs/              ← reference documentation (see below)
  ai.md              ← this file
```

---

## Stage Scope Boundaries

| Feature | Stage 1 (SU) | Stage 2 (MU) | Stage 3 (XMU) | Stage 4 (GXMU) |
|---|---|---|---|---|
| GA pair selection | ✓ | ✓ | ✓ | ✓ |
| Cointegration fitness | ✓ | ✓ | ✓ | ✓ |
| Historical ROI fitness | ✓ | — | — | — |
| Monte Carlo (OU-VG) | — | ✓ | ✓ | ✓ |
| Expected Profitability fitness | — | ✓ | ✓ | ✓ |
| Stop-loss gene (active) | — | ✓ | ✓ | ✓ |
| XGBoost sentiment gatekeeper | — | — | ✓ | ✓ |
| Graph portfolio optimisation | — | — | — | ✓ |
| Sector constraints | — | — | — | ✓ |

---

## Key Hyperparameters (Stage 1)

| Parameter | Value | Source |
|---|---|---|
| Population size | 250 | Paper §optimisation |
| Generations | 80 | Paper §optimisation |
| Capital split L | 75 | Paper §portfolio |
| Recalibration interval | 5 days | Paper §portfolio |
| EMA span | 8 days | Paper §signal smoothing |
| Momentum SMA | 30 days | Paper §constraints |
| Rolling window range | 200–400 days | Paper §chromosome |
| Crossover | Uniform | Paper §GA |
| Mutation | Randomised per gene | Paper §GA |

---

## Docs Index

| File | Contents |
|---|---|
| `docs/00_project_overview.md` | Full thesis summary, 4 stages, empirical results |
| `docs/01_data_layer.md` | CSV schema, preprocessing steps, data quality checks |
| `docs/02_spread_cointegration.md` | Spread formula, Engle-Granger, rolling stats |
| `docs/03_trading_strategy.md` | All trading rules: momentum, reversion test, adaptive exit |
| `docs/04_chromosome_ga.md` | 9-gene chromosome, NSGA-II config, pymoo usage |
| `docs/05_fitness_functions.md` | Two objectives, magnitude penalty Q |
| `docs/06_portfolio_output.md` | Pareto extraction, capital allocation, recalibration, plot |
| `docs/progress.md` | Build progress tracker — updated as phases complete |

---

## Testing

Run the suite (all 120 tests, ~9 s):
```
x:\pt\.venv\Scripts\python.exe -m pytest tests\ -v
```

To enable DEBUG logging during a main.py run, add at the top of `main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Logs emitted by `src.backtest` and `src.fitness` cover every state transition,
entry-guard rejection, PRIMED expiry, beta-out-of-bounds, and force-close event.

---

## Progress Summary
- [x] Planning complete
- [x] Docs written
- [x] Phase 1 — Data layer (`src/data_loader.py`, `src/config.py`)
- [x] Phase 2 — Spread & Cointegration (`src/spread.py`)
- [x] Phase 3 — Trading Strategy (`src/backtest.py`)
- [x] Phase 4 — Fitness Functions (`src/fitness.py`)
- [x] Phase 5 — NSGA-II GA (`src/ga.py`)
- [x] Phase 6 — Portfolio & Trade Plot (`src/plot.py`, `main.py`)
- [x] Phase 7 — Code review fixes (7 bugs resolved)
- [x] Phase 8 — Unit tests + logging (`tests/`, 120 tests all green)
- [ ] Stage 2 — Monte Carlo / Expected Profitability
- [ ] Stage 3 — XGBoost sentiment gatekeeper
- [ ] Stage 4 — Graph portfolio optimisation

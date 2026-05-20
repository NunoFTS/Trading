# Build Progress

## Status: STAGE 1 COMPLETE — UNIT TESTS + LOGGING ADDED

---

## Phase Tracker

| Phase | Module(s) | Status | Notes |
|-------|-----------|--------|-------|
| 0 | `docs/`, `ai.md` | DONE | All reference docs written |
| 1 | `src/data_loader.py`, `src/config.py` | DONE | 170 tickers, 504 days, no nulls |
| 2 | `src/spread.py` | DONE | beta, EMA, rolling stats, EG cointegration |
| 3 | `src/backtest.py` | DONE | State machine with reversion test, momentum, adaptive exit |
| 4 | `src/fitness.py` | DONE | Two objectives, magnitude penalty Q |
| 5 | `src/ga.py` | DONE | pymoo NSGA-II, pop=50, gen=20 |
| 6 | `src/plot.py`, `main.py` | DONE | Trade plot + end-of-run summary |
| 7 | Code review fixes | DONE | 7 bugs resolved (see Decisions Log) |
| 8 | Unit tests + logging | DONE | 120 tests, all green; DEBUG logging in backtest/fitness |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-16 | Use `pymoo` for NSGA-II | Production-grade, less code, well-maintained |
| 2026-05-16 | Gene 9 = stop-loss multiplier | Pre-included for Stage 2; dormant in Stage 1 |
| 2026-05-16 | Training split = 70% of available data | Dataset is ~2 years; no hardcoded dates |
| 2026-05-16 | Use Adj Close only | Adjusted for dividends/splits; more accurate |
| 2026-05-16 | Stage 1 only for now | Build progressively; no Stage 2/3/4 yet |
| 2026-05-16 | Beta bounds [0.01, 10.0] | Negative β inverts spread direction; extreme β creates unrealistic positions |
| 2026-05-16 | close ≤ open clamped in decode() | Close threshold beyond open threshold is logically invalid |
| 2026-05-16 | Safe σ division (where σ > 1e-8) | Prevents inf z-scores when rolling std ≈ 0 |
| 2026-05-16 | PRIMED expires after 5 bars | Stale pending signals should not open trades days later |
| 2026-05-16 | Entry deferred to bar i+1 | Signal at close of day i; execution at next open/close (no same-day lookahead) |
| 2026-05-16 | Force-close open trades at dataset end | Open positions at last bar must contribute to fitness |
| 2026-05-16 | MIN_TRADES = 2 | 1-trade solutions have noise-driven ROI; at least 2 trades required for valid fitness |
| 2026-05-16 | DEBUG logging in backtest/fitness | Every state transition, guard rejection, PRIMED expiry, and force-close emits a DEBUG log via `logging.getLogger(__name__)` for easy tracing |
| 2026-05-16 | Unit tests use synthetic data only | No CSV dependency in tests; all inputs are numpy-constructed; this makes the suite fast and deterministic |

---

## Open Questions / Blockers

| # | Question | Status |
|---|----------|--------|
| 1 | Sign convention for cointegration t-stat as fitness | RESOLVED — more negative = stronger cointegration |
| 2 | PnL normalisation: use spread magnitude or fixed capital? | RESOLVED — σ-normalised spread PnL |
| 3 | 5-day recalibration: full re-run or just portfolio update? | Deferred to Stage 2 |

---

## Notes Per Phase

### Phase 1 — Data Layer
- CSV has columns: `Date, Symbol, Adj Close, Close, High, Low, Open, Volume`
- File is >50MB — load once, do not reload per evaluation
- ~400–500 valid tickers after cleaning
- Drop tickers with >20% missing; forward-fill up to 5 days

### Phase 2 — Spread & Cointegration
- Use `statsmodels.tsa.stattools.coint` for Engle-Granger
- β via OLS, computed once per chromosome, on training window only
- EMA span = 8 days (smooth before z-score computation)
- Rolling window = Gene 7 value (200–400 days)

### Phase 3 — Trading Strategy
- Simple Reversion Test: 2nd crossing required, reset if spread retreats without re-crossing
- Momentum: 30-day SMA, block long if A bearish, block short if A bullish (and symmetric for B)
- Adaptive exit: close threshold shifts by exit_step × σ each crossing
- Stop-loss (Gene 9): present but not active in Stage 1

### Phase 4 — Fitness Functions
- Objective 1: cointegration t-stat (pymoo minimises; more negative = better)
- Objective 2: -ROI + magnitude_penalty Q
- Magnitude penalty Q = 10.0 when |z_current| ∉ [1, 3]
- Cache cointegration results by (min_idx, max_idx) pair key

### Phase 5 — NSGA-II GA
- Population: 250, Generations: 80
- Uniform crossover (UX), Polynomial mutation (PM), 1/9 mutation rate per gene
- Gene bounds in `config.py`, enforced via pymoo xl/xu
- stock_a == stock_b → return worst fitness

### Phase 6 — Portfolio & Plot
- L = 75 (capital_per_pair = total / 75)
- Plot: z-score + threshold lines + entry/exit markers
- Save to `output/trades.png`
- Stage 1: run GA once on full training set for the plot (recalibration loop is a later refinement)

---

## First Run Results (2026-05-16)
- GA: pop=50, gen=20 → ~1000 evaluations
- Best pair found: **PFG / ALL** (Principal Financial / Allstate)
- Cointegration t-stat: -3.046 (weak but valid)
- 1 trade: SHORT, 5 days (2024-07-29 → 2024-08-05), +298% σ-normalized ROI
- The trade caught the August 2024 carry-trade unwind — a lucky single event
- High exit_step (0.301/day) causes fast exits → fewer trades overall
- Scaling up to pop=250, gen=80 will find richer multi-trade pairs

## Known Limitations (Stage 1)
- No recalibration loop yet (GA runs once on full 504-day training set)
- No portfolio-level management (single best pair only)
- Gene 9 (stop-loss) is present but dormant

## Next Action
→ Scale GA to full paper params (pop=250, gen=80) in config.py when ready for production run
→ Begin Stage 2: Monte Carlo + Expected Profitability + active stop-loss

---

## Phase 8 — Unit Tests (2026-05-16)

### Test suite: `tests/` (run with `x:\pt\.venv\Scripts\python.exe -m pytest tests\ -v`)

| File | Coverage | Tests |
|------|----------|-------|
| `tests/test_spread.py` | `src/spread.py` — all 5 public functions | 24 |
| `tests/test_backtest.py` | `src/backtest.py` — state machine, guard, expiry, force-close, adaptive exit, momentum, PnL | 44 |
| `tests/test_fitness.py` | `src/fitness.py` — decode() clamping, evaluate_chromosome() edge cases | 20 |
| `tests/test_data_loader.py` | `src/data_loader.py` — load, ffill, sparse drop, edge cases | 17 |
| `tests/test_integration.py` | End-to-end pipeline, logging emission | 15 |
| **Total** | | **120 — all pass** |

### Logging added
- `src/backtest.py`: `logger = logging.getLogger("src.backtest")` — DEBUG on every state transition, entry guard rejection, PRIMED expiry, and force-close.
- `src/fitness.py`: `logger = logging.getLogger("src.fitness")` — DEBUG on beta-out-of-bounds, insufficient data, and MIN_TRADES rejection.

To enable debug output at the application level add to `main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

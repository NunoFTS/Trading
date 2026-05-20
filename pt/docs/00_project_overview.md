# Project Overview — Pairs Trading System

## What This Is
A modular, machine learning-enhanced **pairs trading** framework built in Python. It evolves through 4 distinct stages (case studies), each adding a layer of sophistication. The foundation is a Genetic Algorithm that simultaneously selects stock pairs and optimises trading parameters.

Pairs trading is a **statistical arbitrage** strategy: when two historically correlated assets diverge, bet on their convergence. The spread `S = A − β·B` is the core signal.

---

## The 4 Stages

### Stage 1 — Simple Unified System (SU) ← CURRENT
Co-optimises pair formation and trading strategy in one evolutionary process.
- **GA:** NSGA-II evolves 9-gene chromosomes
- **Objectives:** Cointegration quality + Historical ROI
- **Output:** Trade plot on training data

### Stage 2 — Monte Carlo Unified System (MU)
Replaces historical ROI with a forward-looking Expected Profitability metric.
- **Spread model:** Ornstein–Uhlenbeck Variance Gamma (OU-VG) process
- **Simulations:** 1000 paths per pair
- **New gene:** Stop-loss multiplier (Gene 9 activated)
- **Metric:** Risk-adjusted Value Function V with variance penalisation (γ)

### Stage 3 — XGBoost Monte Carlo Unified System (XMU)
Adds a sentiment-based gatekeeper to filter trade entries.
- **Classifier:** XGBoost ensemble predicts next-day spread derivative
- **5 regimes:** Strong/Mild Divergence | Stable | Strong/Mild Convergence
- **Entry gate:** Only enter if regime is Stable or Converging
- **Exit trigger:** Close immediately on Strong Divergence prediction
- **GA role:** Selects the XGBoost model from the ensemble that maximises EP

### Stage 4 — Graph-based Portfolio (GXMU)
Optimises capital allocation across all active trades using graph theory.
- **Graph:** Nodes = stocks, Edges = trades weighted by Expected Profit (EP)
- **Problem:** Edge maximisation subject to diversification constraints
- **Constraints:** Max 1 long + 1 short per sector at any time
- **Rebalancing:** Every 5 days; can cash out open trades for higher-EP ones

---

## Empirical Results (S&P 500, 2000–2024)

| System | Sharpe Ratio | Max Drawdown | Annual Return |
|--------|-------------|--------------|---------------|
| S&P 500 (benchmark) | ~0.5–0.8 | ~50% | ~10% |
| Traditional PT | ~1.0 | ~15% | ~8% |
| SU (Stage 1) | — | — | ~10% |
| MU (Stage 2) | — | — | improving |
| XMU (Stage 3) | — | — | improving |
| GXMU (Stage 4) | **3.71** | **5%** | improving |

---

## Data
- **Source:** S&P 500 stocks from Kaggle (`andrewmvd/sp-500-stocks`)
- **File:** `data/sp500_stocks.csv`
- **Columns:** `Date, Symbol, Adj Close, Close, High, Low, Open, Volume`
- **Universe:** All S&P 500 tickers (no sector restriction in Stage 1)
- **Price used:** Adj Close (adjusts for dividends and splits)
- **Split:** ~70% training / ~30% testing (exact dates set at runtime based on available data)

---

## Core Concepts

### Cointegration
Two series A, B are cointegrated if `A − β·B` is stationary (mean-reverting). The Engle-Granger test confirms this. β (hedge ratio) is computed via OLS. A lower test residual = stronger cointegration.

### Spread Z-Score
$$z_t = \frac{S_t - \mu_{roll}}{\sigma_{roll}}$$
Where $\mu_{roll}$ and $\sigma_{roll}$ are rolling mean and std over the chromosome's window.

### Pairs Trading Logic
- **Long spread:** Buy A, Short B (when spread is below its mean → expect it to rise)
- **Short spread:** Short A, Buy B (when spread is above its mean → expect it to fall)

### NSGA-II
Non-dominated Sorting Genetic Algorithm II. Handles multi-objective optimisation by ranking solutions by Pareto dominance and maintaining diversity via crowding distance. A solution dominates another if it's better on at least one objective and no worse on all others.

---

## Technology Stack
- Python 3.11+
- pandas, numpy — data handling
- statsmodels — Engle-Granger cointegration, OLS
- pymoo — NSGA-II implementation
- matplotlib — trade plots
- scipy — statistical utilities
- (Stage 2+) xgboost, scikit-learn

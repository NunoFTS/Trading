# Chromosome Design & NSGA-II GA — Implementation Reference

## The 9-Gene Chromosome

Each individual in the GA population is a vector of 9 values representing a complete trading configuration:

| Gene | Name | Type | Range | Description |
|------|------|------|-------|-------------|
| 0 | `stock_a` | int | [0, N-1] | Index into universe array for stock A |
| 1 | `stock_b` | int | [0, N-1] | Index into universe array for stock B |
| 2 | `open_long` | float | [1.0, 3.0] | Entry threshold multiplier for long trades |
| 3 | `open_short` | float | [1.0, 3.0] | Entry threshold multiplier for short trades |
| 4 | `close_long` | float | [0.0, 1.5] | Exit threshold multiplier for long trades |
| 5 | `close_short` | float | [0.0, 1.5] | Exit threshold multiplier for short trades |
| 6 | `exit_step` | float | [0.05, 0.5] | Adaptive exit step size (Δc in σ units) |
| 7 | `window` | int | [200, 400] | Rolling window for spread statistics (days) |
| 8 | `stop_loss` | float | [0.5, 3.0] | Stop-loss multiplier (dormant in Stage 1) |

**Notes:**
- Genes 0–1 are integers but stored as floats and rounded during evaluation.
- stock_a ≠ stock_b must be enforced (resample if they collide during mutation).
- close_long < open_long and close_short < open_short should hold logically (can't exit at a wider threshold than you entered).

---

## NSGA-II Configuration

### Library: `pymoo`

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
```

### GA Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Population size | 250 | Paper-confirmed optimal |
| Generations | 80 | Paper-confirmed optimal |
| Crossover | Uniform (UX) | Each gene independently picked from either parent |
| Crossover probability | 0.9 | Standard |
| Mutation | Polynomial mutation (PM) | Smooth perturbation within bounds |
| Mutation probability | 1/9 per gene | One gene on average per individual |
| Selection | Tournament (default NSGA2) | Built into pymoo NSGA2 |

### pymoo Problem Class Structure

```python
class PairsTradingProblem(Problem):
    def __init__(self, prices, universe):
        n_var = 9
        n_obj = 2        # cointegration + ROI
        n_constr = 0     # constraints handled in evaluation
        
        xl = [0, 0, 1.0, 1.0, 0.0, 0.0, 0.05, 200, 0.5]
        xu = [N-1, N-1, 3.0, 3.0, 1.5, 1.5, 0.5, 400, 3.0]
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr,
                         xl=xl, xu=xu, elementwise_evaluation=True)
        self.prices = prices
        self.universe = universe
    
    def _evaluate(self, x, out, *args, **kwargs):
        # x is a single chromosome (9 values)
        # out['F'] = [cointegration_objective, roi_objective]
        ...
```

**Important:** Use `elementwise_evaluation=True` so `_evaluate` receives one chromosome at a time. This simplifies the code significantly.

---

## Evaluation Logic (per chromosome)

```
1. Extract ticker indices (round Genes 0–1 to int)
2. If stock_a == stock_b → return worst fitness values (penalise)
3. Extract price series A and B from training data
4. Compute beta = compute_beta(A, B)
5. Compute raw spread, smooth (EMA), compute rolling stats with window=Gene 7
6. Compute cointegration_score(A, B)  → obj_1 (minimise)
7. Run backtest with params from Genes 2–8
8. Compute ROI → obj_2 = -ROI (negate because pymoo minimises)
9. Apply magnitude penalty to obj_2 if needed
10. out['F'] = [obj_1, obj_2]
```

---

## Pareto Dominance

NSGA-II ranks solutions by **Pareto dominance**: solution X dominates Y if:
- X is no worse than Y on ALL objectives
- X is strictly better than Y on AT LEAST ONE objective

The **Pareto front** (Rank 0) = solutions not dominated by any other. These are the "best" trade-offs between cointegration quality and ROI.

**Crowding distance** maintains diversity within each Pareto rank — prevents the population from collapsing to a single point.

---

## Module: `src/ga.py`

### Main Function: `evolve(prices, universe, generations=80, pop_size=250)`

**Returns:** `pymoo.Result` object containing:
- `res.X` — final population chromosomes
- `res.F` — final population fitness values
- `res.algorithm.pop` — full population with rank info

**Usage:**
```python
result = evolve(train_prices, universe)
pareto_X = result.X  # Pareto-front chromosomes
pareto_F = result.F  # their fitness values
```

---

## Gene Bounds in `src/config.py`

All gene bounds should live in `config.py`, not hardcoded in the problem class:

```python
GENE_BOUNDS = {
    'stock_a':    (0, None),       # None = len(universe) - 1, set at runtime
    'stock_b':    (0, None),
    'open_long':  (1.0, 3.0),
    'open_short': (1.0, 3.0),
    'close_long': (0.0, 1.5),
    'close_short':(0.0, 1.5),
    'exit_step':  (0.05, 0.5),
    'window':     (200, 400),
    'stop_loss':  (0.5, 3.0),
}

GA_POPULATION = 250
GA_GENERATIONS = 80
EMA_SPAN = 8
SMA_WINDOW = 30
CAPITAL_SPLIT_L = 75
RECALIBRATION_DAYS = 5
```

---

## Verification Checklist
- [ ] Chromosome round-trips correctly (decode → encode produces same values)
- [ ] `stock_a != stock_b` always enforced before evaluation
- [ ] `pymoo` minimisation: confirm ROI is negated (maximise ROI = minimise -ROI)
- [ ] 5-generation test run produces a non-empty Pareto front
- [ ] Gene bounds are respected in all offspring (pymoo handles this via xl/xu)
- [ ] Fitness improves (or at worst stays stable) over generations

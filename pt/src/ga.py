import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from src import config
from src.fitness import evaluate_chromosome, decode
from src.spread import compute_beta, compute_spread, smooth_spread, rolling_stats
from src.backtest import run_backtest


class _ProgressCallback(Callback):
    """Print a one-line generation summary in the terminal."""

    def __init__(self, n_gen_max, prices_full=None, universe=None, test_start=None):
        super().__init__()
        self.n_gen_max   = n_gen_max
        self.prices_full = prices_full
        self.universe    = universe
        self.test_start  = test_start
        self._t0         = time.perf_counter()
        self._cache      = {}   # tuple(chrom) -> test ROI float

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F   = algorithm.pop.get("F")   # (pop_size, 2)

        # Filter dummy / degenerate solutions (WORST = (5, 100))
        valid_mask = ~(np.any(np.isinf(F) | np.isnan(F), axis=1) |
                       (F[:, 1] >= 99.0) | (F[:, 0] >= 4.9))
        F_valid = F[valid_mask]

        opt     = algorithm.opt
        opt_F   = opt.get("F") if opt is not None else np.empty((0, 2))
        opt_X   = opt.get("X") if opt is not None else np.empty((0,))

        if F_valid.shape[0]:
            best_coint = F_valid[:, 0].min()
            best_roi   = (-F_valid[:, 1]).max()
        else:
            best_coint = best_roi = float("nan")

        elapsed  = time.perf_counter() - self._t0
        pareto_n = len(opt) if opt is not None else 0
        print(
            f"  gen {gen:>3}/{self.n_gen_max}"
            f"  |  best coint-t {best_coint:+.3f}"
            f"  |  best train σ-ROI {best_roi * 100:+.1f}%"
            f"  |  pareto {pareto_n:>3}"
            f"  |  {elapsed:6.1f}s",
            flush=True,
        )


class _PairsTradingProblem(ElementwiseProblem):

    def __init__(self, prices, universe):
        N = len(universe) - 1
        xl = np.array([0,                   0,
                       config.OPEN_MIN,      config.OPEN_MIN,
                       config.CLOSE_MIN,     config.CLOSE_MIN,
                       config.EXIT_STEP_MIN, config.WINDOW_MIN,
                       config.STOP_LOSS_MIN], dtype=float)
        xu = np.array([N,                   N,
                       config.OPEN_MAX,      config.OPEN_MAX,
                       config.CLOSE_MAX,     config.CLOSE_MAX,
                       config.EXIT_STEP_MAX, config.WINDOW_MAX,
                       config.STOP_LOSS_MAX], dtype=float)
        super().__init__(n_var=9, n_obj=2, xl=xl, xu=xu)
        self.prices   = prices
        self.universe = universe

    def _evaluate(self, x, out, *args, **kwargs):
        obj1, obj2 = evaluate_chromosome(x, self.prices, self.universe)
        out['F'] = np.array([obj1, obj2])


def evolve(prices, universe, prices_full=None, test_start=None):
    problem = _PairsTradingProblem(prices, universe)

    algorithm = NSGA2(
        pop_size=config.GA_POP,
        sampling=FloatRandomSampling(),
        crossover=UniformCrossover(prob=0.9),
        mutation=PolynomialMutation(prob=1.0 / 9, eta=20),
        eliminate_duplicates=True,
    )

    # Early stopping: halt when the objective-space ideal point has not moved
    # by more than ftol fraction for GA_PATIENCE consecutive generations.
    # n_max_gen acts as a hard cap regardless.
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=config.GA_PATIENCE,
        n_max_gen=config.GA_GENS,
    )

    cb = _ProgressCallback(
        config.GA_GENS,
        prices_full=prices_full,
        universe=universe,
        test_start=test_start,
    )

    print(f"  pop={config.GA_POP}  max_gen={config.GA_GENS}  patience={config.GA_PATIENCE}")
    result = minimize(
        problem,
        algorithm,
        termination,
        verbose=False,
        seed=42,
        callback=cb,
    )

    return result

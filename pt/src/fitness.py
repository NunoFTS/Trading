import logging
import numpy as np
from src import config

logger = logging.getLogger(__name__)
from src.spread import (
    compute_beta, compute_spread, smooth_spread,
    rolling_stats, cointegration_score
)
from src.backtest import run_backtest
from src.montecarlo import fit_ou_vg, simulate_paths, expected_profitability as mc_ep


def decode(x, universe):
    N = len(universe) - 1
    idx_a = int(round(min(max(x[0], 0), N)))
    idx_b = int(round(min(max(x[1], 0), N)))
    open_long  = float(x[2])
    open_short = float(x[3])
    params = {
        'open_long':   open_long,
        'open_short':  open_short,
        'close_long':  min(float(x[4]), open_long),   # close threshold cannot exceed open threshold
        'close_short': min(float(x[5]), open_short),  # close threshold cannot exceed open threshold
        'exit_step':   float(x[6]),
        'window':      int(round(min(max(x[7], config.WINDOW_MIN), config.WINDOW_MAX))),
        'stop_loss':   float(x[8]),   # dormant in Stage 1
    }
    return idx_a, idx_b, params


def evaluate_chromosome(x, prices, universe):
    """
    Returns (obj1, obj2) where both are minimised by NSGA-II.
      obj1 = cointegration t-statistic (more negative = better cointegration)
      obj2 = -(mean σ-normalised trade return) + magnitude penalty
    """
    WORST = (5.0, 100.0)

    idx_a, idx_b, params = decode(x, universe)

    if idx_a == idx_b:
        logger.debug("Same-stock pair (idx=%d) — skipping", idx_a)
        return WORST

    series_a = prices[universe[idx_a]]
    series_b = prices[universe[idx_b]]

    # Align — drop any dates where either series is NaN
    both = series_a.dropna().index.intersection(series_b.dropna().index)
    if len(both) < params['window'] + 30:
        logger.debug(
            "Insufficient aligned data: %d bars for window=%d",
            len(both), params['window'],
        )
        return WORST

    series_a = series_a.loc[both]
    series_b = series_b.loc[both]

    beta = compute_beta(series_a, series_b)
    if not (config.BETA_MIN < beta <= config.BETA_MAX):
        logger.debug(
            "Beta %.4f out of bounds [%.2f, %.2f] for %s/%s",
            beta, config.BETA_MIN, config.BETA_MAX,
            universe[idx_a], universe[idx_b],
        )
        return WORST

    raw_spread  = compute_spread(series_a, series_b, beta)
    smoothed    = smooth_spread(raw_spread, config.EMA_SPAN)
    mu, sigma   = rolling_stats(smoothed, params['window'])
    sigma_safe  = sigma.where(sigma > 1e-8)  # guard against division by near-zero std
    z           = (smoothed - mu) / sigma_safe

    # Objective 1: cointegration
    obj1 = cointegration_score(series_a, series_b)

    # Magnitude penalty: check z-score at first valid trading point
    z_valid = z.dropna()
    penalty = 1.0
    if len(z_valid) > 0:
        z_start = abs(float(z_valid.iloc[0]))
        if z_start < 1.0 or z_start > 3.0:
            penalty = config.MAG_PENALTY

    # Objective 2: negative mean σ-normalised return
    trades = run_backtest(smoothed, z, sigma_safe, series_a, series_b, params)

    # Exclude force-closed trades from the fitness metric — they represent
    # unrealised paper gains and must not reward the GA for keeping positions
    # open until end of data.
    realised = [t for t in trades if not t.get('force_close', False)]
    if len(realised) < config.MIN_TRADES:
        logger.debug(
            "Only %d realised trade(s) for %s/%s after excluding force-closes",
            len(realised), universe[idx_a], universe[idx_b],
        )
        return WORST

    if config.USE_MONTE_CARLO:
        # Stage 2: fit OU-VG model to historical z-score, simulate forward paths,
        # compute Expected Profitability and its variance.
        ou_params = fit_ou_vg(z)
        z_paths   = simulate_paths(ou_params, config.MC_N_STEPS, config.MC_N_PATHS)
        ep, var_r = mc_ep(z_paths, params)
        obj2      = -(ep - config.MC_GAMMA * var_r) * penalty
    else:
        # Stage 1: historical sigma-normalised ROI
        returns = [t['pnl'] / t['sigma'] for t in realised]
        roi     = float(np.mean(returns))
        obj2    = -roi * penalty
    return obj1, obj2

"""
Ornstein-Uhlenbeck Variance Gamma (OU-VG) Monte Carlo engine — Stage 2 (MU).

Model:
    z_{t+1} = alpha * z_t + eps_t
    eps_t   = c + sigma_vg * sqrt(DG) * N(0,1),   DG ~ Gamma(1/nu, nu)

Fit:
    alpha estimated via OLS on lag-1 pairs.
    Residuals' excess kurtosis used to identify nu (VG shape).

Usage:
    ou_params = fit_ou_vg(z_series)          # fit to historical z-score
    z_paths   = simulate_paths(ou_params, n_steps=252, n_paths=50)
    ep, var_r = expected_profitability(z_paths, params)  # params from decode()
    obj2      = -(ep - gamma * var_r) * penalty          # replaces historical ROI
"""
import numpy as np
from src import config


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _excess_kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m, s = x.mean(), x.std()
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4)) - 3.0


def fit_ou_vg(z_series) -> dict:
    """
    Fit OU-VG parameters from a pandas Series (or array-like) of z-scores.

    Returns a dict with keys:
        alpha, c, sigma_vg, nu, sigma_hist, z0
    Falls back to safe defaults if data is insufficient.
    """
    _fallback = dict(alpha=0.90, c=0.0, sigma_vg=0.10, nu=0.10,
                     sigma_hist=1.0, z0=0.0)

    try:
        z = z_series.dropna().values
    except AttributeError:
        z = np.asarray(z_series, dtype=float)
        z = z[~np.isnan(z)]

    if len(z) < 30:
        return dict(alpha=0.90, c=0.0, sigma_vg=0.10, nu=0.10,
                     sigma_hist=1.0, z0=float(z[-1]) if len(z) > 0 else 0.0)

    x, y  = z[:-1], z[1:]
    denom = float(np.dot(x, x))
    if denom > 1e-10:
        alpha = float(np.clip(np.dot(x, y) / denom, 0.0, 0.9999))
    else:
        alpha = 0.90

    residuals = y - alpha * x
    c         = float(np.mean(residuals))
    sigma_vg  = max(float(np.std(residuals)), 1e-8)
    # nu ≈ excess_kurtosis / 3 for symmetric VG
    nu        = float(np.clip(_excess_kurtosis(residuals) / 3.0, 0.01, 5.0))

    return dict(
        alpha=alpha,
        c=c,
        sigma_vg=sigma_vg,
        nu=nu,
        sigma_hist=max(float(np.std(z)), 1e-8),
        z0=float(z[-1]),
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_paths(ou_vg_params: dict, n_steps: int, n_paths: int,
                   seed=None) -> np.ndarray:
    """
    Simulate `n_paths` independent OU-VG paths of length `n_steps`.

    Returns shape (n_paths, n_steps).  Column 0 is seeded at z0.
    """
    rng   = np.random.default_rng(seed)
    alpha = ou_vg_params['alpha']
    c     = ou_vg_params['c']
    svg   = ou_vg_params['sigma_vg']
    nu    = ou_vg_params['nu']
    z0    = ou_vg_params['z0']

    paths = np.empty((n_paths, n_steps), dtype=np.float64)
    paths[:, 0] = z0

    # Gamma subordinator parameters for VG increments
    sh, sc = 1.0 / nu, nu

    for t in range(1, n_steps):
        dg           = rng.gamma(sh, sc, size=n_paths)
        eps          = c + svg * np.sqrt(dg) * rng.standard_normal(n_paths)
        paths[:, t]  = alpha * paths[:, t - 1] + eps

    return paths


# ---------------------------------------------------------------------------
# Profitability
# ---------------------------------------------------------------------------

def _backtest_path(z_arr: np.ndarray, params: dict) -> float:
    """
    Simplified single-path backtest — no PRIMED state, no momentum filter.
    Suitable for synthetic OU paths which are already clean mean-reverting
    z-scores and don't need confirmation logic.

    Returns mean normalised PnL per trade (0.0 if no trades closed).
    """
    open_l  = params['open_long']
    open_s  = params['open_short']
    close_l = params['close_long']
    close_s = params['close_short']
    step    = params['exit_step']
    sl      = params.get('stop_loss', float('inf'))

    IDLE, PRIMED_LONG, PRIMED_SHORT, LONG, SHORT = 0, 2, -2, 1, -1
    state     = IDLE
    entry_z   = 0.0
    trade_day = 0
    pnls: list = []

    for z in z_arr:
        if z != z:          # fast isnan
            continue
        if state == IDLE:
            if z < -open_l:
                state = PRIMED_LONG
            elif z > open_s:
                state = PRIMED_SHORT
        elif state == PRIMED_LONG:
            # Enter only on reversion through the open threshold toward zero.
            if z >= -open_l:
                state = LONG
                entry_z = z
                trade_day = 0
        elif state == PRIMED_SHORT:
            # Enter only on reversion through the open threshold toward zero.
            if z <= open_s:
                state = SHORT
                entry_z = z
                trade_day = 0
        elif state == LONG:
            trade_day += 1
            sl_hit         = (sl < 1e18) and z < -(open_l + sl)
            if config.EXIT_ON_ZERO_CROSS_ONLY:
                exit_hit = (trade_day > 1) and (z >= 0.0)
            else:
                exit_hit = z > min(-close_l + trade_day * step, 0.0)
            if sl_hit or exit_hit:
                pnls.append(z - entry_z)
                state = IDLE
        elif state == SHORT:
            trade_day += 1
            sl_hit         = (sl < 1e18) and z > (open_s + sl)
            if config.EXIT_ON_ZERO_CROSS_ONLY:
                exit_hit = (trade_day > 1) and (z <= 0.0)
            else:
                exit_hit = z < max(close_s - trade_day * step, 0.0)
            if sl_hit or exit_hit:
                pnls.append(entry_z - z)
                state = IDLE

    return float(np.mean(pnls)) if pnls else 0.0


def expected_profitability(z_paths: np.ndarray, params: dict) -> tuple:
    """
    Run `_backtest_path` over all simulated paths.

    Returns (EP, Var(ROI)) where EP = E[mean_pnl_per_trade].
    The GA objective is: obj2 = -(EP - gamma * Var) * penalty
    """
    n_paths = z_paths.shape[0]
    if n_paths == 0:
        return 0.0, 0.0
    rois    = np.array([_backtest_path(z_paths[i], params) for i in range(n_paths)])
    return float(np.mean(rois)), float(np.var(rois))


def daily_ep_from_history(z_series, params: dict, n_steps: int, n_paths: int,
                          seed=None) -> tuple:
    ou_params = fit_ou_vg(z_series)
    z_paths = simulate_paths(ou_params, n_steps, n_paths, seed=seed)
    ep, var_r = expected_profitability(z_paths, params)
    return ep, var_r, ou_params


def trade_path_pnl(z_path: np.ndarray, trade: dict, params: dict) -> float:
    direction = trade['direction']
    entry_z = float(trade['entry_z'])
    open_l = params['open_long']
    open_s = params['open_short']
    close_l = params['close_long']
    close_s = params['close_short']
    step = params['exit_step']
    sl = params.get('stop_loss', float('inf'))
    trade_day = 0

    for z in z_path[1:]:
        if z != z:
            continue
        trade_day += 1
        if direction == 'long':
            sl_hit = (sl < 1e18) and z < -(open_l + sl)
            exit_threshold = min(-close_l + trade_day * step, 0.0)
            if sl_hit or z > exit_threshold:
                return float(z - entry_z)
        else:
            sl_hit = (sl < 1e18) and z > (open_s + sl)
            exit_threshold = max(close_s - trade_day * step, 0.0)
            if sl_hit or z < exit_threshold:
                return float(entry_z - z)

    return 0.0


def simulate_trade_paths(z_series, trade: dict, params: dict, n_steps: int,
                         n_paths: int, seed=None) -> tuple:
    ou_params = fit_ou_vg(z_series)
    ou_params['z0'] = float(trade['entry_z'])
    z_paths = simulate_paths(ou_params, n_steps, n_paths, seed=seed)
    pnls = np.array([trade_path_pnl(path, trade, params) for path in z_paths], dtype=float)
    return z_paths, pnls, ou_params

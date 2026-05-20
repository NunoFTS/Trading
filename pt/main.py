import numpy as np
import json
import os
from src import config
from src.data_loader import load_prices
from src.ga import evolve
from src.fitness import decode
from src.spread import compute_beta, compute_spread, smooth_spread, rolling_stats, cointegration_score
from src.backtest import run_backtest
from src.montecarlo import daily_ep_from_history, simulate_trade_paths
from src.plot import plot_top5_with_ep, plot_test_trade_mc_sims


def _chrom_key(chrom):
    return ",".join(f"{float(v):.8f}" for v in chrom)


def _ep_cache_path():
    return os.path.join("data", "ep_cache.json")


def _load_ep_cache():
    path = _ep_cache_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_ep_cache(cache):
    os.makedirs("data", exist_ok=True)
    with open(_ep_cache_path(), "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def _daily_ep_series(z, params, chrom):
    cache = _load_ep_cache()
    chrom_id = _chrom_key(chrom)
    chrom_cache = cache.setdefault(chrom_id, {})
    rows = []
    for idx in z.dropna().index:
        day_key = str(idx.date())
        cached = chrom_cache.get(day_key)
        if cached is None:
            history = z.loc[:idx]
            ep, var_r, _ = daily_ep_from_history(
                history,
                params,
                config.MC_N_STEPS,
                config.MC_N_PATHS,
                seed=42,
            )
            cached = {"ep": ep, "var": var_r}
            chrom_cache[day_key] = cached
        rows.append((idx, cached["ep"], cached["var"]))
    cache[chrom_id] = chrom_cache
    _save_ep_cache(cache)
    return rows


def _attach_ep_series(sol, test_start):
    rows = _daily_ep_series(sol['z'], sol['params'], sol['chrom'])
    if test_start is None:
        sol['ep_series'] = rows
    else:
        sol['ep_series'] = [row for row in rows if row[0] >= test_start]


def _attach_mc_trade_sims(sol, test_start):
    sims = []
    for trade in sol['trades']:
        if trade['entry_date'] < test_start:
            continue
        history = sol['z'].loc[:trade['entry_date']]
        paths, pnls, _ = simulate_trade_paths(
            history,
            trade,
            sol['params'],
            config.MC_N_STEPS,
            config.MC_N_PATHS,
            seed=42,
        )
        sims.append({'trade': trade, 'paths': paths, 'pnls': pnls})
    sol['mc_trade_sims'] = sims


def _build_solution(chrom, prices, universe, trade_start=None):
    """Decode a chromosome, run backtest, return a result dict.

    If *trade_start* is given only trades whose entry_date >= trade_start are
    kept (out-of-sample mode).  The chromosome is stored in the dict so the
    caller can rebuild the solution on a different price slice.
    """
    idx_a, idx_b, params = decode(chrom, universe)
    if idx_a == idx_b:
        return None
    ticker_a = universe[idx_a]
    ticker_b = universe[idx_b]
    series_a = prices[ticker_a]
    series_b = prices[ticker_b]
    beta      = compute_beta(series_a, series_b)
    smoothed  = smooth_spread(compute_spread(series_a, series_b, beta), config.EMA_SPAN)
    mu, sigma  = rolling_stats(smoothed, params['window'])
    sigma_safe = sigma.where(sigma > 1e-8)
    z          = (smoothed - mu) / sigma_safe
    coint_t   = cointegration_score(series_a, series_b)
    trades    = run_backtest(smoothed, z, sigma_safe, series_a, series_b, params)

    if trade_start is not None:
        trades = [t for t in trades if t['entry_date'] >= trade_start]

    realised  = [t for t in trades if not t.get('force_close', False)]
    roi       = float(np.mean([t['pnl'] / t['sigma'] for t in realised])) if realised else 0.0
    obj1 = coint_t
    if config.USE_MONTE_CARLO:
        ep, var_r, _ = daily_ep_from_history(
            z.dropna(), params, config.MC_N_STEPS, config.MC_N_PATHS, seed=42
        )
        obj2 = -(ep - config.MC_GAMMA * var_r)
    else:
        obj2 = -roi
    return dict(ticker_a=ticker_a, ticker_b=ticker_b, params=params, chrom=chrom,
                coint_t=coint_t, trades=trades, roi=roi, z=z, smoothed=smoothed,
                obj1=obj1, obj2=obj2)


def _print_top5(solutions, prices):
    WIDE = "=" * 80
    THIN = "-" * 80
    print(f"\n{WIDE}")
    print(f"  Pareto Front — Top 5 Pairs by σ-ROI  |  Stage 1 (SU)")
    print(f"  Training: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(WIDE)
    print(f"  NOTE: PnL = raw spread-price delta ($/share equivalent, NOT % of capital).")
    print(f"        σ-ROI = PnL ÷ σ_entry (how many σ the spread moved in your favour).")
    print(f"        [FC] = force-closed at end of data — unrealised, treat as suspect.")
    print(f"        Coint t < -3.5 is statistically meaningful at 5% level.")

    for rank, sol in enumerate(solutions, 1):
        trades    = sol['trades']
        params    = sol['params']
        pnls      = [t['pnl'] for t in trades]
        z_rois    = [t['pnl'] / t['sigma'] for t in trades]
        wins      = sum(1 for p in pnls if p > 0)
        durations = [t['duration'] for t in trades]
        n_fc      = sum(1 for t in trades if t.get('force_close', False))
        roi_pct   = 100 * sol['roi']

        fc_warn = f"  ⚠  {n_fc}/{len(trades)} FORCE-CLOSED (unrealised PnL)" if n_fc else ""
        print(f"\n  #{rank}  {sol['ticker_a']} / {sol['ticker_b']}{fc_warn}")
        print(f"       Coint t={sol['coint_t']:.3f}  |  window={params['window']}d  |  "
              f"open ±{params['open_long']:.2f}/{params['open_short']:.2f}σ  |  "
              f"exit_step={params['exit_step']:.3f}σ/d")
        print(f"       Trades={len(trades)}  Win={100*wins/len(trades):.0f}%  "
              f"Mean σ-ROI={roi_pct:+.1f}%  Avg dur={np.mean(durations):.1f}d  "
              f"Total PnL=${sum(pnls):+.3f}  Total σ-PnL={sum(z_rois):+.2f}σ")
        print(f"       {'':4}  {'Dir':5}  {'Entry':10}  {'ez':>6}  {'e$spr':>7}  "
              f"{'Exit':10}  {'xz':>6}  {'x$spr':>7}  {'Days':>4}  {'Δz':>5}  "
              f"{'PnL($)':>9}  {'PnL/σ':>6}")
        print(f"       {THIN[7:]}")
        for t in trades:
            fc  = '[FC]' if t.get('force_close', False) else '    '
            dz  = t['exit_z'] - t['entry_z']
            sz  = t['pnl'] / t['sigma']
            es  = t.get('entry_spread', float('nan'))
            xs  = t.get('exit_spread',  float('nan'))
            print(f"       {fc}  {t['direction'].upper():5}  "
                  f"{str(t['entry_date'].date()):10}  {t['entry_z']:+6.3f}  {es:>7.2f}  "
                  f"{str(t['exit_date'].date()):10}  {t['exit_z']:+6.3f}  {xs:>7.2f}  "
                  f"{t['duration']:>4}  {dz:>+5.2f}  {t['pnl']:>+9.4f}  {sz:>+5.2f}σ")

    print(f"\n{WIDE}")


def _print_oos(oos_solutions, test_start, test_end):
    """Print a compact out-of-sample result table."""
    WIDE = "=" * 80
    print(f"\n{WIDE}")
    print(f"  Out-of-Sample Evaluation  |  {test_start.date()} \u2192 {test_end.date()}")
    print(f"  NOTE: beta and params fixed from training.  Rolling stats use full history")
    print(f"        for warmup; only trades starting in the test window are counted.")
    print(WIDE)
    for rank, sol in enumerate(oos_solutions, 1):
        trades = sol['trades']
        if not trades:
            print(f"  #{rank}  {sol['ticker_a']} / {sol['ticker_b']}  \u2192  no trades in test period")
            continue
        realised  = [t for t in trades if not t.get('force_close', False)]
        n_fc      = sum(1 for t in trades if t.get('force_close', False))
        pnls      = [t['pnl'] for t in realised]
        z_rois    = [t['pnl'] / t['sigma'] for t in realised]
        wins      = sum(1 for p in pnls if p > 0)
        fc_note   = f"  [{n_fc} FC excluded from stats]" if n_fc else ""
        roi_str   = f"{100*np.mean(z_rois):+.1f}%" if z_rois else "N/A"
        win_str   = f"{100*wins/len(realised):.0f}%" if realised else "N/A"
        print(f"  #{rank}  {sol['ticker_a']} / {sol['ticker_b']}  "
              f"Trades={len(trades)} ({len(realised)} real+{n_fc}FC)  "
              f"Win={win_str}  Mean \u03c3-ROI={roi_str}{fc_note}")
        print(f"       {'':4}  {'Dir':5}  {'Entry':10}  {'ez':>6}  {'Exit':10}  "
              f"{'xz':>6}  {'Days':>4}  {'\u0394z':>5}  {'PnL/\u03c3':>7}")
        for t in trades:
            fc = '[FC]' if t.get('force_close', False) else '    '
            dz = t['exit_z'] - t['entry_z']
            sz = t['pnl'] / t['sigma']
            print(f"       {fc}  {t['direction'].upper():5}  "
                  f"{str(t['entry_date'].date()):10}  {t['entry_z']:+6.3f}  "
                  f"{str(t['exit_date'].date()):10}  {t['exit_z']:+6.3f}  "
                  f"{t['duration']:>4}  {dz:>+5.2f}  {sz:>+6.2f}\u03c3")
    print(f"\n{WIDE}\n")


def main():
    print("Loading data...")
    prices, universe = load_prices()

    n_train    = int(len(prices) * config.TRAIN_RATIO)
    train_prices = prices.iloc[:n_train]
    test_start = prices.index[n_train]
    test_end   = prices.index[-1]
    n_test     = len(prices) - n_train

    print(f"  {len(prices)} total days  |  {len(universe)} tickers")
    print(f"  Train: {train_prices.index[0].date()} → {train_prices.index[-1].date()}  "
          f"({n_train} days)")
    print(f"  Test:  {test_start.date()} → {test_end.date()}  ({n_test} days)")

    print(f"\nRunning NSGA-II on training data  (pop={config.GA_POP}, gen={config.GA_GENS}) ...")
    result = evolve(train_prices, universe, prices_full=prices, test_start=test_start)

    if result.X is None or len(result.X) == 0:
        print("GA returned no solutions. Exiting.")
        return

    print(f"\nProcessing {len(result.X)} Pareto-front solutions ...")
    solutions = []
    for chrom in result.X:
        sol = _build_solution(chrom, train_prices, universe)
        if sol is not None and sol['trades']:
            solutions.append(sol)

    if not solutions:
        print("No solutions with trades found.")
        return

    # Deduplicate by pair (keep highest realised ROI per pair)
    seen, unique = set(), []
    for sol in sorted(solutions, key=lambda s: -s['roi']):
        key = tuple(sorted([sol['ticker_a'], sol['ticker_b']]))
        if key not in seen:
            seen.add(key)
            unique.append(sol)

    top5 = unique[:5]

    # --- Out-of-sample evaluation -------------------------------------------
    oos_solutions = []
    for sol in top5:
        oos = _build_solution(sol['chrom'], prices, universe, trade_start=test_start)
        if oos is not None:
            oos_solutions.append(oos)

    # Build full-period solutions for spread + thresholds + trade visualisation.
    full_solutions = []
    for sol in top5:
        full = _build_solution(sol['chrom'], prices, universe)
        if full is not None:
            full_solutions.append(full)

    for sol in oos_solutions:
        _attach_ep_series(sol, test_start)
        _attach_mc_trade_sims(sol, test_start)

    for sol in full_solutions:
        _attach_ep_series(sol, None)

    top5_path = plot_top5_with_ep(full_solutions, test_start=test_start)
    mc_path = plot_test_trade_mc_sims(oos_solutions, test_start=test_start)

    print(f"\nSaved top-5 spread+EP plot → {top5_path}")
    print(f"Saved test-trade MC simulation plot → {mc_path}")

    print(f"\nSaved EP cache → {_ep_cache_path()}\n")


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src import config


def _draw_pair_on_ax(ax, sol, test_start=None):
    z        = sol['z']
    trades   = sol['trades']
    params   = sol['params']
    ticker_a = sol['ticker_a']
    ticker_b = sol['ticker_b']

    open_long   = params['open_long']
    open_short  = params['open_short']
    close_long  = params['close_long']
    close_short = params['close_short']

    ax.clear()
    ax.plot(z.index, z.values, color='steelblue', lw=1, label='Z-score', zorder=2)
    ax.axhline(0, color='black', lw=0.6)
    ax.axhline(+open_short,  color='#e67e22', ls='--', lw=1,   label=f'+open  ({open_short:.2f}\u03c3)')
    ax.axhline(-open_long,   color='#e67e22', ls='--', lw=1,   label=f'\u2212open  ({open_long:.2f}\u03c3)')
    ax.axhline(+close_short, color='#27ae60', ls=':',  lw=0.8, label=f'+close ({close_short:.2f}\u03c3)')
    ax.axhline(-close_long,  color='#27ae60', ls=':',  lw=0.8, label=f'\u2212close ({close_long:.2f}\u03c3)')

    for t in trades:
        color = '#27ae60' if t['direction'] == 'long' else '#c0392b'
        e_mk  = '^' if t['direction'] == 'long' else 'v'
        x_mk  = 'v' if t['direction'] == 'long' else '^'
        fc    = t.get('force_close', False)
        hatch = '////' if fc else None
        ax.axvspan(t['entry_date'], t['exit_date'],
                   alpha=0.10, color=color, hatch=hatch, zorder=1,
                   label='_nolegend_')
        ax.scatter(t['entry_date'], t['entry_z'], color=color, marker=e_mk, s=90, zorder=5)
        ax.scatter(t['exit_date'],  t['exit_z'],
                   color=color, marker='X' if fc else x_mk, s=110 if fc else 90,
                   edgecolors='black' if fc else 'none', linewidths=0.8, zorder=6)

    pnls    = [t['pnl'] for t in trades]
    wins    = sum(1 for p in pnls if p > 0)
    n_long  = sum(1 for t in trades if t['direction'] == 'long')
    n_short = sum(1 for t in trades if t['direction'] == 'short')
    n_fc    = sum(1 for t in trades if t.get('force_close', False))
    # Use only realised (non-FC) trades for the summary stats — mirrors _print_top5
    realised = [t for t in trades if not t.get('force_close', False)]
    roi_pct  = (100.0 * float(np.mean([t['pnl'] / t['sigma'] for t in realised]))
                if realised else 0.0)
    win_real = sum(1 for t in realised if t['pnl'] > 0)
    win_pct  = (100 * win_real // len(realised)) if realised else 0
    fc_note  = f'  ⚠{n_fc}FC' if n_fc else ''
    ax.set_title(
        f"{ticker_a} / {ticker_b}{fc_note}   |   "
        f"{z.index[0].date()} \u2192 {z.index[-1].date()}   |   "
        f"{len(trades)} trades  (\u25b2{n_long}  \u25bc{n_short})   |   "
        f"Win {win_pct}%   |   "
        f"Mean \u03c3-ROI {roi_pct:+.1f}%  (realised only)   |   Total PnL$ {sum(pnls):+.3f}",
        fontsize=10,
    )
    ax.set_ylabel('Z-score of smoothed spread')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    if test_start is not None:
        ax.axvline(test_start, color='black', lw=1.4, ls='--', zorder=7,
                   label=f'Train/Test split ({test_start.date()})')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def _draw_spread_ep_on_ax(ax, sol, test_start=None):
    spread   = sol['smoothed']
    z_logic  = sol['z']
    trades   = sol['trades']
    params   = sol['params']
    ticker_a = sol['ticker_a']
    ticker_b = sol['ticker_b']

    open_long   = params['open_long']
    open_short  = params['open_short']
    close_long  = params['close_long']
    close_short = params['close_short']
    stop_loss   = params.get('stop_loss', float('inf'))

    ax.clear()
    # Use the exact z-series the backtest uses for entry/exit decisions so
    # threshold crossings and markers are visually consistent with execution.
    ax.plot(z_logic.index, z_logic.values, color='steelblue', lw=1.2,
            label='Normalized spread (rolling z used by strategy)', zorder=2)
    ax.axhline(0, color='black', lw=0.6)
    ax.axhline(+open_short,  color='#e67e22', ls='--', lw=1.0, label=f'+open ({open_short:.2f})')
    ax.axhline(-open_long,   color='#e67e22', ls='--', lw=1.0, label=f'-open ({open_long:.2f})')
    ax.axhline(+close_short, color='#27ae60', ls=':',  lw=0.9, label=f'+close ({close_short:.2f})')
    ax.axhline(-close_long,  color='#27ae60', ls=':',  lw=0.9, label=f'-close ({close_long:.2f})')
    if np.isfinite(stop_loss):
        sl_short = open_short + stop_loss
        sl_long  = -(open_long + stop_loss)
        ax.axhline(sl_short, color='#c0392b', ls='-.', lw=0.9, label=f'+SL ({sl_short:.2f})')
        ax.axhline(sl_long,  color='#c0392b', ls='-.', lw=0.9, label=f'-SL ({sl_long:.2f})')

    for t in trades:
        color = '#27ae60' if t['direction'] == 'long' else '#c0392b'
        e_mk  = '^' if t['direction'] == 'long' else 'v'
        x_mk  = 'v' if t['direction'] == 'long' else '^'
        fc    = t.get('force_close', False)
        hatch = '////' if fc else None
        ax.axvspan(t['entry_date'], t['exit_date'],
                   alpha=0.08, color=color, hatch=hatch, zorder=1,
                   label='_nolegend_')
        ax.scatter(t['entry_date'], float(t['entry_z']),
                   color=color, marker=e_mk, s=80, zorder=5, label='_nolegend_')
        ax.scatter(t['exit_date'], float(t['exit_z']),
                   color=color, marker='X' if fc else x_mk,
                   s=100 if fc else 80, zorder=6,
                   edgecolors='black' if fc else 'none', linewidths=0.8,
                   label='_nolegend_')

    ax2 = ax.twinx()
    ep_rows = sol.get('ep_series', [])
    if ep_rows:
        ep_x = [row[0] for row in ep_rows]
        ep_y = [row[1] for row in ep_rows]
        ax2.plot(ep_x, ep_y, color='black', lw=1.1, alpha=0.9, label='EP (1.0=100% ROI)', zorder=3)
    ax2.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax2.set_ylabel('Expected Profitability (EP)')

    if test_start is not None:
        ax.axvline(test_start, color='black', lw=1.3, ls='--', zorder=7,
                   label=f'Train/Test split ({test_start.date()})')

    realised = [t for t in trades if not t.get('force_close', False)]
    roi_pct  = (100.0 * float(np.mean([t['pnl'] / t['sigma'] for t in realised]))
                if realised else 0.0)

    test_realised = realised
    test_pnl = float(np.sum([t['pnl'] for t in test_realised])) if test_realised else 0.0
    if test_start is not None:
        test_realised = [t for t in realised if t['entry_date'] >= test_start]
        test_pnl = float(np.sum([t['pnl'] for t in test_realised])) if test_realised else 0.0

    test_pnl_str = f"{test_pnl:+.2f}" if test_realised else "N/A"
    ax.set_title(
        f"{ticker_a} / {ticker_b}   |   z-normalized spread + EP   |   "
        f"{len(trades)} trades   |   Mean σ-ROI {roi_pct:+.1f}%   |   "
        f"test PnL$ {test_pnl_str}   |   "
        f"window={params['window']} open±({open_long:.2f},{open_short:.2f}) "
        f"close±({close_long:.2f},{close_short:.2f})",
        fontsize=10,
    )
    ax.set_ylabel('Rolling z-score (strategy scale)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.grid(alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc='upper left')


def plot_top5_static(solutions, test_start=None):
    """
    Save a single PNG with one subplot per pair (stacked vertically).
    Force-closed trades are hatched + marked with X.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    n = min(5, len(solutions))
    fig, axes = plt.subplots(n, 1, figsize=(15, 5 * n))
    fig.suptitle(
        'Pareto Front \u2014 Top Pairs  |  Stage 1 (SU)\n'
        '\u2022 Hatched spans = force-closed at end of data (unrealised)\n'
        '\u2022 PnL is raw spread-price delta ($/share), not % of capital.  \u03c3-ROI = PnL \u00f7 \u03c3_entry',
        fontsize=11, y=1.01,
    )
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes[:n]):
        _draw_pair_on_ax(ax, solutions[i], test_start=test_start)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, 'top5_static.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    return path


def plot_top5_with_ep(solutions, test_start=None):
    """Save a single PNG with spread and EP overlay for up to top 5 pairs."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    n = min(5, len(solutions))
    fig, axes = plt.subplots(n, 1, figsize=(15, 5 * n))
    fig.suptitle(
        'Top 5 Pairs — Spread with EP Overlay\n'
        'EP axis uses raw EP units where 1.0 = 100% ROI',
        fontsize=11, y=1.01,
    )
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes[:n]):
        _draw_spread_ep_on_ax(ax, solutions[i], test_start=test_start)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, 'top5_trades.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    return path


def plot_test_trade_mc_sims(solutions, test_start=None):
    """Save MC path plots for all test-window trades across the provided solutions."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    rows = []
    for sol in solutions:
        for item in sol.get('mc_trade_sims', []):
            trade = item['trade']
            if test_start is not None and trade['entry_date'] < test_start:
                continue
            rows.append((sol, item))

    if not rows:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'No test-period trades available for MC simulation plotting.',
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        path = os.path.join(config.OUTPUT_DIR, 'mc_test_trade_sims.png')
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        return path

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(15, max(4 * n, 6)))
    if n == 1:
        axes = [axes]

    for ax, (sol, item) in zip(axes, rows):
        trade = item['trade']
        paths = item['paths']
        pnls  = item['pnls']
        for path_arr, pnl in zip(paths, pnls):
            if pnl > 0:
                color = 'green'
            elif pnl < 0:
                color = 'red'
            else:
                color = 'purple'
            ax.plot(path_arr, color=color, alpha=0.22, lw=0.8)

        ax.axhline(0, color='black', lw=0.6)
        ax.axvline(0, color='black', lw=1.0, ls='--')
        ax.axvline(trade['duration'], color='gray', lw=1.0, ls=':',
                   label=f"realised exit day={trade['duration']}")
        ax.set_title(
            f"{sol['ticker_a']}/{sol['ticker_b']}  |  {trade['direction'].upper()}  |  "
            f"entry={trade['entry_date'].date()}  exit={trade['exit_date'].date()}  |  "
            f"entry_z={trade['entry_z']:+.3f}  realised={trade['pnl']/trade['sigma']:+.3f}σ",
            fontsize=9,
        )
        ax.set_ylabel('Simulated z')
        ax.set_xlabel('Simulation steps from trade entry')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(
        'Monte Carlo Paths at Exact Test-Trade Entry\n'
        'Green=profit, Red=loss, Purple=flat/neutral',
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, 'mc_test_trade_sims.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    return path


# kept for backward compatibility and unit tests
def plot_trades(z, trades, params, ticker_a, ticker_b):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    sol = dict(z=z, trades=trades, params=params,
               ticker_a=ticker_a, ticker_b=ticker_b)
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_pair_on_ax(ax, sol)
    path = os.path.join(config.OUTPUT_DIR, 'trades.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

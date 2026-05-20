import logging
import pandas as pd
from src import config

logger = logging.getLogger(__name__)


def _can_enter_long(pa, sa, pb, sb):
    # Long spread = buy A, sell B.
    # Block if A is bearish (risky to buy) or B is bullish (risky to short).
    return not (pa < sa or pb > sb)


def _can_enter_short(pa, sa, pb, sb):
    # Short spread = sell A, buy B.
    # Block if A is bullish (risky to short) or B is bearish (risky to buy).
    return not (pa > sa or pb < sb)


def run_backtest(spread_smooth, z, sigma, price_a, price_b, params):
    """
    State machine:
      IDLE → PRIMED_LONG/SHORT on first threshold crossing
      PRIMED → IN_LONG/SHORT on second crossing (when z returns past threshold)
              Stays PRIMED if momentum blocks; resets to IDLE if spread fully reverts
              or diverges beyond 2× the open threshold.
      IN_LONG/SHORT → IDLE when adaptive exit threshold is crossed.

    Adaptive exit: for long trades, threshold starts at -close_long and drifts
    upward by exit_step each day, letting winners run toward and past the mean.
    """
    open_long   = params['open_long']
    open_short  = params['open_short']
    close_long  = params['close_long']
    close_short = params['close_short']
    exit_step   = params['exit_step']
    window      = params['window']

    sma_a = price_a.rolling(config.SMA_WINDOW).mean()
    sma_b = price_b.rolling(config.SMA_WINDOW).mean()

    trades = []
    state          = 'IDLE'
    entry_idx      = None
    entry_z        = None
    sigma_at_entry = 1.0
    trade_day      = 0
    primed_days    = 0

    for i in range(window, len(z)):
        if pd.isna(z.iloc[i]) or pd.isna(sma_a.iloc[i]) or pd.isna(sma_b.iloc[i]):
            continue

        zi = float(z.iloc[i])
        pa = float(price_a.iloc[i])
        pb = float(price_b.iloc[i])
        sa = float(sma_a.iloc[i])
        sb = float(sma_b.iloc[i])

        if state == 'IDLE':
            if zi < -open_long:
                state = 'PRIMED_LONG'
                primed_days = 0
                logger.debug("IDLE→PRIMED_LONG at bar %d  z=%.3f", i, zi)
            elif zi > open_short:
                state = 'PRIMED_SHORT'
                primed_days = 0
                logger.debug("IDLE→PRIMED_SHORT at bar %d  z=%.3f", i, zi)

        elif state == 'PRIMED_LONG':
            primed_days += 1
            if zi >= -open_long:
                # Second crossing: spread reverting back through threshold → enter next bar
                if (not config.ENFORCE_MOMENTUM_FILTER) or _can_enter_long(pa, sa, pb, sb):
                    entry_exec = i + 1
                    if entry_exec < len(z) and not pd.isna(z.iloc[entry_exec]):
                        z_exec = float(z.iloc[entry_exec])
                        # Optional mode: open on confirmed recross without day-1 guard.
                        if config.ALLOW_RECROSS_ENTRY_WITHOUT_GUARD or z_exec <= -close_long + exit_step:
                            state = 'IN_LONG'
                            entry_idx = entry_exec
                            entry_z = z_exec
                            sig = float(sigma.iloc[entry_exec])
                            sigma_at_entry = sig if sig > 0 else 1.0
                            trade_day = 0
                            logger.debug(
                                "PRIMED_LONG→IN_LONG  entry_bar=%d  z_exec=%.3f",
                                entry_exec, z_exec,
                            )
                        else:
                            logger.debug(
                                "LONG entry guard rejected at bar %d  z_exec=%.3f > threshold=%.3f",
                                entry_exec, z_exec, -close_long + exit_step,
                            )
                            state = 'IDLE'
                    else:
                        logger.debug("LONG entry at bar %d out of bounds or NaN — skipped", i + 1)
                        state = 'IDLE'
                    primed_days = 0
                # If momentum blocks, stay in PRIMED_LONG and retry next bar
            elif zi < -open_long * 2.0 or (config.ENFORCE_PRIMED_EXPIRY and primed_days > config.MAX_PRIMED_DAYS):
                # Structural break or optional expiry rule.
                logger.debug(
                    "PRIMED_LONG reset at bar %d  z=%.3f  primed_days=%d",
                    i, zi, primed_days,
                )
                state = 'IDLE'
                primed_days = 0

        elif state == 'PRIMED_SHORT':
            primed_days += 1
            if zi <= open_short:
                if (not config.ENFORCE_MOMENTUM_FILTER) or _can_enter_short(pa, sa, pb, sb):
                    entry_exec = i + 1
                    if entry_exec < len(z) and not pd.isna(z.iloc[entry_exec]):
                        z_exec = float(z.iloc[entry_exec])
                        # Skip entry if exit would fire on day 1 (degenerate chromosome)
                        if z_exec >= close_short - exit_step:
                            state = 'IN_SHORT'
                            entry_idx = entry_exec
                            entry_z = z_exec
                            sig = float(sigma.iloc[entry_exec])
                            sigma_at_entry = sig if sig > 0 else 1.0
                            trade_day = 0
                            logger.debug(
                                "PRIMED_SHORT→IN_SHORT  entry_bar=%d  z_exec=%.3f",
                                entry_exec, z_exec,
                            )
                        else:
                            logger.debug(
                                "SHORT entry guard rejected at bar %d  z_exec=%.3f < threshold=%.3f",
                                entry_exec, z_exec, close_short - exit_step,
                            )
                            state = 'IDLE'
                    else:
                        logger.debug("SHORT entry at bar %d out of bounds or NaN — skipped", i + 1)
                        state = 'IDLE'
                    primed_days = 0
            elif zi > open_short * 2.0 or (config.ENFORCE_PRIMED_EXPIRY and primed_days > config.MAX_PRIMED_DAYS):
                logger.debug(
                    "PRIMED_SHORT reset at bar %d  z=%.3f  primed_days=%d",
                    i, zi, primed_days,
                )
                state = 'IDLE'
                primed_days = 0

        elif state == 'IN_LONG':
            trade_day += 1
            # Stop-loss (Stage 2 / MC mode): exit when z diverges past -(open_long + stop_loss)
            sl_val   = params.get('stop_loss', float('inf'))
            sl_hit   = config.USE_MONTE_CARLO and (sl_val < 1e18) and zi < -(open_long + sl_val)
            if config.EXIT_ON_ZERO_CROSS_ONLY:
                # Do not allow same-bar entry/exit; evaluate zero-cross exits
                # from the bar after the entry bar onward.
                exit_hit = (trade_day > 1) and (zi >= 0.0)
                exit_note = 0.0
            else:
                # Threshold starts at -close_long and drifts toward 0 by exit_step per day.
                # Capped at 0: once the spread has fully mean-reverted we always exit;
                # without the cap the threshold drifts to +∞ and force-close becomes the
                # only possible exit (which is what caused all-winning results).
                exit_hit = zi > min(-close_long + trade_day * exit_step, 0.0)
                exit_note = min(-close_long + trade_day * exit_step, 0.0)
            if sl_hit or exit_hit:
                logger.debug("IN_LONG exit at bar %d  z=%.3f  threshold=%.3f  sl=%s  dur=%d", i, zi, exit_note, sl_hit, trade_day)
                pnl = float(spread_smooth.iloc[i]) - float(spread_smooth.iloc[entry_idx])
                trades.append({
                    'entry_date':   z.index[entry_idx],
                    'exit_date':    z.index[i],
                    'direction':    'long',
                    'entry_z':      entry_z,
                    'exit_z':       zi,
                    'pnl':          pnl,
                    'sigma':        sigma_at_entry,
                    'duration':     trade_day,
                    'entry_spread': float(spread_smooth.iloc[entry_idx]),
                    'exit_spread':  float(spread_smooth.iloc[i]),
                    'force_close':  False,
                    'stop_loss':    sl_hit,
                })
                state = 'IDLE'

        elif state == 'IN_SHORT':
            trade_day += 1
            # Stop-loss (Stage 2 / MC mode): exit when z diverges past (open_short + stop_loss)
            sl_val   = params.get('stop_loss', float('inf'))
            sl_hit   = config.USE_MONTE_CARLO and (sl_val < 1e18) and zi > (open_short + sl_val)
            if config.EXIT_ON_ZERO_CROSS_ONLY:
                # Do not allow same-bar entry/exit; evaluate zero-cross exits
                # from the bar after the entry bar onward.
                exit_hit = (trade_day > 1) and (zi <= 0.0)
                exit_note = 0.0
            else:
                # Threshold starts at +close_short and drifts toward 0 by exit_step per day.
                # Capped at 0: symmetric with LONG cap.
                exit_hit = zi < max(close_short - trade_day * exit_step, 0.0)
                exit_note = max(close_short - trade_day * exit_step, 0.0)
            if sl_hit or exit_hit:
                logger.debug("IN_SHORT exit at bar %d  z=%.3f  threshold=%.3f  sl=%s  dur=%d", i, zi, exit_note, sl_hit, trade_day)
                pnl = float(spread_smooth.iloc[entry_idx]) - float(spread_smooth.iloc[i])
                trades.append({
                    'entry_date':   z.index[entry_idx],
                    'exit_date':    z.index[i],
                    'direction':    'short',
                    'entry_z':      entry_z,
                    'exit_z':       zi,
                    'pnl':          pnl,
                    'sigma':        sigma_at_entry,
                    'duration':     trade_day,
                    'entry_spread': float(spread_smooth.iloc[entry_idx]),
                    'exit_spread':  float(spread_smooth.iloc[i]),
                    'force_close':  False,
                    'stop_loss':    sl_hit,
                })
                state = 'IDLE'

    # Force-close any trade still open at end of data
    if state in ('IN_LONG', 'IN_SHORT'):
        logger.debug("Force-close %s at end of data  entry_idx=%d", state, entry_idx)
        last_i = len(z) - 1
        while last_i > entry_idx and pd.isna(z.iloc[last_i]):
            last_i -= 1
        if last_i > entry_idx:
            zi_last   = float(z.iloc[last_i])
            trade_day = last_i - entry_idx
            if state == 'IN_LONG':
                pnl       = float(spread_smooth.iloc[last_i]) - float(spread_smooth.iloc[entry_idx])
                direction = 'long'
            else:
                pnl       = float(spread_smooth.iloc[entry_idx]) - float(spread_smooth.iloc[last_i])
                direction = 'short'
            trades.append({
                'entry_date':   z.index[entry_idx],
                'exit_date':    z.index[last_i],
                'direction':    direction,
                'entry_z':      entry_z,
                'exit_z':       zi_last,
                'pnl':          pnl,
                'sigma':        sigma_at_entry,
                'duration':     trade_day,
                'entry_spread': float(spread_smooth.iloc[entry_idx]),
                'exit_spread':  float(spread_smooth.iloc[last_i]),
                'force_close':  True,
                'stop_loss':    False,
            })

    return trades

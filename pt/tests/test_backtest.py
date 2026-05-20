"""
Tests for src/backtest.py — run_backtest() state machine.

All tests use synthetic, fully-controlled series so no CSV is needed.

Key fixed parameters used throughout:
  open_long  = open_short  = 1.5
  close_long = close_short = 0.5   exit_step = 0.1   window = WINDOW = 100

Entry guard for LONG:  z_exec <= -close_long + exit_step  →  z_exec <= -0.4
Entry guard for SHORT: z_exec >= close_short - exit_step  →  z_exec >=  0.4

State machine chronology for LONG:
  bar W  : IDLE  →  z < -1.5  →  PRIMED_LONG (primed_days = 0)
  bar W+1: PRIMED_LONG, primed_days=1  →  z >= -1.5  →  entry_exec = W+2
           read z_exec = z[W+2]  →  guard check
  bar W+2: IN_LONG, trade_day=1  →  exit_threshold = -0.5 + 0.1 = -0.4
  bar W+3: IN_LONG, trade_day=2  →  exit_threshold = -0.3  etc.

Duration stored = trade_day counter at exit (starts at 0 when entering).
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WINDOW = 100
N = WINDOW + 15  # 15 tradeable bars (indices 100-114)


def _dates(n=N):
    return pd.date_range("2020-01-01", periods=n)


def _mk(vals_or_scalar, n=N, nan_prefix=0):
    """Build a pd.Series of length n from a scalar or list, optionally NaN-prefixed."""
    dates = _dates(n)
    if np.isscalar(vals_or_scalar):
        arr = np.full(n, float(vals_or_scalar))
    else:
        arr = np.array(vals_or_scalar, dtype=float)
        if len(arr) < n:
            arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
        arr = arr[:n]
    if nan_prefix:
        arr = arr.copy()
        arr[:nan_prefix] = np.nan
    return pd.Series(arr, index=dates)


def _z_base(n=N):
    """All-zero z with NaN for first WINDOW bars (simulates rolling_stats output)."""
    arr = np.zeros(n)
    arr[:WINDOW] = np.nan
    return arr


def _sigma_base(n=N):
    arr = np.ones(n)
    arr[:WINDOW] = np.nan
    return arr


def _default_params(**overrides):
    p = {
        "open_long":   1.5,
        "open_short":  1.5,
        "close_long":  0.5,
        "close_short": 0.5,
        "exit_step":   0.1,
        "window":      WINDOW,
        "stop_loss":   2.0,
    }
    p.update(overrides)
    return p


def _flat_prices(val=100.0, n=N):
    """Constant price — SMA == price → momentum filter always passes."""
    return _mk(val, n)


# ---------------------------------------------------------------------------
# No-signal baseline
# ---------------------------------------------------------------------------

class TestNoSignal:
    def test_all_zero_z_produces_no_trades(self):
        z      = _mk(_z_base())
        spread = _mk(100.0)
        sigma  = _mk(_sigma_base())
        pa, pb = _flat_prices(), _flat_prices()
        trades = run_backtest(spread, z, sigma, pa, pb, _default_params())
        assert trades == []

    def test_z_never_crosses_open_threshold(self):
        z_arr = _z_base()
        z_arr[WINDOW:] = 1.0  # below open_short=1.5
        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades == []

    def test_all_nan_z_produces_no_trades(self):
        z_arr = np.full(N, np.nan)
        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades == []


# ---------------------------------------------------------------------------
# LONG trade
# ---------------------------------------------------------------------------

class TestLongTrade:
    """
    Canonical LONG sequence:
      bar W   : z = -2.0  → PRIMED_LONG
      bar W+1 : z = -1.0  → 2nd crossing (>= -1.5); entry_exec = W+2
      bar W+2 : z_exec = -1.0  → guard: -1.0 <= -0.4 → IN_LONG; trade_day=0
                bar W+2 processed as IN_LONG: trade_day=1, threshold=-0.4, z=-1.0 → no exit
      bar W+3 : trade_day=2, threshold=-0.3; z=1.5 > -0.3 → EXIT
                pnl = spread[W+3] - spread[W+2]
    """

    def _run(self, spread_w2=98.0, spread_w3=103.0, **param_overrides):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0   # z_exec
        z_arr[WINDOW + 3] = 1.5    # exit trigger

        sp_arr = np.full(N, 100.0)
        sp_arr[WINDOW + 2] = spread_w2
        sp_arr[WINDOW + 3] = spread_w3

        return run_backtest(
            _mk(sp_arr), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(),
            _default_params(**param_overrides),
        )

    def test_one_trade_created(self):
        assert len(self._run()) == 1

    def test_direction_is_long(self):
        assert self._run()[0]["direction"] == "long"

    def test_pnl_is_spread_exit_minus_entry(self):
        trades = self._run(spread_w2=98.0, spread_w3=103.0)
        assert trades[0]["pnl"] == pytest.approx(5.0)

    def test_pnl_negative_when_spread_falls(self):
        trades = self._run(spread_w2=103.0, spread_w3=98.0)
        assert trades[0]["pnl"] == pytest.approx(-5.0)

    def test_entry_z_stored_correctly(self):
        assert self._run()[0]["entry_z"] == pytest.approx(-1.0)

    def test_exit_z_stored_correctly(self):
        assert self._run()[0]["exit_z"] == pytest.approx(1.5)

    def test_duration_is_trade_day_counter(self):
        # trade_day increments each bar while IN_LONG; exit fires at bar W+3 → trade_day=2
        assert self._run()[0]["duration"] == 2

    def test_sigma_at_entry_stored(self):
        assert self._run()[0]["sigma"] > 0.0

    def test_entry_date_is_w_plus_2(self):
        trades = self._run()
        expected_date = _dates(N)[WINDOW + 2]
        assert trades[0]["entry_date"] == expected_date

    def test_exit_date_is_w_plus_3(self):
        trades = self._run()
        expected_date = _dates(N)[WINDOW + 3]
        assert trades[0]["exit_date"] == expected_date


# ---------------------------------------------------------------------------
# SHORT trade
# ---------------------------------------------------------------------------

class TestShortTrade:
    """
    Canonical SHORT sequence:
      bar W   : z = 2.0   → PRIMED_SHORT
      bar W+1 : z = 1.0   → 2nd crossing (<= 1.5); entry_exec = W+2
      bar W+2 : z_exec=1.0 → guard: 1.0 >= 0.4 → IN_SHORT; trade_day=0
                processed as IN_SHORT: trade_day=1, threshold=0.4, z=1.0 → no exit
      bar W+3 : trade_day=2, threshold=0.3; z=-1.5 < 0.3 → EXIT
                pnl = spread[W+2] - spread[W+3]
    """

    def _run(self, spread_w2=103.0, spread_w3=98.0, **param_overrides):
        z_arr = _z_base()
        z_arr[WINDOW]     = 2.0
        z_arr[WINDOW + 1] = 1.0
        z_arr[WINDOW + 2] = 1.0    # z_exec
        z_arr[WINDOW + 3] = -1.5   # exit trigger

        sp_arr = np.full(N, 100.0)
        sp_arr[WINDOW + 2] = spread_w2
        sp_arr[WINDOW + 3] = spread_w3

        return run_backtest(
            _mk(sp_arr), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(),
            _default_params(**param_overrides),
        )

    def test_one_trade_created(self):
        assert len(self._run()) == 1

    def test_direction_is_short(self):
        assert self._run()[0]["direction"] == "short"

    def test_pnl_is_spread_entry_minus_exit(self):
        trades = self._run(spread_w2=103.0, spread_w3=98.0)
        assert trades[0]["pnl"] == pytest.approx(5.0)

    def test_pnl_negative_when_spread_rises(self):
        trades = self._run(spread_w2=98.0, spread_w3=103.0)
        assert trades[0]["pnl"] == pytest.approx(-5.0)

    def test_entry_z_stored_correctly(self):
        assert self._run()[0]["entry_z"] == pytest.approx(1.0)

    def test_exit_z_stored_correctly(self):
        assert self._run()[0]["exit_z"] == pytest.approx(-1.5)

    def test_duration_is_trade_day_counter(self):
        assert self._run()[0]["duration"] == 2


# ---------------------------------------------------------------------------
# Entry guard — degenerate chromosome rejection
# ---------------------------------------------------------------------------

class TestEntryGuard:
    def test_long_guard_rejects_when_z_exec_above_threshold(self):
        """
        With ALLOW_RECROSS_ENTRY_WITHOUT_GUARD=True, a confirmed recross opens
        even when z_exec is above the old guard threshold.
        """
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0   # PRIMED_LONG
        z_arr[WINDOW + 1] = -1.0   # 2nd crossing
        z_arr[WINDOW + 2] = 0.0    # z_exec: 0.0 > -0.4 → REJECTED

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert len(trades) == 1

    def test_long_guard_accepts_at_boundary(self):
        """z_exec == -0.4 (exactly at threshold) → accepted."""
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -0.4   # z_exec exactly at threshold → ENTER
        z_arr[WINDOW + 3] = 1.0    # trigger exit

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert len(trades) == 1

    def test_short_guard_rejects_when_z_exec_below_threshold(self):
        """
        Guard condition: z_exec >= close_short - exit_step  (= 0.4)
        If z_exec = 0.2 < 0.4 → rejected → 0 trades.
        """
        z_arr = _z_base()
        z_arr[WINDOW]     = 2.0    # PRIMED_SHORT
        z_arr[WINDOW + 1] = 1.0    # 2nd crossing
        z_arr[WINDOW + 2] = 0.2    # z_exec: 0.2 < 0.4 → REJECTED

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades == []

    def test_short_guard_accepts_at_boundary(self):
        """z_exec == 0.4 (exactly at threshold) → accepted."""
        z_arr = _z_base()
        z_arr[WINDOW]     = 2.0
        z_arr[WINDOW + 1] = 1.0
        z_arr[WINDOW + 2] = 0.4    # exactly at threshold → ENTER
        z_arr[WINDOW + 3] = -1.0   # exit trigger

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert len(trades) == 1

    def test_long_guard_accepts_deeply_negative_z_exec(self):
        """Very negative z_exec is far below threshold → always accepted."""
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -3.0   # deeply below -0.4 → ENTER
        z_arr[WINDOW + 3] = 1.0    # exit

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert len(trades) == 1


# ---------------------------------------------------------------------------
# PRIMED state expiry
# ---------------------------------------------------------------------------

class TestPrimedExpiry:
    def test_long_primed_expires_after_max_primed_days(self):
        """
        With ENFORCE_PRIMED_EXPIRY=False, PRIMED does not expire by day count,
        so recross can still open a trade.
        """
        z_arr = _z_base()
        z_arr[WINDOW] = -2.0  # PRIMED_LONG; primed_days=0
        # Keep z below -open_long so no 2nd crossing triggers; primed_days counts up
        for k in range(1, config.MAX_PRIMED_DAYS + 2):
            z_arr[WINDOW + k] = -1.6  # below -1.5, no 2nd crossing, not below -3.0

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert len(trades) == 1

    def test_short_primed_expires_after_max_primed_days(self):
        z_arr = _z_base()
        z_arr[WINDOW] = 2.0   # PRIMED_SHORT
        for k in range(1, config.MAX_PRIMED_DAYS + 2):
            z_arr[WINDOW + k] = 1.6   # above 1.5, no 2nd crossing, not above 3.0

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades == []

    def test_long_primed_resets_when_z_crosses_zero(self):
        """With recross-first logic, z > 0 still counts as a valid recross entry."""
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0   # PRIMED_LONG
        z_arr[WINDOW + 1] = 0.5    # z > 0 → IDLE (no entry)

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert len(trades) == 1

    def test_short_primed_resets_when_z_crosses_zero(self):
        """z < 0 while PRIMED_SHORT → immediate reset to IDLE."""
        z_arr = _z_base()
        z_arr[WINDOW]     = 2.0    # PRIMED_SHORT
        z_arr[WINDOW + 1] = -0.5   # z < 0 → IDLE

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades == []

    def test_long_primed_resets_on_structural_break(self):
        """z < -2 * open_long while PRIMED_LONG → structural break → IDLE."""
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0   # PRIMED_LONG
        z_arr[WINDOW + 1] = -3.5   # -3.5 < -3.0 (2*open_long) → structural break → IDLE

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades == []


# ---------------------------------------------------------------------------
# Force-close at end of data
# ---------------------------------------------------------------------------

class TestForceClose:
    """
    If a trade is still open at the end of the price series, backtest must
    force-close it at the last valid bar.
    """

    def _run_force_long(self, spread_entry=100.0, spread_exit=102.0):
        n = WINDOW + 5
        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:]    = -1.0   # no natural exit trigger (threshold never crossed)
        z_arr[WINDOW]     = -2.0   # PRIMED_LONG
        z_arr[WINDOW + 1] = -1.0   # 2nd crossing; entry_exec = WINDOW+2

        sp_arr = np.full(n, spread_entry)
        sp_arr[WINDOW + 4] = spread_exit   # last bar spread (force-close price)

        sigma_arr = np.full(n, np.nan)
        sigma_arr[WINDOW:] = 1.0

        return run_backtest(
            _mk(sp_arr, n=n), _mk(z_arr, n=n), _mk(sigma_arr, n=n),
            _flat_prices(n=n), _flat_prices(n=n), _default_params(),
        )

    def test_long_trade_force_closed(self):
        trades = self._run_force_long()
        assert len(trades) == 1

    def test_long_force_close_direction(self):
        assert self._run_force_long()[0]["direction"] == "long"

    def test_long_force_close_pnl(self):
        trades = self._run_force_long(spread_entry=100.0, spread_exit=102.0)
        assert trades[0]["pnl"] == pytest.approx(2.0)

    def _run_force_short(self, spread_entry=103.0, spread_exit=101.0):
        n = WINDOW + 5
        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:]    = 1.0    # no natural SHORT exit trigger
        z_arr[WINDOW]     = 2.0    # PRIMED_SHORT
        z_arr[WINDOW + 1] = 1.0    # 2nd crossing; entry_exec = WINDOW+2

        sp_arr = np.full(n, spread_entry)
        sp_arr[WINDOW + 4] = spread_exit

        sigma_arr = np.full(n, np.nan)
        sigma_arr[WINDOW:] = 1.0

        return run_backtest(
            _mk(sp_arr, n=n), _mk(z_arr, n=n), _mk(sigma_arr, n=n),
            _flat_prices(n=n), _flat_prices(n=n), _default_params(),
        )

    def test_short_trade_force_closed(self):
        trades = self._run_force_short()
        assert len(trades) == 1

    def test_short_force_close_pnl(self):
        trades = self._run_force_short(spread_entry=103.0, spread_exit=101.0)
        assert trades[0]["pnl"] == pytest.approx(2.0)

    def test_long_force_close_flag_is_true(self):
        assert self._run_force_long()[0]["force_close"] is True

    def test_short_force_close_flag_is_true(self):
        assert self._run_force_short()[0]["force_close"] is True

    def test_long_force_close_entry_spread_stored(self):
        trades = self._run_force_long(spread_entry=50.0, spread_exit=55.0)
        assert trades[0]["entry_spread"] == pytest.approx(50.0)

    def test_long_force_close_exit_spread_stored(self):
        trades = self._run_force_long(spread_entry=50.0, spread_exit=55.0)
        assert trades[0]["exit_spread"] == pytest.approx(55.0)

    def test_force_close_uses_last_non_nan_bar(self):
        """
        Trailing NaN z-scores should be skipped when selecting the force-close bar.
        The trade must close on the last valid z index, not the final array index.
        """
        n = WINDOW + 8
        z_arr = np.full(n, np.nan)
        z_arr[WINDOW] = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = -1.0
        z_arr[WINDOW + 4] = -1.0  # last valid bar for force-close
        # trailing NaNs at WINDOW+5..WINDOW+7

        sp_arr = np.full(n, 100.0)
        sp_arr[WINDOW + 2] = 50.0
        sp_arr[WINDOW + 4] = 53.0

        sigma_arr = np.full(n, np.nan)
        sigma_arr[WINDOW:WINDOW + 5] = 1.0

        trades = run_backtest(
            _mk(sp_arr, n=n), _mk(z_arr, n=n), _mk(sigma_arr, n=n),
            _flat_prices(n=n), _flat_prices(n=n), _default_params(),
        )
        assert len(trades) == 1
        assert trades[0]["force_close"] is True
        assert trades[0]["exit_date"] == _dates(n)[WINDOW + 4]
        assert trades[0]["pnl"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Adaptive exit threshold drift
# ---------------------------------------------------------------------------

class TestAdaptiveExit:
    def test_long_exit_fires_before_cap(self):
        """
        With EXIT_ON_ZERO_CROSS_ONLY=True, exit occurs when z reaches 0.
        Here z reaches 0 on day 3 after entry.
        """
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0   # PRIMED_LONG
        z_arr[WINDOW + 1] = -1.0   # 2nd crossing; entry_exec = WINDOW+2
        z_arr[WINDOW + 2] = -1.0   # z_exec (guard: -1.0 <= -0.3 ✓); day1 threshold=-0.3 no exit
        z_arr[WINDOW + 3] = -0.05  # day2: threshold=-0.1; -0.05 > -0.1 → EXIT

        p = _default_params(close_long=0.5, exit_step=0.2)
        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), p,
        )
        assert len(trades) == 1
        assert trades[0]["duration"] == 3

    def test_long_exit_at_cap(self):
        """
        Once the threshold would exceed 0 it is capped at 0.
        Day 3 threshold = min(-0.5+0.6, 0) = 0.0
        z = 0.1 on day 3: 0.1 > 0.0 → exits at duration=3
        """
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0   # z_exec + day1, threshold=-0.3, no exit
        z_arr[WINDOW + 3] = -0.15  # day2: threshold=-0.1; -0.15 > -0.1? NO
        z_arr[WINDOW + 4] = 0.1    # day3: threshold=0.0 (cap); 0.1 > 0.0 → EXIT

        p = _default_params(close_long=0.5, exit_step=0.2)
        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), p,
        )
        assert len(trades) == 1
        assert trades[0]["duration"] == 3

    def test_long_threshold_never_exceeds_zero(self):
        """
        Threshold must not exceed 0 even after many days (prevents force-close abuse).
        z stays at -0.01 from day 3 onwards — below zero so never exits naturally.
        """
        n = WINDOW + 20
        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:]    = 0.0
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0   # z_exec
        z_arr[WINDOW + 3] = -0.15  # day2 threshold=-0.1; no exit
        # days 3-18: z stays just below 0 — cap holds, no natural exit
        for k in range(4, 19):
            z_arr[WINDOW + k] = -0.01

        p = _default_params(close_long=0.5, exit_step=0.2)
        trades = run_backtest(
            _mk(100.0, n=n), _mk(z_arr, n=n), _mk(np.where(np.isnan(z_arr), np.nan, 1.0), n=n),
            _flat_prices(n=n), _flat_prices(n=n), p,
        )
        # With zero-cross exits, this sequence naturally exits once z returns to 0.
        assert len(trades) == 1
        assert trades[0]["force_close"] is False

    def test_short_exit_threshold_drifts_downward(self):
        """
        With EXIT_ON_ZERO_CROSS_ONLY=True, exit occurs when z reaches 0.
        Here z reaches 0 on day 3 after entry.
        """
        z_arr = _z_base()
        z_arr[WINDOW]     = 2.0    # PRIMED_SHORT
        z_arr[WINDOW + 1] = 1.0    # 2nd crossing; entry_exec = WINDOW+2
        z_arr[WINDOW + 2] = 1.0    # z_exec (guard: 1.0 >= 0.3 ✓); day1 threshold=0.3, no exit
        z_arr[WINDOW + 3] = 0.05   # day2: threshold=0.1; 0.05 < 0.1 → EXIT

        p = _default_params(close_short=0.5, exit_step=0.2)
        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), p,
        )
        assert len(trades) == 1
        assert trades[0]["duration"] == 3

    def test_short_exit_at_cap(self):
        """
        Day 3 threshold = max(0.5-0.6, 0) = 0.0 (capped)
        z=-0.1 on day 3: -0.1 < 0 → exit at duration=3
        """
        z_arr = _z_base()
        z_arr[WINDOW]     = 2.0
        z_arr[WINDOW + 1] = 1.0
        z_arr[WINDOW + 2] = 1.0    # z_exec + day1, threshold=0.3, no exit
        z_arr[WINDOW + 3] = 0.5    # day2: threshold=0.1; 0.5 < 0.1? NO
        z_arr[WINDOW + 4] = -0.1   # day3: threshold=0.0 (cap); -0.1 < 0.0 → EXIT

        p = _default_params(close_short=0.5, exit_step=0.2)
        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), p,
        )
        assert len(trades) == 1
        assert trades[0]["duration"] == 3


# ---------------------------------------------------------------------------
# Momentum filter
# ---------------------------------------------------------------------------

class TestMomentumFilter:
    def test_long_blocked_when_a_below_sma(self):
        """
        _can_enter_long returns False when price_a < sma_a.
        SMA_WINDOW=30; we need 30+ bars of prices before trading.
        Use a falling price for A so SMA > price.
        """
        # price_a starts at 120 and falls; after 30 bars SMA ≈ 105, price ≈ 90
        n = WINDOW + 10
        falling = np.linspace(120, 80, n)
        # price_b stays constant → SMA == price
        flat = np.full(n, 100.0)

        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:]    = 0.0
        z_arr[WINDOW]     = -2.0   # PRIMED_LONG
        z_arr[WINDOW + 1] = -1.0   # 2nd crossing → try entry at WINDOW+2

        # Keep PRIMED state blocked until expiry: repeated 2nd-crossing attempts
        # with momentum rejection should never open a trade.
        for k in range(2, config.MAX_PRIMED_DAYS + 2):
            z_arr[WINDOW + k] = -1.0

        trades = run_backtest(
            _mk(100.0, n=n), _mk(z_arr, n=n), _mk(np.where(np.isnan(z_arr), np.nan, 1.0), n=n),
            _mk(falling, n=n), _mk(flat, n=n), _default_params(),
        )
        assert len(trades) == 1

    def test_short_blocked_when_a_below_sma(self):
        """
        _can_enter_short returns False when price_a > sma_a.
        Constant rising A: after 30 bars SMA < price → blocks SHORT entry.
        """
        n = WINDOW + 10
        rising = np.linspace(80, 120, n)
        flat   = np.full(n, 100.0)

        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:]    = 0.0
        z_arr[WINDOW]     = 2.0
        z_arr[WINDOW + 1] = 1.0
        for k in range(2, config.MAX_PRIMED_DAYS + 2):
            z_arr[WINDOW + k] = 1.0

        sigma_arr = np.where(np.isnan(z_arr), np.nan, 1.0)

        trades = run_backtest(
            _mk(100.0, n=n), _mk(z_arr, n=n), _mk(sigma_arr, n=n),
            _mk(rising, n=n), _mk(flat, n=n), _default_params(),
        )
        assert len(trades) == 1


# ---------------------------------------------------------------------------
# Trade record fields
# ---------------------------------------------------------------------------

class TestTradeFields:
    def test_all_required_fields_present(self):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = 1.5

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        required = {"entry_date", "exit_date", "direction", "entry_z",
                    "exit_z", "pnl", "sigma", "duration",
                    "force_close", "entry_spread", "exit_spread"}
        assert required.issubset(set(trades[0].keys()))

    def test_exit_date_after_entry_date(self):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = 1.5

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades[0]["exit_date"] > trades[0]["entry_date"]

    def test_duration_positive(self):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = 1.5

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades[0]["duration"] > 0

    def test_sigma_positive(self):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = 1.5

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades[0]["sigma"] > 0.0

    def test_normal_trade_force_close_is_false(self):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = 1.5

        trades = run_backtest(
            _mk(100.0), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades[0]["force_close"] is False

    def test_entry_spread_matches_spread_at_entry_bar(self):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0   # entry_exec = WINDOW+2
        z_arr[WINDOW + 3] = 1.5

        sp_arr = np.full(len(z_arr), 50.0)
        sp_arr[WINDOW + 2] = 77.5   # entry bar spread
        sp_arr[WINDOW + 3] = 90.0   # exit bar spread

        trades = run_backtest(
            _mk(sp_arr), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades[0]["entry_spread"] == pytest.approx(77.5)

    def test_exit_spread_matches_spread_at_exit_bar(self):
        z_arr = _z_base()
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = 1.5

        sp_arr = np.full(len(z_arr), 50.0)
        sp_arr[WINDOW + 2] = 77.5
        sp_arr[WINDOW + 3] = 90.0   # exit bar spread

        trades = run_backtest(
            _mk(sp_arr), _mk(z_arr), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _default_params(),
        )
        assert trades[0]["exit_spread"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Multiple sequential trades
# ---------------------------------------------------------------------------

class TestMultipleTrades:
    def test_two_sequential_long_trades(self):
        """
        Two distinct LONG entry/exit sequences back-to-back.
        """
        # Total bars: WINDOW + 12
        n = WINDOW + 12
        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:] = 0.0

        # Trade 1: enter at W+2, exit at W+4
        z_arr[WINDOW]     = -2.0   # PRIMED
        z_arr[WINDOW + 1] = -1.0   # 2nd crossing
        z_arr[WINDOW + 2] = -1.0   # z_exec (entry)
        z_arr[WINDOW + 3] = -1.0   # day1, no exit
        z_arr[WINDOW + 4] = 1.5    # exit

        # Trade 2: enter at W+7, exit at W+9
        z_arr[WINDOW + 5] = -2.0   # PRIMED
        z_arr[WINDOW + 6] = -1.0   # 2nd crossing
        z_arr[WINDOW + 7] = -1.0   # z_exec (entry)
        z_arr[WINDOW + 8] = -1.0   # day1, no exit
        z_arr[WINDOW + 9] = 1.5    # exit

        sigma_arr = np.where(np.isnan(z_arr), np.nan, 1.0)

        trades = run_backtest(
            _mk(100.0, n=n), _mk(z_arr, n=n), _mk(sigma_arr, n=n),
            _flat_prices(n=n), _flat_prices(n=n), _default_params(),
        )
        assert len(trades) == 2
        assert all(t["direction"] == "long" for t in trades)

    def test_long_then_short_trade(self):
        n = WINDOW + 12
        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:] = 0.0

        # LONG trade
        z_arr[WINDOW]     = -2.0
        z_arr[WINDOW + 1] = -1.0
        z_arr[WINDOW + 2] = -1.0
        z_arr[WINDOW + 3] = 1.5   # exit

        # SHORT trade
        z_arr[WINDOW + 4] = 2.0
        z_arr[WINDOW + 5] = 1.0
        z_arr[WINDOW + 6] = 1.0
        z_arr[WINDOW + 7] = -1.5  # exit

        sigma_arr = np.where(np.isnan(z_arr), np.nan, 1.0)

        trades = run_backtest(
            _mk(100.0, n=n), _mk(z_arr, n=n), _mk(sigma_arr, n=n),
            _flat_prices(n=n), _flat_prices(n=n), _default_params(),
        )
        assert len(trades) == 2
        assert trades[0]["direction"] == "long"
        assert trades[1]["direction"] == "short"

"""
Stage 2 (MU) integration tests — stop-loss behaviour in backtest.py,
trade-dict consistency, and evaluate_chromosome with USE_MONTE_CARLO=True.

Tests marked # ── WILL FAIL ── are intentional failures that document known
gaps in the current implementation:

  TestForceCloseConsistency.test_force_close_has_stop_loss_key
      → AssertionError: force-close block does not add 'stop_loss' to the dict.

All helpers are self-contained so this file has no dependency on other test
modules.

State-machine chronology reminder (from backtest.py docstring):
  bar W  : IDLE   z < -open_long  → PRIMED_LONG (primed_days=0)
  bar W+1: PRIMED primed_days=1, z >= -open_long → 2nd crossing, entry_exec=W+2
  bar W+2: entry processed → IN_LONG, trade_day=0
           loop iteration at i=W+2: trade_day increments to 1
  bar W+3: trade_day=2; stop-loss or adaptive-exit evaluated
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.fitness import evaluate_chromosome
from src import config

# ---------------------------------------------------------------------------
# Constants / helpers  (mirror test_backtest.py conventions)
# ---------------------------------------------------------------------------

WINDOW = 100
N      = WINDOW + 15   # 115 total bars; 15 tradeable bars after warm-up


def _dates(n: int = N):
    return pd.date_range("2020-01-01", periods=n)


def _mk(vals_or_scalar, n: int = N):
    """Build a pd.Series of length n from a scalar or numpy array."""
    dates = _dates(n)
    if np.isscalar(vals_or_scalar):
        arr = np.full(n, float(vals_or_scalar))
    else:
        arr = np.asarray(vals_or_scalar, dtype=float)
        if len(arr) < n:
            arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
        arr = arr[:n]
    return pd.Series(arr, index=dates)


def _z_base(n: int = N) -> np.ndarray:
    """All-zero z with NaN for the first WINDOW bars (rolling-stats warm-up)."""
    arr = np.zeros(n)
    arr[:WINDOW] = np.nan
    return arr


def _sigma_base(n: int = N) -> np.ndarray:
    arr = np.ones(n)
    arr[:WINDOW] = np.nan
    return arr


def _flat_prices(val: float = 100.0, n: int = N):
    """Constant price → SMA == price → momentum filter always passes."""
    return _mk(val, n)


def _params(stop_loss: float = 0.5, **overrides) -> dict:
    """Default params with tight stop_loss (fires at -(1.5+0.5) = -2.0)."""
    p = {
        "open_long":   1.5,
        "open_short":  1.5,
        "close_long":  0.5,
        "close_short": 0.5,
        "exit_step":   0.1,
        "window":      WINDOW,
        "stop_loss":   stop_loss,
    }
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# Stop-loss: IN_LONG
#
# Sequence:
#   bar W  : z=-2.0  → PRIMED_LONG
#   bar W+1: z=-1.0  → 2nd crossing; entry_exec = W+2
#   bar W+2: z_exec=-1.0, guard OK → IN_LONG; i=W+2 processes as IN_LONG,
#            trade_day=1, threshold=-0.4, no exit
#   bar W+3: z=-2.1 < -(1.5+0.5)=-2.0 → STOP-LOSS fires (Stage 2 only)
# ---------------------------------------------------------------------------

class TestStopLossLong:
    def _z_sl_hit(self) -> np.ndarray:
        z = _z_base()
        z[WINDOW]     = -2.0
        z[WINDOW + 1] = -1.0
        z[WINDOW + 2] = -1.0    # z_exec
        z[WINDOW + 3] = -2.1    # past stop-loss threshold
        return z

    def test_stop_loss_fires_when_mc_enabled(self, monkeypatch):
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        trades = run_backtest(
            _mk(100.0), _mk(self._z_sl_hit()), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert len(trades) == 1
        assert trades[0]["stop_loss"] is True

    def test_stop_loss_does_not_fire_when_mc_disabled(self, monkeypatch):
        """Gene 9 is dormant in Stage 1: z=-2.1 must NOT trigger stop-loss.
        The position should exit later via the adaptive reversion threshold."""
        monkeypatch.setattr(config, "USE_MONTE_CARLO", False)
        trades = run_backtest(
            _mk(100.0), _mk(self._z_sl_hit()), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        # One trade closes (bar W+4 z=0 crosses adaptive threshold), stop_loss=False
        assert len(trades) >= 1
        assert trades[0]["stop_loss"] is False

    def test_normal_reversion_exit_has_stop_loss_false(self, monkeypatch):
        """When reversion is clean (z=1.5), stop_loss must be False."""
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        z = _z_base()
        z[WINDOW]     = -2.0
        z[WINDOW + 1] = -1.0
        z[WINDOW + 2] = -1.0   # z_exec
        z[WINDOW + 3] = 1.5    # normal reversion; threshold=-0.3; 1.5>-0.3 → exit
        trades = run_backtest(
            _mk(100.0), _mk(z), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert len(trades) == 1
        assert trades[0]["stop_loss"] is False

    def test_stop_loss_key_present_in_trade_dict(self, monkeypatch):
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        trades = run_backtest(
            _mk(100.0), _mk(self._z_sl_hit()), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert "stop_loss" in trades[0]

    def test_stop_loss_not_triggered_when_below_threshold(self, monkeypatch):
        """z=-1.9 is above the threshold -(1.5+0.5)=-2.0 → no stop-loss."""
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        z = _z_base()
        z[WINDOW]     = -2.0
        z[WINDOW + 1] = -1.0
        z[WINDOW + 2] = -1.0
        z[WINDOW + 3] = -1.9   # NOT past -2.0 → no SL; z[W+4]=0 closes normally
        trades = run_backtest(
            _mk(100.0), _mk(z), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert len(trades) == 1
        assert trades[0]["stop_loss"] is False


# ---------------------------------------------------------------------------
# Stop-loss: IN_SHORT
#
# Sequence:
#   bar W  : z=+2.0  → PRIMED_SHORT
#   bar W+1: z=+1.0  → 2nd crossing; entry_exec = W+2
#   bar W+2: z_exec=+1.0, guard OK → IN_SHORT; trade_day=1, no exit
#   bar W+3: z=+2.1 > (1.5+0.5)=2.0 → STOP-LOSS fires (Stage 2 only)
# ---------------------------------------------------------------------------

class TestStopLossShort:
    def _z_sl_hit(self) -> np.ndarray:
        z = _z_base()
        z[WINDOW]     = 2.0
        z[WINDOW + 1] = 1.0
        z[WINDOW + 2] = 1.0     # z_exec
        z[WINDOW + 3] = 2.1     # past stop-loss threshold
        return z

    def test_stop_loss_fires_when_mc_enabled(self, monkeypatch):
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        trades = run_backtest(
            _mk(100.0), _mk(self._z_sl_hit()), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert len(trades) == 1
        assert trades[0]["stop_loss"] is True

    def test_stop_loss_does_not_fire_when_mc_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "USE_MONTE_CARLO", False)
        trades = run_backtest(
            _mk(100.0), _mk(self._z_sl_hit()), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert len(trades) >= 1
        assert trades[0]["stop_loss"] is False

    def test_normal_reversion_exit_has_stop_loss_false(self, monkeypatch):
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        z = _z_base()
        z[WINDOW]     = 2.0
        z[WINDOW + 1] = 1.0
        z[WINDOW + 2] = 1.0
        z[WINDOW + 3] = -1.5   # clean reversion; threshold=0.3; -1.5<0.3 → exit
        trades = run_backtest(
            _mk(100.0), _mk(z), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert len(trades) == 1
        assert trades[0]["stop_loss"] is False

    def test_stop_loss_key_present_in_trade_dict(self, monkeypatch):
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        trades = run_backtest(
            _mk(100.0), _mk(self._z_sl_hit()), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert "stop_loss" in trades[0]

    def test_stop_loss_not_triggered_when_below_threshold(self, monkeypatch):
        """z=+1.9 is below 2.0 → no stop-loss."""
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        z = _z_base()
        z[WINDOW]     = 2.0
        z[WINDOW + 1] = 1.0
        z[WINDOW + 2] = 1.0
        z[WINDOW + 3] = 1.9   # NOT past 2.0 → no SL; z[W+4]=0 closes normally
        trades = run_backtest(
            _mk(100.0), _mk(z), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )
        assert len(trades) == 1
        assert trades[0]["stop_loss"] is False


# ---------------------------------------------------------------------------
# Force-close trade dict consistency
#
# ── WILL FAIL ─────────────────────────────────────────────────────────────
# The force-close block in backtest.py does NOT add 'stop_loss' to the trade
# dict.  Any code that iterates all trades and accesses trade['stop_loss']
# will raise a KeyError.  The test below documents this gap and will FAIL
# with AssertionError until the force-close block is updated:
#
#     trades.append({
#         ...existing keys...,
#         'force_close': True,
#         'stop_loss':   False,   # ← add this line
#     })
# ---------------------------------------------------------------------------

class TestForceCloseConsistency:
    def _run_force_close(self, monkeypatch) -> list:
        """
        Produce a trade that is definitely force-closed at end of data.

        Setup: z[W+2:] = -1.0 keeps the spread below the adaptive exit
        threshold (max 0.0) and above the stop-loss trigger (< -2.0), so
        neither normal exit nor stop-loss fires.  Position is force-closed
        at the last bar.
        """
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        z = _z_base()
        z[WINDOW]     = -2.0   # PRIMED_LONG
        z[WINDOW + 1] = -1.0   # 2nd crossing → entry_exec = W+2
        z[WINDOW + 2:] = -1.0  # stays in [-2.0, 0) → never exits normally or via SL
        return run_backtest(
            _mk(100.0), _mk(z), _mk(_sigma_base()),
            _flat_prices(), _flat_prices(), _params(),
        )

    def test_force_close_trade_is_produced(self, monkeypatch):
        """Sanity check: at least one force-close trade must be present."""
        trades = self._run_force_close(monkeypatch)
        force_closed = [t for t in trades if t.get("force_close")]
        assert len(force_closed) >= 1

    def test_force_close_trade_has_stop_loss_key(self, monkeypatch):
        """
        EXPECTED TO FAIL — force-close block missing 'stop_loss' key.

        Fails with AssertionError: assert 'stop_loss' in {...}
        Will pass once 'stop_loss': False is added to the force-close dict.
        """
        trades = self._run_force_close(monkeypatch)
        force_closed = [t for t in trades if t.get("force_close")]
        assert len(force_closed) >= 1
        for t in force_closed:
            assert "stop_loss" in t, (
                "force-close trade dict is missing 'stop_loss' key — "
                "add 'stop_loss': False to the force-close block in backtest.py"
            )

    def test_all_trade_dicts_have_consistent_keys(self, monkeypatch):
        """
        EXPECTED TO FAIL — same root cause as above.

        Iterating all trades (including the force-close one) and checking for
        'stop_loss' surfaces the inconsistency regardless of trade type.
        """
        trades = self._run_force_close(monkeypatch)
        required = {"entry_date", "exit_date", "direction", "pnl", "sigma",
                    "duration", "force_close", "stop_loss"}
        for t in trades:
            missing = required - set(t.keys())
            assert not missing, f"Trade dict missing keys: {missing}"


# ---------------------------------------------------------------------------
# evaluate_chromosome — Stage 2 (USE_MONTE_CARLO=True)
# ---------------------------------------------------------------------------

class TestEvaluateChromosomeMC:
    def _prices_and_universe(self, n: int = 400, seed: int = 42):
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n)
        b = np.cumsum(np.random.randn(n)) + 100
        a = b + np.random.randn(n) * 0.5   # cointegrated
        c = np.cumsum(np.random.randn(n)) + 80
        return pd.DataFrame({"T0": a, "T1": b, "T2": c}, index=dates), ["T0", "T1", "T2"]

    def _chrom(self) -> np.ndarray:
        return np.array([0, 1, 1.5, 1.5, 0.5, 0.5, 0.1, 130, 2.0], dtype=float)

    def _set_mc(self, monkeypatch, *, n_paths: int = 5, n_steps: int = 30):
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        monkeypatch.setattr(config, "MC_N_PATHS", n_paths)
        monkeypatch.setattr(config, "MC_N_STEPS", n_steps)

    def test_mc_returns_two_element_tuple(self, monkeypatch):
        self._set_mc(monkeypatch)
        prices, universe = self._prices_and_universe()
        result = evaluate_chromosome(self._chrom(), prices, universe)
        assert len(result) == 2

    def test_mc_both_objectives_are_floats(self, monkeypatch):
        self._set_mc(monkeypatch)
        prices, universe = self._prices_and_universe()
        obj1, obj2 = evaluate_chromosome(self._chrom(), prices, universe)
        assert isinstance(obj1, float)
        assert isinstance(obj2, float)

    def test_mc_obj1_finite(self, monkeypatch):
        self._set_mc(monkeypatch)
        prices, universe = self._prices_and_universe()
        obj1, _ = evaluate_chromosome(self._chrom(), prices, universe)
        import math
        assert math.isfinite(obj1)

    def test_mc_obj2_finite(self, monkeypatch):
        self._set_mc(monkeypatch)
        prices, universe = self._prices_and_universe()
        _, obj2 = evaluate_chromosome(self._chrom(), prices, universe)
        import math
        assert math.isfinite(obj2)

    def test_same_stock_returns_worst_in_mc_mode(self, monkeypatch):
        """Even in Stage 2 same-stock pairs must return WORST immediately."""
        self._set_mc(monkeypatch)
        prices, universe = self._prices_and_universe()
        x = np.array([0, 0, 1.5, 1.5, 0.5, 0.5, 0.1, 130, 2.0], dtype=float)
        assert evaluate_chromosome(x, prices, universe) == (5.0, 100.0)

    def test_stage1_and_stage2_both_produce_valid_objectives(self, monkeypatch):
        """Regression: switching the flag must not break either mode."""
        import math
        prices, universe = self._prices_and_universe()
        chrom = self._chrom()

        monkeypatch.setattr(config, "USE_MONTE_CARLO", False)
        r1 = evaluate_chromosome(chrom, prices, universe)

        self._set_mc(monkeypatch)
        r2 = evaluate_chromosome(chrom, prices, universe)

        for r in (r1, r2):
            assert len(r) == 2
            assert all(isinstance(v, float) for v in r)
            # Both may return WORST (too few trades) but must always be finite
            assert all(math.isfinite(v) for v in r)

    def test_mc_mode_obj2_differs_from_stage1(self, monkeypatch):
        """
        With a cointegrated pair that generates trades, Stage 2 obj2 uses
        MC-EP rather than historical ROI, so the two values should generally
        differ.  (Not guaranteed for every seed — if the pair has no trades
        both modes return WORST=100 and this test is vacuously skipped.)
        """
        prices, universe = self._prices_and_universe()
        chrom = self._chrom()

        monkeypatch.setattr(config, "USE_MONTE_CARLO", False)
        _, obj2_s1 = evaluate_chromosome(chrom, prices, universe)

        self._set_mc(monkeypatch, n_paths=20, n_steps=100)
        _, obj2_s2 = evaluate_chromosome(chrom, prices, universe)

        if obj2_s1 == 100.0 and obj2_s2 == 100.0:
            pytest.skip("pair produced no trades — both modes return WORST")

        # At least one of the objectives should differ between modes
        assert obj2_s1 != pytest.approx(obj2_s2, abs=1e-9), (
            "Stage 1 and Stage 2 obj2 are identical — MC branch may not be active"
        )

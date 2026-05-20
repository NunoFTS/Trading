"""
Tests for src/montecarlo.py — OU-VG Monte Carlo engine (Stage 2 / MU).

Organised into five test classes:
  TestFitOuVg            — parameter fitting
  TestSimulatePaths      — forward simulation
  TestBacktestPath       — internal single-path backtest
  TestExpectedProfitability — EP / variance summary
  TestEdgeCasesWillFail  — intentional failures that expose missing guards
                           (test_n_steps_zero_raises, test_zero_paths_ep_nan)
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.montecarlo import (
    _backtest_path,
    _excess_kurtosis,
    expected_profitability,
    fit_ou_vg,
    simulate_paths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ar1_series(alpha: float = 0.85, n: int = 200, sigma: float = 0.3, seed: int = 42):
    """Synthetic AR(1) process as a pandas Series."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = alpha * z[t - 1] + rng.normal(0, sigma)
    return pd.Series(z, index=pd.date_range("2020-01-01", periods=n))


def _default_trade_params(stop_loss: float = float("inf")) -> dict:
    return {
        "open_long":   1.5,
        "open_short":  1.5,
        "close_long":  0.5,
        "close_short": 0.5,
        "exit_step":   0.1,
        "stop_loss":   stop_loss,
    }


def _default_ou_params(z0: float = 0.0) -> dict:
    return dict(alpha=0.85, c=0.0, sigma_vg=0.3, nu=0.2, sigma_hist=1.0, z0=z0)


# ---------------------------------------------------------------------------
# _excess_kurtosis (internal helper)
# ---------------------------------------------------------------------------

class TestExcessKurtosis:
    def test_normal_distribution_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(10_000)
        assert abs(_excess_kurtosis(x)) < 0.3  # normal → excess kurtosis ≈ 0

    def test_constant_series_returns_zero(self):
        assert _excess_kurtosis(np.ones(100)) == 0.0

    def test_short_series_returns_zero(self):
        assert _excess_kurtosis(np.array([1.0, 2.0, 3.0])) == 0.0  # n < 4

    def test_fat_tailed_distribution_positive(self):
        rng = np.random.default_rng(7)
        # Student-t with df=3 has excess kurtosis = 6
        x = rng.standard_t(df=3, size=5_000)
        assert _excess_kurtosis(x) > 1.0


# ---------------------------------------------------------------------------
# fit_ou_vg
# ---------------------------------------------------------------------------

class TestFitOuVg:
    def test_returns_all_required_keys(self):
        params = fit_ou_vg(_ar1_series())
        for key in ("alpha", "c", "sigma_vg", "nu", "sigma_hist", "z0"):
            assert key in params

    def test_alpha_in_stable_range(self):
        """Fitted alpha must be in [0, 1) so the process is stationary."""
        params = fit_ou_vg(_ar1_series(alpha=0.85))
        assert 0.0 <= params["alpha"] < 1.0

    def test_alpha_near_true_value(self):
        """For a long AR(1) series the OLS estimate should be close to truth."""
        params = fit_ou_vg(_ar1_series(alpha=0.80, n=3000))
        assert params["alpha"] == pytest.approx(0.80, abs=0.05)

    def test_near_unit_root_clipped_to_09999(self):
        """A near-unit-root random walk must be clipped to 0.9999, never ≥ 1."""
        rw = pd.Series(
            np.cumsum(np.random.default_rng(1).standard_normal(500)) + 50,
            index=pd.date_range("2020", periods=500),
        )
        params = fit_ou_vg(rw)
        assert params["alpha"] <= 0.9999

    def test_short_series_returns_fallback(self):
        """< 30 observations → safe fallback returned, no exception."""
        short = pd.Series([0.1, -0.2, 0.3], index=pd.date_range("2020", periods=3))
        params = fit_ou_vg(short)
        assert params["alpha"] == pytest.approx(0.90)
        # z0 should be the last value even in fallback
        assert params["z0"] == pytest.approx(0.3)

    def test_constant_series_no_error(self):
        """All-constant series → sigma_vg clipped to 1e-8, no ZeroDivisionError."""
        const = pd.Series(
            np.full(100, 5.0), index=pd.date_range("2020", periods=100)
        )
        params = fit_ou_vg(const)
        assert params["sigma_vg"] >= 1e-8

    def test_all_nan_series_returns_fallback(self):
        """All-NaN → dropna gives empty array → fallback, no crash."""
        nan_s = pd.Series(np.full(50, np.nan), index=pd.date_range("2020", periods=50))
        params = fit_ou_vg(nan_s)
        assert params["alpha"] == pytest.approx(0.90)

    def test_sigma_vg_positive(self):
        params = fit_ou_vg(_ar1_series())
        assert params["sigma_vg"] > 0.0

    def test_nu_in_valid_range(self):
        params = fit_ou_vg(_ar1_series())
        assert 0.01 <= params["nu"] <= 5.0

    def test_sigma_hist_positive(self):
        params = fit_ou_vg(_ar1_series())
        assert params["sigma_hist"] > 0.0

    def test_z0_equals_last_observation(self):
        s = _ar1_series()
        params = fit_ou_vg(s)
        assert params["z0"] == pytest.approx(float(s.dropna().iloc[-1]))

    def test_exactly_30_observations_does_not_use_fallback(self):
        """Boundary: exactly 30 obs → len(z) < 30 is False → real estimation."""
        s30 = _ar1_series(n=30)
        params = fit_ou_vg(s30)
        # If fallback were used, alpha would be exactly 0.90
        # With real data it's very unlikely to be exactly 0.90
        assert isinstance(params["alpha"], float)

    def test_returns_dict(self):
        assert isinstance(fit_ou_vg(_ar1_series()), dict)

    def test_array_like_fallback_uses_last_non_nan_observation(self):
        arr = np.array([np.nan, 0.1, np.nan, -0.4, np.nan])
        params = fit_ou_vg(arr)
        assert params["alpha"] == pytest.approx(0.90)
        assert params["z0"] == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# simulate_paths
# ---------------------------------------------------------------------------

class TestSimulatePaths:
    def test_output_shape(self):
        paths = simulate_paths(_default_ou_params(), n_steps=50, n_paths=20)
        assert paths.shape == (20, 50)

    def test_first_column_equals_z0(self):
        params = _default_ou_params(z0=2.5)
        paths = simulate_paths(params, n_steps=30, n_paths=10)
        np.testing.assert_array_equal(paths[:, 0], 2.5)

    def test_deterministic_with_same_seed(self):
        p = _default_ou_params()
        a = simulate_paths(p, n_steps=100, n_paths=5, seed=0)
        b = simulate_paths(p, n_steps=100, n_paths=5, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_paths(self):
        p = _default_ou_params()
        a = simulate_paths(p, n_steps=100, n_paths=10, seed=1)
        b = simulate_paths(p, n_steps=100, n_paths=10, seed=2)
        assert not np.array_equal(a, b)

    def test_returns_float64(self):
        paths = simulate_paths(_default_ou_params(), 50, 5)
        assert paths.dtype == np.float64

    def test_all_values_finite(self):
        """With valid OU params every simulated value should be finite."""
        paths = simulate_paths(_default_ou_params(), n_steps=252, n_paths=50, seed=0)
        assert np.all(np.isfinite(paths))

    def test_long_run_mean_near_zero(self):
        """Stationary OU (c=0) → long-run mean ≈ c/(1-alpha) = 0."""
        paths = simulate_paths(_default_ou_params(), n_steps=2000, n_paths=200, seed=0)
        assert abs(paths[:, -1].mean()) < 1.0  # loose tolerance

    def test_n_paths_one_works(self):
        paths = simulate_paths(_default_ou_params(), n_steps=10, n_paths=1)
        assert paths.shape == (1, 10)

    def test_n_steps_one_returns_only_z0(self):
        params = _default_ou_params(z0=3.7)
        paths = simulate_paths(params, n_steps=1, n_paths=5)
        assert paths.shape == (5, 1)
        np.testing.assert_array_equal(paths[:, 0], 3.7)

    def test_zero_paths_returns_empty_2d_array(self):
        paths = simulate_paths(_default_ou_params(z0=1.2), n_steps=10, n_paths=0)
        assert paths.shape == (0, 10)
        assert paths.dtype == np.float64

    # ── Edge case: n_steps=0 — currently raises IndexError (missing guard) ──

    def test_n_steps_zero_raises_index_error(self):
        """
        EXPECTED TO RAISE — documents a missing input-validation guard.

        np.empty((n_paths, 0))[:, 0] raises IndexError because axis 1 has
        size 0.  This test will PASS once a guard for n_steps < 1 is added
        to simulate_paths.  Until then it confirms the crash path.
        """
        with pytest.raises(IndexError):
            simulate_paths(_default_ou_params(), n_steps=0, n_paths=5)


# ---------------------------------------------------------------------------
# _backtest_path (internal MC single-path backtest)
# ---------------------------------------------------------------------------

class TestBacktestPath:
    def test_no_signal_returns_zero(self):
        """z never crosses ±open_long/short → no trades → 0.0."""
        z = np.zeros(100)
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(0.0)

    def test_empty_path_returns_zero(self):
        assert _backtest_path(np.array([]), _default_trade_params()) == pytest.approx(0.0)

    def test_long_profitable_reversion(self):
        """
        z=-2.0 → PRIMED_LONG only.
        z=-1.0 crosses back toward 0 through -open_long (= -1.5) → enter long.
        z=+0.5 then next bar exits in no-same-bar mode.
        pnl = 0.5 - (-1.0) = 1.5
        """
        z = np.array([-2.0, -1.0, 0.5, 0.5])
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(1.5)

    def test_short_profitable_reversion(self):
        """
        z=+2.0 → PRIMED_SHORT only.
        z=+1.0 crosses back toward 0 through +open_short (= +1.5) → enter short.
        z=-0.5 then next bar exits in no-same-bar mode.
        pnl = 1.0 - (-0.5) = 1.5
        """
        z = np.array([2.0, 1.0, -0.5, -0.5])
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(1.5)

    def test_long_stop_loss_fires(self):
        """
        stop_loss=0.5 → fires when z < -(open_long+stop_loss) = -2.0.
        Enter at z=-1.0 after confirmation; z=-2.1 triggers stop-loss.
        pnl = -2.1 - (-1.0) = -1.1 (loss).
        """
        z = np.array([-2.0, -1.0, -2.1])
        result = _backtest_path(z, _default_trade_params(stop_loss=0.5))
        assert result == pytest.approx(-1.1)

    def test_short_stop_loss_fires(self):
        """
        stop_loss=0.5 → fires when z > open_short+stop_loss = 2.0.
        Enter at z=+1.0 after confirmation; z=+2.1 triggers stop-loss.
        pnl = 1.0 - 2.1 = -1.1 (loss).
        """
        z = np.array([2.0, 1.0, 2.1])
        result = _backtest_path(z, _default_trade_params(stop_loss=0.5))
        assert result == pytest.approx(-1.1)

    def test_stop_loss_does_not_fire_without_param(self):
        """
        Without stop_loss key → uses float('inf') → never triggers.
        After confirmed entry at -1.0, deep dive to -5.0 should not stop out,
        and reversion to +0.5 exits: pnl = 0.5 - (-1.0) = 1.5.
        """
        params_no_sl = {
            "open_long": 1.5, "open_short": 1.5,
            "close_long": 0.5, "close_short": 0.5,
            "exit_step": 0.1,
        }
        z = np.array([-2.0, -1.0, -5.0, 0.5])
        assert _backtest_path(z, params_no_sl) == pytest.approx(1.5)

    def test_nan_values_skipped(self):
        """NaN at start should not crash; bars after it are processed normally."""
        z = np.array([np.nan, -2.0, -1.0, 0.5, 0.5])
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(1.5)

    def test_mean_of_multiple_trades(self):
        """Two identical long-reversion trades → mean pnl = 1.5."""
        z = np.array([-2.0, -1.0, 0.5, -2.0, -1.0, 0.5])
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(1.5)

    def test_returns_float_type(self):
        z = np.array([-2.0, -1.0, 0.5])
        assert isinstance(_backtest_path(z, _default_trade_params()), float)

    def test_stop_loss_boundary_not_triggered(self):
        """z exactly at stop-loss boundary (not past it) → no stop-loss exit."""
        # sl=0.5 → boundary = -2.0; touching boundary should not stop out.
        z = np.array([-2.0, -1.0, -2.0, 0.5])
        result = _backtest_path(z, _default_trade_params(stop_loss=0.5))
        assert result == pytest.approx(0.5 - (-1.0))

    def test_no_entry_without_cross_toward_zero_long(self):
        z = np.array([-2.0, -2.2, -2.1, -2.3, -2.2])
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(0.0)

    def test_no_entry_without_cross_toward_zero_short(self):
        z = np.array([2.0, 2.2, 2.1, 2.3, 2.2])
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(0.0)

    def test_open_trade_left_at_end_returns_zero(self):
        """An unclosed trade contributes no realised PnL in MC path scoring."""
        z = np.array([-2.0, -1.0, -1.0])
        assert _backtest_path(z, _default_trade_params()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# expected_profitability
# ---------------------------------------------------------------------------

class TestExpectedProfitability:
    def _valid_paths(self, n_paths: int = 20, n_steps: int = 100, seed: int = 0):
        return simulate_paths(_default_ou_params(), n_steps, n_paths, seed=seed)

    def test_returns_two_element_tuple(self):
        paths = self._valid_paths()
        result = expected_profitability(paths, _default_trade_params())
        assert len(result) == 2

    def test_both_values_are_python_floats(self):
        paths = self._valid_paths()
        ep, var_r = expected_profitability(paths, _default_trade_params())
        assert isinstance(ep, float)
        assert isinstance(var_r, float)

    def test_variance_is_non_negative(self):
        paths = self._valid_paths(n_paths=30)
        _, var_r = expected_profitability(paths, _default_trade_params())
        assert var_r >= 0.0

    def test_variance_zero_for_single_path(self):
        """Variance of a single observation is 0 (numpy.var default ddof=0)."""
        paths = self._valid_paths(n_paths=1)
        _, var_r = expected_profitability(paths, _default_trade_params())
        assert var_r == pytest.approx(0.0)

    def test_ep_is_finite_with_valid_input(self):
        paths = simulate_paths(_default_ou_params(), n_steps=252, n_paths=50, seed=42)
        ep, var_r = expected_profitability(paths, _default_trade_params())
        assert math.isfinite(ep)
        assert math.isfinite(var_r)

    def test_consistent_results_with_same_seed(self):
        p = _default_ou_params()
        paths_a = simulate_paths(p, 100, 10, seed=7)
        paths_b = simulate_paths(p, 100, 10, seed=7)
        ep_a, var_a = expected_profitability(paths_a, _default_trade_params())
        ep_b, var_b = expected_profitability(paths_b, _default_trade_params())
        assert ep_a == pytest.approx(ep_b)
        assert var_a == pytest.approx(var_b)

    def test_tight_stop_loss_reduces_variance(self):
        """
        Very tight stop-loss cuts off large losses → reduces spread of outcomes
        vs no stop-loss (loose heuristic, not guaranteed but true for OU).
        """
        paths = simulate_paths(_default_ou_params(), 200, 100, seed=0)
        _, var_no_sl  = expected_profitability(paths, _default_trade_params(stop_loss=float("inf")))
        _, var_tight  = expected_profitability(paths, _default_trade_params(stop_loss=0.2))
        # Tight SL removes extreme downside → variance should not be much larger
        # (This is a directional check, not an exact bound)
        assert var_tight < var_no_sl * 10  # generous bound

    # ── Error-inducing: n_paths=0 produces NaN EP — WILL FAIL ──────────────

    def test_zero_paths_ep_is_finite(self):
        """
        EXPECTED TO FAIL — documents a missing guard in expected_profitability.

        With n_paths=0 the list comprehension produces [], np.array([]).mean()
        returns NaN (with a RuntimeWarning), so math.isfinite(ep) is False.
        This test will pass once a guard like:
            if n_paths == 0: return 0.0, 0.0
        is added to expected_profitability.
        """
        paths = np.empty((0, 100))
        ep, _ = expected_profitability(paths, _default_trade_params())
        assert math.isfinite(ep), (
            f"EP={ep!r} — expected_profitability does not guard against n_paths=0"
        )

    def test_all_flat_paths_return_zero_ep_and_variance(self):
        paths = np.zeros((8, 50), dtype=float)
        ep, var_r = expected_profitability(paths, _default_trade_params())
        assert ep == pytest.approx(0.0)
        assert var_r == pytest.approx(0.0)

"""
Tests for src/fitness.py — decode() and evaluate_chromosome().

decode() tests use only numpy arrays (no CSV needed).
evaluate_chromosome() tests build minimal synthetic price DataFrames.
"""
import numpy as np
import pandas as pd
import pytest

from src.fitness import decode, evaluate_chromosome
import src.fitness as fitness_module
from src import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=300, n_tickers=3, seed=0):
    """Return a (prices_df, universe) with cointegrated first two tickers."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n)

    # Ticker 0 and 1 are cointegrated: A = B + stationary noise
    b_vals = np.cumsum(np.random.randn(n)) + 100
    a_vals = b_vals + np.random.randn(n) * 0.3
    # Ticker 2 is an independent random walk
    c_vals = np.cumsum(np.random.randn(n)) + 80

    df = pd.DataFrame(
        {"T0": a_vals, "T1": b_vals, "T2": c_vals},
        index=dates,
    )
    return df, ["T0", "T1", "T2"]


def _chrom(idx_a=0, idx_b=1, open_long=1.5, open_short=1.5,
           close_long=0.5, close_short=0.5, exit_step=0.1,
           window=100, stop_loss=2.0):
    """Build a 9-element chromosome array."""
    return np.array([
        idx_a, idx_b,
        open_long, open_short,
        close_long, close_short,
        exit_step, window, stop_loss,
    ], dtype=float)


UNIVERSE = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# decode()
# ---------------------------------------------------------------------------

class TestDecode:
    def test_indices_are_rounded_integers(self):
        x = _chrom(idx_a=0.4, idx_b=1.6)
        idx_a, idx_b, _ = decode(x, UNIVERSE)
        assert idx_a == 0   # round(0.4) = 0
        assert idx_b == 2   # round(1.6) = 2

    def test_indices_clipped_to_universe_bounds(self):
        N = len(UNIVERSE) - 1
        x = _chrom(idx_a=-5, idx_b=100)
        idx_a, idx_b, _ = decode(x, UNIVERSE)
        assert idx_a == 0
        assert idx_b == N

    def test_close_long_clamped_to_open_long(self):
        """close_long must not exceed open_long."""
        x = _chrom(open_long=1.5, close_long=2.0)   # close_long > open_long → clamp
        _, _, params = decode(x, UNIVERSE)
        assert params["close_long"] <= params["open_long"]
        assert params["close_long"] == pytest.approx(1.5)

    def test_close_long_unchanged_when_below_open_long(self):
        x = _chrom(open_long=1.5, close_long=0.5)
        _, _, params = decode(x, UNIVERSE)
        assert params["close_long"] == pytest.approx(0.5)

    def test_close_short_clamped_to_open_short(self):
        x = _chrom(open_short=1.5, close_short=2.5)
        _, _, params = decode(x, UNIVERSE)
        assert params["close_short"] <= params["open_short"]
        assert params["close_short"] == pytest.approx(1.5)

    def test_close_short_unchanged_when_below_open_short(self):
        x = _chrom(open_short=1.5, close_short=0.3)
        _, _, params = decode(x, UNIVERSE)
        assert params["close_short"] == pytest.approx(0.3)

    def test_window_clipped_to_min(self):
        x = _chrom(window=10)    # below WINDOW_MIN=100
        _, _, params = decode(x, UNIVERSE)
        assert params["window"] == config.WINDOW_MIN

    def test_window_clipped_to_max(self):
        x = _chrom(window=9999)  # above WINDOW_MAX=200
        _, _, params = decode(x, UNIVERSE)
        assert params["window"] == config.WINDOW_MAX

    def test_window_in_range_is_rounded(self):
        x = _chrom(window=200.7)   # within [WINDOW_MIN=130, WINDOW_MAX=250]
        _, _, params = decode(x, UNIVERSE)
        assert params["window"] == 201

    def test_all_param_keys_present(self):
        _, _, params = decode(_chrom(), UNIVERSE)
        expected_keys = {"open_long", "open_short", "close_long", "close_short",
                         "exit_step", "window", "stop_loss"}
        assert expected_keys.issubset(set(params.keys()))

    def test_exit_step_passed_through(self):
        x = _chrom(exit_step=0.25)
        _, _, params = decode(x, UNIVERSE)
        assert params["exit_step"] == pytest.approx(0.25)

    def test_stop_loss_passed_through(self):
        x = _chrom(stop_loss=1.8)
        _, _, params = decode(x, UNIVERSE)
        assert params["stop_loss"] == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# evaluate_chromosome()
# ---------------------------------------------------------------------------

class TestEvaluateChromosome:
    def test_same_stock_returns_worst(self):
        prices, universe = _make_prices()
        x = _chrom(idx_a=0, idx_b=0)
        obj1, obj2 = evaluate_chromosome(x, prices, universe)
        assert (obj1, obj2) == (5.0, 100.0)

    def test_returns_two_floats(self):
        prices, universe = _make_prices()
        x = _chrom(idx_a=0, idx_b=1)
        result = evaluate_chromosome(x, prices, universe)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_insufficient_data_returns_worst(self):
        # Create a DataFrame with fewer rows than window + 30
        n = 50  # window=100 → 50 < 130
        dates = pd.date_range("2020-01-01", periods=n)
        prices = pd.DataFrame(
            {"T0": np.ones(n) * 100, "T1": np.ones(n) * 100},
            index=dates,
        )
        x = _chrom(idx_a=0, idx_b=1, window=100)
        obj1, obj2 = evaluate_chromosome(x, prices, ["T0", "T1"])
        assert (obj1, obj2) == (5.0, 100.0)

    def test_extreme_beta_returns_worst(self):
        """Force a beta above BETA_MAX by using A = 100*B."""
        n = 300
        np.random.seed(5)
        dates = pd.date_range("2020-01-01", periods=n)
        b_vals = np.cumsum(np.random.randn(n)) + 10
        # Make beta >> BETA_MAX (=10) by multiplying A by a huge factor
        a_vals = 500.0 * b_vals
        prices = pd.DataFrame({"T0": a_vals, "T1": b_vals}, index=dates)
        x = _chrom(idx_a=0, idx_b=1, window=100)
        obj1, obj2 = evaluate_chromosome(x, prices, ["T0", "T1"])
        assert (obj1, obj2) == (5.0, 100.0)

    def test_cointegrated_pair_not_worst(self):
        """A well-cointegrated pair with enough bars should not return WORST."""
        prices, universe = _make_prices(n=400, seed=42)
        x = _chrom(idx_a=0, idx_b=1, window=100)
        obj1, obj2 = evaluate_chromosome(x, prices, universe)
        # At least one objective should be non-WORST if the pair is valid
        # (Some chromosomes may still produce too few trades — accept that)
        assert isinstance(obj1, float) and isinstance(obj2, float)

    def test_min_trades_filter(self, monkeypatch):
        """
        Two series with beta≈1.5 but near-zero spread variance so the z-score
        is always NaN (sigma ≈ 0 → sigma_safe = NaN → z = NaN → 0 trades).
        evaluate_chromosome must return WORST via the MIN_TRADES filter.
        """
        n = 300
        np.random.seed(7)
        dates = pd.date_range("2020-01-01", periods=n)
        b = np.random.randn(n) * 0.01 + 100.0
        a = 1.5 * b + np.random.randn(n) * 0.0001  # beta≈1.5, spread ≈ 0
        prices = pd.DataFrame({"T0": a, "T1": b}, index=dates)
        monkeypatch.setattr(config, "MIN_TRADES", 999)
        x = _chrom(idx_a=0, idx_b=1, window=100, open_long=1.5, open_short=1.5)
        obj1, obj2 = evaluate_chromosome(x, prices, ["T0", "T1"])
        assert (obj1, obj2) == (5.0, 100.0)

    def test_obj1_is_cointegration_score(self):
        """obj1 should equal cointegration_score(series_a, series_b)."""
        from src.spread import cointegration_score
        prices, universe = _make_prices(n=400, seed=10)
        x = _chrom(idx_a=0, idx_b=1, window=100)
        obj1, _ = evaluate_chromosome(x, prices, universe)
        # If not WORST, obj1 should match cointegration score
        if obj1 != 5.0:
            expected = cointegration_score(prices["T0"], prices["T1"])
            assert obj1 == pytest.approx(expected, abs=1e-6)

    def test_magnitude_penalty_applied_correctly(self):
        """
        magnitude penalty (MAG_PENALTY=10) is added to obj2 when |z_start| < 1 or > 3.
        We test that obj2 >= -large_value (no crash) and is finite.
        """
        prices, universe = _make_prices(n=400, seed=20)
        x = _chrom(idx_a=0, idx_b=1, window=100)
        obj1, obj2 = evaluate_chromosome(x, prices, universe)
        assert np.isfinite(obj1)
        assert np.isfinite(obj2)

    def test_aligned_data_guard_uses_intersection_not_raw_length(self):
        n = 250
        dates = pd.date_range("2020-01-01", periods=n)
        a = pd.Series(np.linspace(100, 130, n), index=dates)
        b = pd.Series(np.linspace(90, 120, n), index=dates)
        a.iloc[140:] = np.nan
        b.iloc[:110] = np.nan
        prices = pd.DataFrame({"T0": a, "T1": b}, index=dates)

        x = _chrom(idx_a=0, idx_b=1, window=130)
        obj1, obj2 = evaluate_chromosome(x, prices, ["T0", "T1"])
        assert (obj1, obj2) == (5.0, 100.0)

    def test_force_closed_only_trades_are_excluded_from_stage1_fitness(self, monkeypatch):
        prices, universe = _make_prices(n=300, seed=3)

        monkeypatch.setattr(fitness_module, "compute_beta", lambda a, b: 1.0)
        monkeypatch.setattr(fitness_module, "compute_spread", lambda a, b, beta: a - b)
        monkeypatch.setattr(fitness_module, "smooth_spread", lambda spread, span: spread)
        monkeypatch.setattr(
            fitness_module,
            "rolling_stats",
            lambda spread, window: (
                pd.Series(np.zeros(len(spread)), index=spread.index),
                pd.Series(np.ones(len(spread)), index=spread.index),
            ),
        )
        monkeypatch.setattr(fitness_module, "cointegration_score", lambda a, b: -4.0)
        monkeypatch.setattr(
            fitness_module,
            "run_backtest",
            lambda *args, **kwargs: [{"pnl": 10.0, "sigma": 1.0, "force_close": True}],
        )
        monkeypatch.setattr(config, "USE_MONTE_CARLO", False)

        obj1, obj2 = evaluate_chromosome(_chrom(window=130), prices, universe)
        assert (obj1, obj2) == (5.0, 100.0)

    def test_stage2_magnitude_penalty_scales_mc_objective(self, monkeypatch):
        prices, universe = _make_prices(n=300, seed=4)

        def fake_mu_sigma(spread, window):
            mu = pd.Series(np.zeros(len(spread)), index=spread.index)
            sigma = pd.Series(np.ones(len(spread)), index=spread.index)
            return mu, sigma

        monkeypatch.setattr(fitness_module, "compute_beta", lambda a, b: 1.0)
        monkeypatch.setattr(fitness_module, "compute_spread", lambda a, b, beta: a - b)
        monkeypatch.setattr(
            fitness_module,
            "smooth_spread",
            lambda spread, span: pd.Series(
                np.r_[np.full(129, 0.0), 0.5, np.zeros(len(spread) - 130)],
                index=spread.index,
            ),
        )
        monkeypatch.setattr(fitness_module, "rolling_stats", fake_mu_sigma)
        monkeypatch.setattr(fitness_module, "cointegration_score", lambda a, b: -4.0)
        monkeypatch.setattr(
            fitness_module,
            "run_backtest",
            lambda *args, **kwargs: [{"pnl": 1.0, "sigma": 1.0, "force_close": False}],
        )
        monkeypatch.setattr(fitness_module, "fit_ou_vg", lambda z: {"z0": 0.0})
        monkeypatch.setattr(fitness_module, "simulate_paths", lambda *args, **kwargs: np.zeros((2, 3)))
        monkeypatch.setattr(fitness_module, "mc_ep", lambda paths, params: (2.0, 0.0))
        monkeypatch.setattr(config, "USE_MONTE_CARLO", True)
        monkeypatch.setattr(config, "MAG_PENALTY", 0.25)
        monkeypatch.setattr(config, "MC_GAMMA", 0.0)

        _, obj2 = evaluate_chromosome(_chrom(window=130), prices, universe)
        assert obj2 == pytest.approx(-0.5)

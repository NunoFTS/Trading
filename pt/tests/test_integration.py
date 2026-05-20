"""
Integration / smoke tests — wire the full pipeline end-to-end with synthetic data.

These do NOT touch the real CSV and do NOT run the GA (too slow for a test suite).
Instead they call the same functions main.py uses: compute_beta → compute_spread →
smooth_spread → rolling_stats → run_backtest → cointegration_score.

One test also exercises evaluate_chromosome() directly (the GA fitness function)
to confirm it survives a synthetic dataset without errors.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.spread import (
    compute_beta,
    compute_spread,
    smooth_spread,
    rolling_stats,
    cointegration_score,
)
from src.backtest import run_backtest
from src.fitness import decode, evaluate_chromosome
from src import config


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _cointegrated_prices(n=400, seed=42):
    """
    Build a prices DataFrame with two cointegrated tickers (T0, T1) and
    one independent ticker (T2).  Universe = ['T0', 'T1', 'T2'].
    """
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n)

    b = np.cumsum(np.random.randn(n)) + 100
    a = b + np.random.randn(n) * 0.5      # A ≈ B + noise → cointegrated
    c = np.cumsum(np.random.randn(n)) + 80  # independent

    prices = pd.DataFrame({"T0": a, "T1": b, "T2": c}, index=dates)
    universe = ["T0", "T1", "T2"]
    return prices, universe


# ---------------------------------------------------------------------------
# Pipeline smoke test
# ---------------------------------------------------------------------------

class TestPipelineSmoke:
    def test_full_pipeline_runs_without_error(self):
        prices, universe = _cointegrated_prices()
        series_a = prices["T0"]
        series_b = prices["T1"]

        beta     = compute_beta(series_a, series_b)
        raw      = compute_spread(series_a, series_b, beta)
        smoothed = smooth_spread(raw, config.EMA_SPAN)
        mu, sigma = rolling_stats(smoothed, config.WINDOW_MIN)
        sigma_safe = sigma.where(sigma > 1e-8)
        z = (smoothed - mu) / sigma_safe

        params = {
            "open_long":   1.5,
            "open_short":  1.5,
            "close_long":  0.5,
            "close_short": 0.5,
            "exit_step":   0.1,
            "window":      config.WINDOW_MIN,
            "stop_loss":   2.0,
        }
        trades = run_backtest(smoothed, z, sigma_safe, series_a, series_b, params)

        # Pipeline should complete; result is a list (possibly empty)
        assert isinstance(trades, list)

    def test_pipeline_returns_valid_trade_fields(self):
        prices, _ = _cointegrated_prices()
        series_a = prices["T0"]
        series_b = prices["T1"]

        beta      = compute_beta(series_a, series_b)
        smoothed  = smooth_spread(compute_spread(series_a, series_b, beta), config.EMA_SPAN)
        mu, sigma = rolling_stats(smoothed, config.WINDOW_MIN)
        sigma_safe = sigma.where(sigma > 1e-8)
        z = (smoothed - mu) / sigma_safe

        params = {
            "open_long":   1.5,
            "open_short":  1.5,
            "close_long":  0.5,
            "close_short": 0.5,
            "exit_step":   0.1,
            "window":      config.WINDOW_MIN,
            "stop_loss":   2.0,
        }
        trades = run_backtest(smoothed, z, sigma_safe, series_a, series_b, params)

        required_fields = {"entry_date", "exit_date", "direction", "pnl",
                           "entry_z", "exit_z", "sigma", "duration"}
        for t in trades:
            assert required_fields.issubset(set(t.keys()))
            assert t["duration"] > 0
            assert t["sigma"] > 0.0
            assert t["exit_date"] > t["entry_date"]
            assert t["direction"] in ("long", "short")

    def test_z_score_has_correct_shape(self):
        prices, _ = _cointegrated_prices()
        series_a = prices["T0"]
        series_b = prices["T1"]
        beta      = compute_beta(series_a, series_b)
        smoothed  = smooth_spread(compute_spread(series_a, series_b, beta), config.EMA_SPAN)
        mu, sigma = rolling_stats(smoothed, config.WINDOW_MIN)
        sigma_safe = sigma.where(sigma > 1e-8)
        z = (smoothed - mu) / sigma_safe

        assert len(z) == len(prices)
        assert z.iloc[:config.WINDOW_MIN - 1].isna().all()

    def test_beta_is_reasonable_for_cointegrated_pair(self):
        prices, _ = _cointegrated_prices()
        beta = compute_beta(prices["T0"], prices["T1"])
        assert config.BETA_MIN < beta <= config.BETA_MAX

    def test_cointegration_score_negative_for_cointegrated_pair(self):
        prices, _ = _cointegrated_prices()
        score = cointegration_score(prices["T0"], prices["T1"])
        assert score < -3.0   # strongly cointegrated


# ---------------------------------------------------------------------------
# evaluate_chromosome end-to-end
# ---------------------------------------------------------------------------

class TestEvaluateChromosomeIntegration:
    def test_valid_chromosome_returns_finite_objectives(self):
        prices, universe = _cointegrated_prices()
        x = np.array([0, 1, 1.5, 1.5, 0.5, 0.5, 0.1, 100.0, 2.0])
        obj1, obj2 = evaluate_chromosome(x, prices, universe)
        assert np.isfinite(obj1)
        assert np.isfinite(obj2)

    def test_same_ticker_returns_worst(self):
        prices, universe = _cointegrated_prices()
        x = np.array([0, 0, 1.5, 1.5, 0.5, 0.5, 0.1, 100.0, 2.0])
        assert evaluate_chromosome(x, prices, universe) == (5.0, 100.0)

    def test_does_not_raise_with_random_chromosomes(self):
        prices, universe = _cointegrated_prices()
        np.random.seed(99)
        for _ in range(20):
            x = np.array([
                np.random.randint(0, 3),
                np.random.randint(0, 3),
                np.random.uniform(1.0, 3.0),
                np.random.uniform(1.0, 3.0),
                np.random.uniform(0.0, 1.5),
                np.random.uniform(0.0, 1.5),
                np.random.uniform(0.05, 0.5),
                np.random.uniform(100, 200),
                np.random.uniform(0.5, 3.0),
            ])
            result = evaluate_chromosome(x, prices, universe)
            assert len(result) == 2
            assert all(np.isfinite(v) for v in result)


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------

class TestLoggingIntegration:
    def test_backtest_logger_exists(self):
        """Verify that src.backtest exposes a logger (not just no crash)."""
        import src.backtest as bt
        assert hasattr(bt, "logger")

    def test_fitness_logger_exists(self):
        import src.fitness as ft
        assert hasattr(ft, "logger")

    def test_debug_messages_emitted_on_primed_expiry(self, caplog):
        """
        Force a PRIMED_LONG expiry and verify debug output is produced.
        """
        import numpy as np
        import pandas as pd
        from src.backtest import run_backtest

        WINDOW = 100
        n = WINDOW + 10
        z_arr = np.full(n, np.nan)
        z_arr[WINDOW:]    = -1.6   # stays below open_long, no 2nd crossing
        z_arr[WINDOW]     = -2.0   # PRIMED trigger

        sigma_arr = np.where(np.isnan(z_arr), np.nan, 1.0)
        dates = pd.date_range("2020-01-01", periods=n)

        def _s(arr):
            return pd.Series(arr, index=dates, dtype=float)

        params = {
            "open_long": 1.5, "open_short": 1.5,
            "close_long": 0.5, "close_short": 0.5,
            "exit_step": 0.1, "window": WINDOW, "stop_loss": 2.0,
        }

        with caplog.at_level(logging.DEBUG, logger="src.backtest"):
            run_backtest(_s(100.0 * np.ones(n)), _s(z_arr), _s(sigma_arr),
                         _s(100.0 * np.ones(n)), _s(100.0 * np.ones(n)), params)

        # At least one "PRIMED_LONG" debug message should have been emitted
        assert any("PRIMED_LONG" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Plot integration
# ---------------------------------------------------------------------------

class TestPlotIntegration:
    def test_plot_top5_static_returns_path(self, tmp_path, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg, "OUTPUT_DIR", str(tmp_path))

        prices, universe = _cointegrated_prices(n=400)
        series_a = prices["T0"]
        series_b = prices["T1"]
        from src.spread import compute_beta, compute_spread, smooth_spread, rolling_stats
        beta     = compute_beta(series_a, series_b)
        smoothed = smooth_spread(compute_spread(series_a, series_b, beta), cfg.EMA_SPAN)
        mu, sigma = rolling_stats(smoothed, 200)
        sigma_safe = sigma.where(sigma > 1e-8)
        z = (smoothed - mu) / sigma_safe
        params = {
            "open_long": 1.5, "open_short": 1.5,
            "close_long": 0.5, "close_short": 0.5,
            "exit_step": 0.1, "window": 200, "stop_loss": 2.0,
        }
        from src.backtest import run_backtest
        trades = run_backtest(smoothed, z, sigma_safe, series_a, series_b, params)
        sol = dict(ticker_a="T0", ticker_b="T1", params=params,
                   z=z, trades=trades, smoothed=smoothed)

        import matplotlib
        matplotlib.use("Agg")
        from src.plot import plot_top5_static
        path = plot_top5_static([sol])

        import os
        assert path.endswith(".png")
        assert os.path.exists(path)

    def test_plot_top5_static_with_force_close(self, tmp_path, monkeypatch):
        """Ensure hatching code path for force-closed trades does not crash."""
        import src.config as cfg
        monkeypatch.setattr(cfg, "OUTPUT_DIR", str(tmp_path))

        prices, universe = _cointegrated_prices(n=400)
        series_a = prices["T0"]
        series_b = prices["T1"]
        from src.spread import compute_beta, compute_spread, smooth_spread, rolling_stats
        beta     = compute_beta(series_a, series_b)
        smoothed = smooth_spread(compute_spread(series_a, series_b, beta), cfg.EMA_SPAN)
        mu, sigma = rolling_stats(smoothed, 200)
        sigma_safe = sigma.where(sigma > 1e-8)
        z = (smoothed - mu) / sigma_safe
        # Inject a synthetic force-closed trade
        idx = z.dropna().index
        fake_trade = {
            "entry_date": idx[0], "exit_date": idx[-1],
            "direction": "long", "entry_z": -1.5, "exit_z": 0.5,
            "pnl": 5.0, "sigma": 1.0, "duration": len(idx),
            "entry_spread": float(smoothed.loc[idx[0]]),
            "exit_spread":  float(smoothed.loc[idx[-1]]),
            "force_close": True,
        }
        params = {
            "open_long": 1.5, "open_short": 1.5,
            "close_long": 0.5, "close_short": 0.5,
            "exit_step": 0.1, "window": 200, "stop_loss": 2.0,
        }
        sol = dict(ticker_a="T0", ticker_b="T1", params=params,
                   z=z, trades=[fake_trade], smoothed=smoothed)

        import matplotlib
        matplotlib.use("Agg")
        from src.plot import plot_top5_static
        path = plot_top5_static([sol])

        import os
        assert os.path.exists(path)

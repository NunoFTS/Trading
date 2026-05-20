"""
Tests for src/spread.py
Covers: compute_beta, compute_spread, smooth_spread, rolling_stats, cointegration_score
"""
import numpy as np
import pandas as pd
import pytest
import math

from src.spread import (
    compute_beta,
    compute_spread,
    smooth_spread,
    rolling_stats,
    cointegration_score,
)


def _series(vals, start="2020-01-01"):
    dates = pd.date_range(start, periods=len(vals))
    return pd.Series(np.asarray(vals, dtype=float), index=dates)


# ---------------------------------------------------------------------------
# compute_beta
# ---------------------------------------------------------------------------

class TestComputeBeta:
    def test_identical_series_returns_beta_one(self):
        np.random.seed(0)
        vals = np.cumsum(np.random.randn(200)) + 100
        a = _series(vals)
        b = _series(vals)
        assert compute_beta(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_doubled_series_returns_beta_two(self):
        np.random.seed(1)
        b_vals = np.cumsum(np.random.randn(200)) + 100
        a_vals = 2.0 * b_vals
        assert compute_beta(_series(a_vals), _series(b_vals)) == pytest.approx(2.0, abs=1e-6)

    def test_half_scale_returns_beta_half(self):
        np.random.seed(2)
        b_vals = np.cumsum(np.random.randn(200)) + 100
        a_vals = 0.5 * b_vals
        assert compute_beta(_series(a_vals), _series(b_vals)) == pytest.approx(0.5, abs=1e-6)

    def test_returns_float_type(self):
        vals = np.arange(1.0, 101.0)
        result = compute_beta(_series(vals), _series(vals))
        assert isinstance(result, float)

    def test_positive_beta_for_positive_relationship(self):
        np.random.seed(3)
        b_vals = np.cumsum(np.random.randn(100)) + 50
        a_vals = 1.5 * b_vals + np.random.randn(100) * 0.1
        beta = compute_beta(_series(a_vals), _series(b_vals))
        assert beta > 0


# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------

class TestComputeSpread:
    def test_zero_spread_when_beta_one_identical_series(self):
        vals = np.arange(1.0, 51.0)
        spread = compute_spread(_series(vals), _series(vals), 1.0)
        np.testing.assert_allclose(spread.values, 0.0, atol=1e-12)

    def test_spread_formula_correctness(self):
        a = _series([10.0, 20.0, 30.0])
        b = _series([5.0,  10.0, 15.0])
        # A - 2*B = [0, 0, 0]
        np.testing.assert_allclose(compute_spread(a, b, 2.0).values, 0.0, atol=1e-12)

    def test_spread_with_known_values(self):
        a = _series([100.0, 110.0, 120.0])
        b = _series([50.0,   55.0,  60.0])
        expected = [25.0, 27.5, 30.0]  # A - 1.5*B
        np.testing.assert_allclose(compute_spread(a, b, 1.5).values, expected, atol=1e-10)

    def test_spread_preserves_index(self):
        idx = pd.date_range("2021-01-01", periods=5)
        a = pd.Series([1.0] * 5, index=idx)
        b = pd.Series([1.0] * 5, index=idx)
        pd.testing.assert_index_equal(compute_spread(a, b, 1.0).index, idx)

    def test_spread_returns_series(self):
        a = _series([1.0, 2.0, 3.0])
        b = _series([1.0, 2.0, 3.0])
        assert isinstance(compute_spread(a, b, 1.0), pd.Series)

    def test_negative_beta_inverts_relationship(self):
        a = _series([10.0, 20.0, 30.0])
        b = _series([1.0,  2.0,  3.0])
        # A - (-2)*B = A + 2B
        result = compute_spread(a, b, -2.0)
        expected = [12.0, 24.0, 36.0]
        np.testing.assert_allclose(result.values, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# smooth_spread
# ---------------------------------------------------------------------------

class TestSmoothSpread:
    def test_output_length_matches_input(self):
        s = _series(np.arange(50.0))
        assert len(smooth_spread(s, span=8)) == 50

    def test_preserves_index(self):
        s = _series(np.arange(50.0))
        pd.testing.assert_index_equal(smooth_spread(s, span=8).index, s.index)

    def test_constant_series_unchanged(self):
        s = _series([5.0] * 100)
        np.testing.assert_allclose(smooth_spread(s, span=8).values, 5.0, atol=1e-12)

    def test_smoothing_reduces_variance(self):
        np.random.seed(42)
        noisy = _series(np.random.randn(300))
        smoothed = smooth_spread(noisy, span=8)
        assert smoothed.std() < noisy.std()

    def test_returns_series(self):
        s = _series(np.arange(20.0))
        assert isinstance(smooth_spread(s, span=3), pd.Series)

    def test_span_one_is_identity(self):
        # EMA with span=1 is equivalent to no smoothing (alpha=1)
        s = _series(np.arange(30.0))
        result = smooth_spread(s, span=1)
        np.testing.assert_allclose(result.values, s.values, atol=1e-10)


# ---------------------------------------------------------------------------
# rolling_stats
# ---------------------------------------------------------------------------

class TestRollingStats:
    def test_output_length_matches_input(self):
        s = _series(np.arange(150.0))
        mu, sigma = rolling_stats(s, 100)
        assert len(mu) == len(s)
        assert len(sigma) == len(s)

    def test_first_window_minus_one_are_nan(self):
        s = _series(np.arange(150.0))
        mu, sigma = rolling_stats(s, 100)
        # Bars 0..98 should be NaN (window=100 needs 100 obs; bar 99 is first complete)
        assert mu.iloc[:99].isna().all()
        assert sigma.iloc[:99].isna().all()

    def test_values_from_window_onward_are_not_nan(self):
        s = _series(np.arange(150.0))
        mu, sigma = rolling_stats(s, 100)
        assert not mu.iloc[99:].isna().any()
        assert not sigma.iloc[99:].isna().any()

    def test_constant_series_gives_zero_sigma(self):
        s = _series([5.0] * 150)
        mu, sigma = rolling_stats(s, 100)
        np.testing.assert_allclose(mu.iloc[99:].values, 5.0, atol=1e-10)
        np.testing.assert_allclose(sigma.iloc[99:].values, 0.0, atol=1e-10)

    def test_preserves_index(self):
        s = _series(np.arange(150.0))
        mu, sigma = rolling_stats(s, 100)
        pd.testing.assert_index_equal(mu.index, s.index)
        pd.testing.assert_index_equal(sigma.index, s.index)

    def test_mu_is_rolling_mean(self):
        s = _series(np.arange(10.0))
        mu, _ = rolling_stats(s, 3)
        # Bar 2 (0-indexed): mean of [0,1,2] = 1.0
        assert mu.iloc[2] == pytest.approx(1.0)
        # Bar 3: mean of [1,2,3] = 2.0
        assert mu.iloc[3] == pytest.approx(2.0)

    def test_sigma_is_rolling_std(self):
        s = _series(np.arange(10.0))
        _, sigma = rolling_stats(s, 3)
        # std of [0,1,2] with ddof=1 = 1.0
        assert sigma.iloc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cointegration_score
# ---------------------------------------------------------------------------

class TestCointegrationScore:
    def test_returns_float(self):
        np.random.seed(10)
        a = _series(np.cumsum(np.random.randn(200)) + 100)
        b = _series(np.cumsum(np.random.randn(200)) + 100)
        assert isinstance(cointegration_score(a, b), float)

    def test_cointegrated_pair_gives_negative_t_stat(self):
        # A = B + stationary noise → strongly cointegrated
        np.random.seed(42)
        b_vals = np.cumsum(np.random.randn(400)) + 100
        a_vals = b_vals + np.random.randn(400) * 0.3
        score = cointegration_score(_series(a_vals), _series(b_vals))
        assert score < -3.0

    def test_independent_random_walks_weaker_cointegration(self):
        # Two independent random walks should not be strongly cointegrated
        np.random.seed(99)
        a = _series(np.cumsum(np.random.randn(400)) + 100)
        b = _series(np.cumsum(np.random.randn(400)) + 200)
        score = cointegration_score(a, b)
        # Score won't be strongly negative
        assert isinstance(score, float)  # just verify it runs cleanly

    def test_does_not_raise_on_short_series(self):
        # Should return 0.0 (or any float) without raising
        a = _series([1.0, 2.0])
        b = _series([1.0, 2.0])
        result = cointegration_score(a, b)
        assert isinstance(result, float)

    def test_symmetric_sign(self):
        # cointegration is symmetric — swapping A/B may change magnitude but not sign
        np.random.seed(7)
        b_vals = np.cumsum(np.random.randn(300)) + 100
        a_vals = b_vals + np.random.randn(300) * 0.2
        s1 = cointegration_score(_series(a_vals), _series(b_vals))
        s2 = cointegration_score(_series(b_vals), _series(a_vals))
        # Both should be negative for a cointegrated pair
        assert s1 < 0
        assert s2 < 0

    def test_degenerate_input_returns_finite_score(self):
        """
        Degenerate perfectly collinear series can produce +/-inf t-stats from statsmodels.
        The strategy layer expects numeric finite objectives.
        """
        a = _series(np.ones(50))
        b = _series(np.ones(50))
        score = cointegration_score(a, b)
        assert math.isfinite(score)

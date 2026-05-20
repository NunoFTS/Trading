"""
Tests for src/data_loader.py — load_prices().

Uses a temporary CSV file built in each test so no real data file is required.
"""
import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_prices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(rows, path):
    """Write a list-of-dict rows to a CSV at path."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _basic_csv(
    tickers=("AA", "BB", "CC"),
    n_days=100,
    start="2022-01-03",
    missing_ticker=None,
    missing_fraction=0.0,
):
    """
    Build a minimal CSV with columns: Date, Symbol, Adj Close.
    If missing_ticker is set, that ticker will have `missing_fraction` of its
    rows removed (to test the 80% threshold drop).
    """
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    np.random.seed(0)
    for ticker in tickers:
        prices = np.cumsum(np.random.randn(n_days)) + 100
        for i, (d, p) in enumerate(zip(dates, prices)):
            if ticker == missing_ticker and i < int(n_days * missing_fraction):
                continue   # omit this row (simulates NaN after pivot)
            rows.append({"Date": d.strftime("%Y-%m-%d"), "Symbol": ticker, "Adj Close": p})
    return rows


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestLoadPricesBasic:
    def test_returns_dataframe_and_list(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_csv(_basic_csv(), csv_path)
        prices, universe = load_prices(str(csv_path))
        assert isinstance(prices, pd.DataFrame)
        assert isinstance(universe, list)

    def test_index_is_datetime(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_csv(_basic_csv(), csv_path)
        prices, _ = load_prices(str(csv_path))
        assert isinstance(prices.index, pd.DatetimeIndex)

    def test_index_is_sorted_ascending(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_csv(_basic_csv(), csv_path)
        prices, _ = load_prices(str(csv_path))
        assert prices.index.is_monotonic_increasing

    def test_columns_are_ticker_symbols(self, tmp_path):
        tickers = ("AA", "BB", "CC")
        csv_path = tmp_path / "prices.csv"
        _write_csv(_basic_csv(tickers=tickers), csv_path)
        prices, universe = load_prices(str(csv_path))
        assert set(prices.columns) == set(tickers)
        assert set(universe) == set(tickers)

    def test_universe_matches_columns(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_csv(_basic_csv(), csv_path)
        prices, universe = load_prices(str(csv_path))
        assert sorted(universe) == sorted(prices.columns.tolist())

    def test_row_count_matches_trading_days(self, tmp_path):
        n = 60
        csv_path = tmp_path / "prices.csv"
        _write_csv(_basic_csv(n_days=n), csv_path)
        prices, _ = load_prices(str(csv_path))
        assert len(prices) == n

    def test_values_are_numeric(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_csv(_basic_csv(), csv_path)
        prices, _ = load_prices(str(csv_path))
        assert prices.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()


# ---------------------------------------------------------------------------
# Forward-fill behaviour
# ---------------------------------------------------------------------------

class TestForwardFill:
    def test_single_nan_filled(self, tmp_path):
        """
        A single missing row for one ticker should be forward-filled, not NaN.
        """
        tickers = ("AA", "BB")
        n = 20
        dates = pd.date_range("2022-01-03", periods=n, freq="B")
        np.random.seed(1)
        rows = []
        for ticker in tickers:
            prices = np.cumsum(np.random.randn(n)) + 100
            for i, (d, p) in enumerate(zip(dates, prices)):
                if ticker == "AA" and i == 5:
                    continue   # row 5 missing for AA
                rows.append({"Date": d.strftime("%Y-%m-%d"), "Symbol": ticker, "Adj Close": p})
        csv_path = tmp_path / "prices.csv"
        _write_csv(rows, csv_path)
        prices_df, _ = load_prices(str(csv_path))
        # After ffill, no NaN should remain in AA (single gap is filled)
        assert not prices_df["AA"].isna().any()

    def test_ffill_preserves_last_known_value(self, tmp_path):
        """Forward-fill uses the last known value."""
        tickers = ("AA", "BB")
        rows = [
            {"Date": "2022-01-03", "Symbol": "AA", "Adj Close": 50.0},
            {"Date": "2022-01-03", "Symbol": "BB", "Adj Close": 60.0},
            # AA missing on 2022-01-04
            {"Date": "2022-01-04", "Symbol": "BB", "Adj Close": 61.0},
            {"Date": "2022-01-05", "Symbol": "AA", "Adj Close": 52.0},
            {"Date": "2022-01-05", "Symbol": "BB", "Adj Close": 62.0},
        ]
        csv_path = tmp_path / "prices.csv"
        _write_csv(rows, csv_path)
        prices_df, _ = load_prices(str(csv_path))
        # AA on 2022-01-04 should be forward-filled from 50.0
        assert prices_df.loc["2022-01-04", "AA"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Sparse ticker drop (80% threshold)
# ---------------------------------------------------------------------------

class TestSparseDrop:
    def test_ticker_with_over_20pct_missing_is_dropped(self, tmp_path):
        """
        A ticker missing >20% of rows should be dropped from the final DataFrame.
        """
        n = 100
        # Ticker "SPARSE" has 30% rows omitted → missing_fraction=0.30
        rows = _basic_csv(
            tickers=("AA", "BB", "SPARSE"),
            n_days=n,
            missing_ticker="SPARSE",
            missing_fraction=0.30,
        )
        csv_path = tmp_path / "prices.csv"
        _write_csv(rows, csv_path)
        prices, universe = load_prices(str(csv_path))
        assert "SPARSE" not in prices.columns
        assert "SPARSE" not in universe

    def test_ticker_with_under_20pct_missing_is_kept(self, tmp_path):
        """
        A ticker missing only 10% of rows should survive after forward-fill.
        """
        n = 100
        rows = _basic_csv(
            tickers=("AA", "BB", "SPARSE"),
            n_days=n,
            missing_ticker="SPARSE",
            missing_fraction=0.10,
        )
        csv_path = tmp_path / "prices.csv"
        _write_csv(rows, csv_path)
        prices, universe = load_prices(str(csv_path))
        assert "SPARSE" in prices.columns

    def test_dense_tickers_retained_when_sparse_dropped(self, tmp_path):
        """Dense tickers must not be removed when a sparse one is dropped."""
        n = 100
        rows = _basic_csv(
            tickers=("AA", "BB", "SPARSE"),
            n_days=n,
            missing_ticker="SPARSE",
            missing_fraction=0.30,
        )
        csv_path = tmp_path / "prices.csv"
        _write_csv(rows, csv_path)
        prices, _ = load_prices(str(csv_path))
        assert "AA" in prices.columns
        assert "BB" in prices.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_ticker(self, tmp_path):
        rows = _basic_csv(tickers=("ONLY",), n_days=50)
        csv_path = tmp_path / "prices.csv"
        _write_csv(rows, csv_path)
        prices, universe = load_prices(str(csv_path))
        assert "ONLY" in prices.columns
        assert len(universe) == 1

    def test_unsorted_dates_sorted_on_load(self, tmp_path):
        """CSV with dates out of order should produce a sorted index."""
        rows = [
            {"Date": "2022-01-05", "Symbol": "AA", "Adj Close": 52.0},
            {"Date": "2022-01-03", "Symbol": "AA", "Adj Close": 50.0},
            {"Date": "2022-01-04", "Symbol": "AA", "Adj Close": 51.0},
        ]
        csv_path = tmp_path / "prices.csv"
        _write_csv(rows, csv_path)
        prices, _ = load_prices(str(csv_path))
        assert prices.index.is_monotonic_increasing

    def test_custom_path_used_when_provided(self, tmp_path):
        """load_prices(path=...) uses the given path, not config.DATA_PATH."""
        csv_path = tmp_path / "custom.csv"
        _write_csv(_basic_csv(tickers=("CUSTOM",), n_days=30), csv_path)
        prices, universe = load_prices(str(csv_path))
        assert "CUSTOM" in universe

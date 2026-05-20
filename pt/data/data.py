import pandas as pd
from pathlib import Path

SRC = Path(__file__).parent / "sp500_stocks.csv"
OUT = Path(__file__).parent / "sp500_4yr_train.csv"
COVERAGE_THRESHOLD = 0.90

print("Loading sp500_stocks.csv ...")
df = pd.read_csv(SRC, parse_dates=["Date"])

# Drop rows where all price/volume columns are missing
price_cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
df = df.dropna(subset=price_cols, how="all")

# Determine 2-year window from the most recent date in the file
max_date = df["Date"].max()
start_date = max_date - pd.DateOffset(years=4)
print(f"Date range in source: {df['Date'].min().date()} → {max_date.date()}")
print(f"Filtering to: {start_date.date()} → {max_date.date()}")

df = df[(df["Date"] >= start_date) & (df["Date"] <= max_date)].copy()

# Count total unique trading days in this window
total_days = df["Date"].nunique()
print(f"Total trading days in window: {total_days}")

# Keep symbols with >= 90% coverage
counts = df.groupby("Symbol")["Date"].count()
qualified = counts[counts >= total_days * COVERAGE_THRESHOLD].index
df = df[df["Symbol"].isin(qualified)]

df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

df.to_csv(OUT, index=False)

print(f"\nSaved: {OUT}")
print(f"Stocks included : {df['Symbol'].nunique()} / {len(counts)}")
print(f"Total rows      : {len(df):,}")
print(f"Date range      : {df['Date'].min().date()} → {df['Date'].max().date()}")

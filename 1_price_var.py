"""
Enriches sp500_news.csv with daily price data fetched via yfinance.

Added columns:
  - price_var   : close(t) - close(t-1)
  - prc_var     : price_var / close(t-1)  expressed as a fraction (e.g. 0.012 = 1.2%)
  - var_class   : -1 if prc_var < -1%, 0 if between -1% and 1%, +1 if > 1%

Requirements:
    pip install yfinance pandas
"""

import datetime
import time
import pandas as pd
import yfinance as yf

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "sp500_news.csv"
OUTPUT_FILE  = "sp500_news_with_prices.csv"
THRESHOLD    = 0.01        # 1%
SLEEP_BETWEEN = 0.01        # seconds between yfinance calls (be polite)
# ─────────────────────────────────────────────────────────────────────────────


def get_daily_returns(ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker and return a DataFrame indexed by date
    with columns: price_var, prc_var, var_class.
    Returns an empty DataFrame on failure.
    """
    try:
        # Download with a 1-day buffer before start_date to allow diff() on first row
        raw = yf.download(
            ticker,
            start=start_date - datetime.timedelta(days=5),
            end=end_date + datetime.timedelta(days=1),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return pd.DataFrame()

    if raw.empty or len(raw) < 2:
        return pd.DataFrame()

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    close = raw["Close"]
    price_var = close.diff()
    prc_var   = price_var / close.shift(1)

    df = pd.DataFrame({
        "price_var": price_var,
        "prc_var":   prc_var,
    })
    df.index = pd.to_datetime(df.index).date   # convert to plain date objects
    df = df.dropna()

    def classify(x):
        if x < -THRESHOLD:
            return -1
        elif x > THRESHOLD:
            return 1
        else:
            return 0

    df["var_class"] = df["prc_var"].apply(classify)
    return df


def main():
    # ── Load news CSV ────────────────────────────────────────────────────────
    news = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    news["Date"] = news["Date"].dt.date
    print(f"Loaded {len(news)} rows from '{INPUT_FILE}'")

    tickers    = news["Ticker"].unique()
    start_date = news["Date"].min()
    end_date   = news["Date"].max()
    print(f"Tickers: {len(tickers)} | Period: {start_date} → {end_date}")

    # ── Fetch price data for every ticker ───────────────────────────────────
    price_rows = []
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i:>3}/{len(tickers)}] {ticker} …", end=" ", flush=True)
        df = get_daily_returns(ticker, start_date, end_date)
        if df.empty:
            print("no data")
        else:
            df["Ticker"] = ticker
            df["Date"]   = df.index
            price_rows.append(df.reset_index(drop=True))
            print(f"{len(df)} days")
        time.sleep(SLEEP_BETWEEN)

    if not price_rows:
        print("No price data retrieved. Exiting.")
        return

    prices = pd.concat(price_rows, ignore_index=True)

    # ── Merge with news ──────────────────────────────────────────────────────
    enriched = news.merge(prices[["Date", "Ticker", "price_var", "prc_var", "var_class"]],
                          on=["Date", "Ticker"],
                          how="left")

    missing = enriched["var_class"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows have no price data (weekends, holidays, or missing tickers)")

    # ── Save ─────────────────────────────────────────────────────────────────
    enriched.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(enriched)} rows saved to '{OUTPUT_FILE}'")
    print(enriched.head())


if __name__ == "__main__":
    main()
"""
Fetch daily news headlines for all S&P 500 stocks over the last 3 months
using the Finnhub free API, with rate-limit-safe pauses.

Output: sp500_news.csv  (columns: Date, Ticker, Title)

Requirements:
    pip install finnhub-python pandas requests
"""

import time
import datetime
import pandas as pd
import finnhub

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY       = "d7pmvf9r01qosaap6qp0d7pmvf9r01qosaap6qpg"   # free key at https://finnhub.io
OUTPUT_FILE   = "sp500_news.csv"
MONTHS_BACK   = 3
REQUESTS_PER_MIN = 55          # free tier allows 60; use 55 to stay safe
SLEEP_BETWEEN  = 60 / REQUESTS_PER_MIN   # ~1.09 s between requests
# ─────────────────────────────────────────────────────────────────────────────


def get_sp500_tickers() -> list[str]:
    """Scrape S&P 500 tickers from Wikipedia (with browser User-Agent to avoid 403)."""
    import requests
    from io import StringIO

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df = pd.read_html(StringIO(response.text))[0]
    # Wikipedia uses '.' in some tickers (e.g. BRK.B); Finnhub expects '-' (BRK-B)
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def date_to_unix(d: datetime.date) -> int:
    return int(datetime.datetime(d.year, d.month, d.day).timestamp())


def fetch_news_for_ticker(
    client: finnhub.Client,
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[dict]:
    """
    Call Finnhub company-news endpoint and return a flat list of
    {Date, Ticker, Title} dicts.
    """
    try:
        articles = client.company_news(
            ticker,
            _from=str(start_date),
            to=str(end_date),
        )
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return []

    rows = []
    for article in articles:
        ts = article.get("datetime", 0)
        if not ts:
            continue
        article_date = datetime.datetime.utcfromtimestamp(ts).date()
        # Discard anything outside the requested window
        if not (start_date <= article_date <= end_date):
            continue
        title = article.get("headline", "").strip()
        if title:
            rows.append({"Date": article_date.strftime("%Y-%m-%d"), "Ticker": ticker, "Title": title})
    return rows


def main():
    # ── Date range ──────────────────────────────────────────────────────────
    end_date   = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=MONTHS_BACK * 31)
    print(f"Fetching news from {start_date} to {end_date}")

    # ── Tickers ─────────────────────────────────────────────────────────────
    print("Loading S&P 500 tickers …")
    tickers = get_sp500_tickers()
    print(f"  {len(tickers)} tickers found")

    # ── Finnhub client ───────────────────────────────────────────────────────
    client = finnhub.Client(api_key=API_KEY)

    # ── Main loop ────────────────────────────────────────────────────────────
    all_rows = []
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i:>3}/{len(tickers)}] {ticker} …", end=" ", flush=True)

        rows = fetch_news_for_ticker(client, ticker, start_date, end_date)
        all_rows.extend(rows)
        print(f"{len(rows)} articles")

        # Rate-limit pause (free tier: 60 req/min)
        time.sleep(SLEEP_BETWEEN)

        # Extra courtesy pause every 50 tickers to avoid any burst penalties
        if i % 50 == 0:
            print("  [pause 10 s …]")
            time.sleep(10)

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows, columns=["Date", "Ticker", "Title"])
    df = df.sort_values(["Date", "Ticker"]).drop_duplicates()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(df)} rows saved to '{OUTPUT_FILE}'")
    print(df.head())


if __name__ == "__main__":
    main()
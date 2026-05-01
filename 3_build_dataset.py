"""
Builds the final training dataset from the enriched news+price CSV.

Logic:
  - Drop rows with NaN price data
  - Group by (Date, Ticker): concatenate all headlines into one string
  - Keep only (Date, Ticker) pairs with >= 5 articles
  - Output columns: Date, Ticker, titles, price_var, prc_var, var_class

Requirements:
    pip install pandas
"""

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE    = "sp500_news_with_prices.csv"
OUTPUT_FILE   = "sp500_final_dataset.csv"
MIN_ARTICLES  = 5
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Load & clean ─────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    print(f"Rows before dropna : {len(df)}")
    df = df.dropna(subset=["var_class", "price_var", "prc_var"])
    print(f"Rows after  dropna : {len(df)}")

    df["var_class"] = df["var_class"].astype(int)

    # ── Aggregate headlines per (Date, Ticker) ────────────────────────────────
    agg = (
        df.groupby(["Date", "Ticker"])
        .agg(
            n_articles=("Title",     "count"),
            titles    =("Title",     lambda x: " ".join(x)),
            price_var =("price_var", "first"),
            prc_var   =("prc_var",   "first"),
            var_class =("var_class", "first"),
        )
        .reset_index()
    )
    print(f"Samples before threshold filter : {len(agg)}")

    # ── Apply threshold ───────────────────────────────────────────────────────
    final = agg[agg["n_articles"] >= MIN_ARTICLES].copy()
    final = final.drop(columns=["n_articles"])
    final = final.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(f"Samples after  threshold filter : {len(final)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = final["var_class"].value_counts().sort_index()
    label_map = {-1: "DOWN", 0: "FLAT", 1: "UP"}
    print("\nClass distribution:")
    for cls, cnt in counts.items():
        pct = cnt / len(final) * 100
        print(f"  {label_map[cls]:>4} ({cls:+d}) : {cnt:>5,}  ({pct:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal dataset saved to '{OUTPUT_FILE}'  —  {len(final)} samples")
    print(final.head())


if __name__ == "__main__":
    main()
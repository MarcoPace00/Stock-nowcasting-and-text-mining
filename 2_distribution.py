"""
Loads the enriched news+price CSV, drops NaN rows, aggregates headlines
per (Date, Ticker), and plots the target-class distribution for multiple
minimum-article-count thresholds — mirroring the dataset exploration in
the original notebook.

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE = "sp500_news_with_prices.csv"
THRESHOLDS = [1, 2, 3, 5, 7, 10]   # min number of articles per ticker per day
CLASS_LABELS = {-1: "DOWN", 0: "FLAT", 1: "UP"}
CLASS_COLORS = {-1: "#e05c5c", 0: "#f0c040", 1: "#5ca85c"}
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Load & clean ─────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    print(f"Rows before dropna : {len(df)}")
    df = df.dropna(subset=["var_class"])
    print(f"Rows after  dropna : {len(df)}")

    df["var_class"] = df["var_class"].astype(int)

    # ── Aggregate: one row per (Date, Ticker) ────────────────────────────────
    # Concatenate all titles into one string; keep var_class (same for all rows
    # of the same ticker/day by construction)
    agg = (
        df.groupby(["Date", "Ticker"])
        .agg(
            n_articles=("Title", "count"),
            titles=("Title", lambda x: " ".join(x)),
            var_class=("var_class", "first"),
        )
        .reset_index()
    )
    print(f"Unique (Date, Ticker) samples : {len(agg)}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(
        2, n_thresh,
        figsize=(4 * n_thresh, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle(
        "Class distribution vs. minimum articles per ticker per day",
        fontsize=14, fontweight="bold", y=1.01,
    )

    classes = sorted(CLASS_LABELS.keys())   # [-1, 0, 1]

    for col, thr in enumerate(THRESHOLDS):
        subset = agg[agg["n_articles"] >= thr]
        counts = subset["var_class"].value_counts().reindex(classes, fill_value=0)
        total  = counts.sum()
        pcts   = counts / total * 100 if total > 0 else counts * 0

        colors = [CLASS_COLORS[c] for c in classes]
        labels = [CLASS_LABELS[c] for c in classes]

        # ── Top: absolute counts bar chart ───────────────────────────────────
        ax_bar = axes[0, col]
        bars = ax_bar.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
        ax_bar.set_title(f"≥ {thr} article{'s' if thr > 1 else ''}\n({total:,} samples)", fontsize=10)
        ax_bar.set_ylabel("Count" if col == 0 else "")
        ax_bar.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_bar.spines[["top", "right"]].set_visible(False)

        # Annotate bars with count
        for bar, cnt in zip(bars, counts.values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{cnt:,}",
                ha="center", va="bottom", fontsize=8,
            )

        # ── Bottom: percentage pie chart ─────────────────────────────────────
        ax_pie = axes[1, col]
        if total > 0:
            wedges, texts, autotexts = ax_pie.pie(
                counts.values,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )
            for at in autotexts:
                at.set_fontsize(7)
        else:
            ax_pie.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_pie.axis("off")

    plt.tight_layout()
    out_path = "class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to '{out_path}'")
    plt.show()


if __name__ == "__main__":
    main()
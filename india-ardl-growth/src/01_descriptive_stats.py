"""
01_descriptive_stats.py
------------------------
Replicates Table 1 (Summary Statistics) and Table 2 (Pearson Correlation
Matrix) of the paper, and produces the time-series plots referenced in
Section 4.3.1 (Graphical Analysis).
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "india_gdp_gfcf_gds_1990_2025.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

COLS = ["GDP_Growth", "GFCF_Growth", "GDS_pct_GDP"]


def summary_table(df):
    rows = []
    for c in COLS:
        s = df[c]
        jb_stat, jb_p = stats.jarque_bera(s)
        rows.append({
            "Variable": c,
            "Observations": s.count(),
            "Mean": round(s.mean(), 2),
            "Std. Dev.": round(s.std(ddof=1), 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
            "Jarque-Bera (p-value)": round(jb_p, 2),
        })
    return pd.DataFrame(rows)


def correlation_table(df):
    return df[COLS].corr().round(2)


def plot_series(df):
    labels = {
        "GDP_Growth": "GDP Growth (%)",
        "GFCF_Growth": "GFCF Growth (%)",
        "GDS_pct_GDP": "GDS (% of GDP)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, c in zip(axes, COLS):
        ax.plot(df["Year"], df[c], marker="o", markersize=3, linewidth=1)
        ax.set_title(labels[c])
        ax.axhline(df[c].mean(), color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "series_levels.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["Year"][1:], df["GDS_pct_GDP"].diff().dropna(), marker="o", markersize=3)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("D(GDS), first difference")
    ax.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gds_first_difference.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA)

    summ = summary_table(df)
    corr = correlation_table(df)

    print("Table 1: Summary Statistics\n", summ.to_string(index=False))
    print("\nTable 2: Pearson Correlation Matrix\n", corr)

    summ.to_csv(TAB_DIR / "table1_summary_statistics.csv", index=False)
    corr.to_csv(TAB_DIR / "table2_correlation_matrix.csv")

    plot_series(df)
    print(f"\nFigures saved to {FIG_DIR}")
    print(f"Tables saved to {TAB_DIR}")


if __name__ == "__main__":
    main()

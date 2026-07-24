"""
00_generate_data.py
--------------------
Generates data/india_gdp_gfcf_gds_1990_2025.csv.

IMPORTANT — DATA PROVENANCE:
The original paper ("The Impact of Gross Fixed Capital Formation and Gross
Domestic Savings on Economic Growth in India: An ARDL Approach, 1990-2025")
was built on data compiled from RBI's Handbook of Statistics on the Indian
Economy, MoSPI's National Account Statistics, and NITI Aayog, then analysed
in EViews 9. That raw EViews workfile is not part of this repository.

To make the analysis in this repo fully reproducible end-to-end, this script
generates a CALIBRATED SYNTHETIC series: it searches over random seeds until
it finds a series whose descriptive statistics (mean, std. dev., min, max)
and Pearson correlation matrix closely match Table 1 and Table 2 of the
paper. It is a stand-in for the real series, built ONLY so that the rest of
the pipeline (stationarity tests, ARDL, bounds test, diagnostics) can be run
and produces results of the same qualitative shape as the paper. It should
NOT be cited as real macroeconomic data — see docs/DATA_NOTES.md.

Targets (from the paper):
  GDP growth (%):   mean 6.15, sd 2.80, min -5.77, max 9.68   -> I(0)
  GFCF growth (%):  mean 7.81, sd 6.75, min -7.70, max 22.19  -> I(0)
  GDS (% of GDP):   mean 28.80, sd 4.01, min 21.63, max 34.37 -> I(1), trending up
  Correlations:     corr(GDP,GFCF)=0.62, corr(GDP,GDS)=0.35, corr(GFCF,GDS)=0.15
"""

import numpy as np
import pandas as pd
from pathlib import Path

N = 36
YEARS = np.arange(1990, 1990 + N)  # 1990..2025 inclusive

TARGET_CORR = np.array([
    [1.00, 0.62, 0.35],
    [0.62, 1.00, 0.15],
    [0.35, 0.15, 1.00],
])

TARGETS = {
    "GDP_Growth": dict(mean=6.15, std=2.80, min=-5.77, max=9.68),
    "GFCF_Growth": dict(mean=7.81, std=6.75, min=-7.70, max=22.19),
    "GDS_pct_GDP": dict(mean=28.80, std=4.01, min=21.63, max=34.37),
}


def score(df):
    """Lower is better: weighted distance from target moments + correlations."""
    err = 0.0
    for col, t in TARGETS.items():
        s = df[col]
        err += (s.mean() - t["mean"]) ** 2
        err += (s.std(ddof=1) - t["std"]) ** 2
        err += 0.25 * (s.min() - t["min"]) ** 2
        err += 0.25 * (s.max() - t["max"]) ** 2
    corr = df[list(TARGETS)].corr().values
    err += 4 * np.sum((corr - TARGET_CORR) ** 2)
    return err


def build_series(seed):
    rng = np.random.default_rng(seed)

    # Correlated standard-normal draws for GDP growth, GFCF growth, and the
    # first difference of GDS (kept stationary; GDS itself is then built as
    # a random walk with drift so it is I(1), matching the paper's ADF/PP
    # results).
    L = np.linalg.cholesky(TARGET_CORR)
    z = rng.standard_normal((N, 3)) @ L.T

    gdp = TARGETS["GDP_Growth"]["mean"] + TARGETS["GDP_Growth"]["std"] * z[:, 0]
    gfcf = TARGETS["GFCF_Growth"]["mean"] + TARGETS["GFCF_Growth"]["std"] * z[:, 1]

    # GDS: a genuine stochastic trend (random walk with drift, i.e. a true
    # unit root) built by cumulating correlated innovations, then
    # standardised to the reported mean/std. This is what makes GDS
    # non-stationary at level but stationary after first-differencing --
    # a deterministic trend would not reliably reproduce that ADF/PP
    # behaviour on a short annual sample.
    drift = 0.12
    d_gds = drift + z[:, 2]
    gds_level = np.cumsum(d_gds)
    gds_level = (gds_level - gds_level.mean()) / gds_level.std(ddof=1)
    gds = TARGETS["GDS_pct_GDP"]["mean"] + gds_level * TARGETS["GDS_pct_GDP"]["std"]

    df = pd.DataFrame({"Year": YEARS, "GDP_Growth": gdp, "GFCF_Growth": gfcf, "GDS_pct_GDP": gds})
    return df


def clip_to_range(df):
    for col, t in TARGETS.items():
        df[col] = df[col].clip(lower=t["min"], upper=t["max"])
    return df


def main():
    best = None
    best_score = np.inf
    for seed in range(20000):
        df = build_series(seed)
        s = score(df)
        if s < best_score:
            best_score = s
            best = df.copy()
        if best_score < 0.35:
            break

    df = clip_to_range(best)
    # Nudge exact min/max onto the target years so reported extremes match
    df.loc[df["GDP_Growth"].idxmin(), "GDP_Growth"] = TARGETS["GDP_Growth"]["min"]
    df.loc[df["GDP_Growth"].idxmax(), "GDP_Growth"] = TARGETS["GDP_Growth"]["max"]
    df.loc[df["GFCF_Growth"].idxmin(), "GFCF_Growth"] = TARGETS["GFCF_Growth"]["min"]
    df.loc[df["GFCF_Growth"].idxmax(), "GFCF_Growth"] = TARGETS["GFCF_Growth"]["max"]
    df.loc[df["GDS_pct_GDP"].idxmin(), "GDS_pct_GDP"] = TARGETS["GDS_pct_GDP"]["min"]
    df.loc[df["GDS_pct_GDP"].idxmax(), "GDS_pct_GDP"] = TARGETS["GDS_pct_GDP"]["max"]

    out = Path(__file__).resolve().parents[1] / "data" / "india_gdp_gfcf_gds_1990_2025.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(out, index=False)

    print(f"best score: {best_score:.4f}")
    print(df[list(TARGETS)].describe())
    print(df[list(TARGETS)].corr())
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

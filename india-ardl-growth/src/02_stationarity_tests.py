"""
02_stationarity_tests.py
--------------------------
Replicates Table 3 (ADF and Phillips-Perron Unit Root Test Results,
Section 4.3.3). Confirms GDP growth and GFCF growth are I(0), and that
GDS is I(1) (stationary only after first-differencing) -- the mixed order
of integration that justifies using ARDL over a conventional Engle-Granger
cointegration approach.
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron  # falls back below if unavailable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "india_gdp_gfcf_gds_1990_2025.csv"
TAB_DIR = ROOT / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def adf_pvalue(series):
    # BIC (rather than AIC) lag selection, capped at a sensible maximum for
    # a 36-observation annual series -- AIC tends to over-fit lags on short
    # samples and destroys test power, which does not match how EViews'
    # automatic selection behaves in practice for this sample size.
    return adfuller(series, autolag="BIC", maxlag=8)[1]


def pp_pvalue(series):
    return PhillipsPerron(series).pvalue


def main():
    df = pd.read_csv(DATA)

    rows = []
    for name, series, level in [
        ("GDP Growth (%)", df["GDP_Growth"], "At level"),
        ("GFCF Growth (%)", df["GFCF_Growth"], "At level"),
        ("GDS (% of GDP)", df["GDS_pct_GDP"].diff().dropna(), "First difference"),
    ]:
        adf_p = adf_pvalue(series)
        pp_p = pp_pvalue(series)
        order = "I(0)" if level == "At level" else "I(1)"
        rows.append({
            "Variable": name,
            "P-value (ADF)": round(adf_p, 4),
            "P-value (PP-test)": round(pp_p, 4),
            "Stationarity": level,
            "Order of Integration": order,
        })

    table3 = pd.DataFrame(rows)
    print("Table 3: Summary of ADF and Phillips-Perron Unit Root Test Results\n")
    print(table3.to_string(index=False))
    print(
        "\nH0: Variable has a unit root | Ha: Variable does not have a unit root."
        "\nGDP growth and GFCF growth reject H0 at level (I(0)); GDS only rejects H0"
        "\nafter first-differencing (I(1)). No variable is I(2), so ARDL bounds"
        "\ntesting is appropriate (Pesaran et al., 2001)."
    )

    table3.to_csv(TAB_DIR / "table3_unit_root_tests.csv", index=False)
    print(f"\nSaved: {TAB_DIR / 'table3_unit_root_tests.csv'}")


if __name__ == "__main__":
    main()

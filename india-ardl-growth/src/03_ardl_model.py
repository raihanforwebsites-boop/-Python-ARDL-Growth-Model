"""
03_ardl_model.py
------------------
Replicates Section 4.4: fits ARDL(1,0,0) of GDP growth on its own first
lag, contemporaneous GFCF growth, and the first difference of GDS
(since GDS is I(1) and GDP/GFCF are I(0)) -- then runs the Pesaran,
Shin & Smith (2001) bounds test for a long-run relationship via the
corresponding unrestricted error-correction model (UECM).
"""

import pandas as pd
from statsmodels.tsa.ardl import ARDL, UECM
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "india_gdp_gfcf_gds_1990_2025.csv"
TAB_DIR = ROOT / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def load():
    df = pd.read_csv(DATA)
    df["D_GDS"] = df["GDS_pct_GDP"].diff()
    df = df.dropna().reset_index(drop=True)
    # Plain RangeIndex (rather than the Year column) avoids a statsmodels
    # ARDL quirk where a non-default/non-contiguous index is treated as
    # requiring out-of-sample exog values during prediction.
    return df


def main():
    df = load()
    endog = df["GDP_Growth"]
    exog = df[["GFCF_Growth", "D_GDS"]]

    # ARDL(1,0,0): one lag on the dependent variable, no lags on either
    # regressor (both entered at level/contemporaneous first difference).
    model = ARDL(endog, lags=1, exog=exog, order={"GFCF_Growth": 0, "D_GDS": 0}, trend="c")
    res = model.fit()

    print("=" * 70)
    print("Table 4/5: ARDL(1,0,0) results")
    print("=" * 70)
    print(res.summary())

    coef_table = pd.DataFrame({
        "Coefficient": res.params.round(3),
        "Std. Error": res.bse.round(3),
        "t-stat": res.tvalues.round(2),
        "p-value": res.pvalues.round(3),
    })
    coef_table["Inference"] = coef_table["p-value"].apply(
        lambda p: "Significant" if p < 0.05 else "Insignificant"
    )
    coef_table.to_csv(TAB_DIR / "table4_ardl_coefficients.csv")

    resid = res.resid
    y_actual = endog.loc[resid.index]
    ss_res = (resid ** 2).sum()
    ss_tot = ((y_actual - y_actual.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    n, k = res.nobs, res.df_model + 1  # + const
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    model_summary = pd.DataFrame({
        "Statistic": ["Observations", "R-squared", "Adj. R-squared", "AIC", "BIC"],
        "Value": [int(res.nobs), round(r2, 3), round(r2_adj, 3),
                  round(res.aic, 3), round(res.bic, 3)],
    })
    model_summary.to_csv(TAB_DIR / "table5_model_summary.csv", index=False)

    # ---- Bounds test / long-run form via UECM ----
    # UECM requires exog lag order >= 1 (it works in first-difference lags
    # internally), unlike the ARDL(1,0,0) above which uses order 0 on both
    # regressors -- this is a UECM API constraint, not a change in model
    # specification for the levels relationship being tested.
    uecm = UECM(endog, lags=1, exog=exog, order={"GFCF_Growth": 1, "D_GDS": 1}, trend="c")
    uecm_res = uecm.fit()
    # Case 3: unrestricted intercept, no trend -- matches trend="c" above.
    bounds = uecm_res.bounds_test(case=3)

    print("\n" + "=" * 70)
    print("Bounds test for level relationship (Pesaran, Shin & Smith, 2001)")
    print("=" * 70)
    print(bounds)

    ec_term = [c for c in uecm_res.params.index if "GDP_Growth" in c and "L1" in c]
    print("\nError-correction coefficient (speed of adjustment):")
    if ec_term:
        print(uecm_res.params[ec_term[0]].round(4))

    with open(TAB_DIR / "bounds_test.txt", "w") as f:
        f.write(str(bounds))
        f.write("\n\nUECM long-run/error-correction form:\n")
        f.write(str(uecm_res.summary()))

    print(f"\nSaved tables to {TAB_DIR}")


if __name__ == "__main__":
    main()

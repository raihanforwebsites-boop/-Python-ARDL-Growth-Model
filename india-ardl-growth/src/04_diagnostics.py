"""
04_diagnostics.py
--------------------
Replicates Table 6 (Summary of Post-Estimation Diagnostic Tests, Section
4.5): Breusch-Pagan-Godfrey heteroskedasticity test, Breusch-Godfrey serial
correlation LM test, Jarque-Bera normality test, Variance Inflation Factors,
and the Ramsey RESET test.

The ARDL(1,0,0) conditional-MLE fit is numerically equivalent to an OLS
regression of GDP growth on its own first lag, GFCF growth, and D(GDS) plus
a constant, so it is re-estimated here via statsmodels.OLS purely so that
the standard statsmodels.stats.diagnostic test suite (built for OLS
results objects) can be applied directly.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    linear_reset,
)
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "india_gdp_gfcf_gds_1990_2025.csv"
TAB_DIR = ROOT / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def build_ols():
    df = pd.read_csv(DATA)
    df["D_GDS"] = df["GDS_pct_GDP"].diff()
    df["GDP_Growth_L1"] = df["GDP_Growth"].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = sm.add_constant(df[["GDP_Growth_L1", "GFCF_Growth", "D_GDS"]])
    y = df["GDP_Growth"]
    res = sm.OLS(y, X).fit()
    return res, X


def main():
    res, X = build_ols()

    rows = []

    # Breusch-Pagan-Godfrey heteroskedasticity test
    bp_stat, bp_p, f_stat, f_p = het_breuschpagan(res.resid, res.model.exog)
    rows.append({
        "Diagnostic Test": "Breusch-Pagan-Godfrey",
        "Null Hypothesis": "Error variances are homoscedastic",
        "Test Statistic": f"F = {f_stat:.2f}",
        "p-value": round(f_p, 2),
        "Decision": "Fail to reject H0" if f_p > 0.05 else "Reject H0",
        "Inference": "No evidence of heteroskedasticity" if f_p > 0.05 else "Evidence of heteroskedasticity",
    })

    # Breusch-Godfrey serial correlation LM test (2 lags, as commonly used)
    bg_lm, bg_lm_p, bg_f, bg_f_p = acorr_breusch_godfrey(res, nlags=2)
    rows.append({
        "Diagnostic Test": "Breusch-Godfrey LM",
        "Null Hypothesis": "No serial correlation exists",
        "Test Statistic": f"F = {bg_f:.2f}",
        "p-value": round(bg_f_p, 2),
        "Decision": "Fail to reject H0" if bg_f_p > 0.05 else "Reject H0",
        "Inference": "Residuals are free from serial correlation" if bg_f_p > 0.05 else "Evidence of serial correlation",
    })

    # Jarque-Bera normality test
    jb_stat, jb_p, skew, kurt = jarque_bera(res.resid)
    rows.append({
        "Diagnostic Test": "Jarque-Bera",
        "Null Hypothesis": "Residuals are normally distributed",
        "Test Statistic": f"JB = {jb_stat:.2f}",
        "p-value": round(jb_p, 2),
        "Decision": "Fail to reject H0" if jb_p > 0.05 else "Reject H0",
        "Inference": "Residuals are normally distributed" if jb_p > 0.05 else "Residuals are not normally distributed",
    })

    # Variance Inflation Factor (regressors only, excluding const)
    exog_no_const = X.drop(columns="const")
    vifs = [variance_inflation_factor(exog_no_const.values, i) for i in range(exog_no_const.shape[1])]
    max_vif = max(vifs)
    rows.append({
        "Diagnostic Test": "VIF",
        "Null Hypothesis": "No multicollinearity exists",
        "Test Statistic": f"max VIF = {max_vif:.2f}",
        "p-value": "-",
        "Decision": "-",
        "Inference": "No multicollinearity (VIF < 5)" if max_vif < 5 else "Multicollinearity present (VIF >= 5)",
    })

    # Ramsey RESET test
    reset_res = linear_reset(res, power=2, use_f=True)
    rows.append({
        "Diagnostic Test": "Ramsey RESET",
        "Null Hypothesis": "Model is correctly specified",
        "Test Statistic": f"F = {reset_res.fvalue:.2f}",
        "p-value": round(reset_res.pvalue, 2),
        "Decision": "Fail to reject H0" if reset_res.pvalue > 0.05 else "Reject H0",
        "Inference": "No evidence of misspecification" if reset_res.pvalue > 0.05 else "Possible model misspecification / omitted variables",
    })

    table6 = pd.DataFrame(rows)
    print("Table 6: Diagnostics Summary\n")
    print(table6.to_string(index=False))

    print("\nVIF by regressor:")
    for name, v in zip(exog_no_const.columns, vifs):
        print(f"  {name}: {v:.2f}")

    table6.to_csv(TAB_DIR / "table6_diagnostics.csv", index=False)
    print(f"\nSaved: {TAB_DIR / 'table6_diagnostics.csv'}")


if __name__ == "__main__":
    main()

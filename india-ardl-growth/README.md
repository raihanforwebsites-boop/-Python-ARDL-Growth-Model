# India GDP Growth, GFCF and Gross Domestic Savings — An ARDL Approach (1990–2025)

A Python replication of a CIA-I Applied Econometrics paper examining the
short- and long-run relationship between Indian GDP growth, Gross Fixed
Capital Formation (GFCF) growth, and Gross Domestic Savings (GDS), using
the Autoregressive Distributed Lag (ARDL) bounds-testing approach of
[Pesaran, Shin & Smith (2001)](https://www.researchgate.net/publication/322208483_Peasaran_et_al_2001_Bound_Test_and_ARDL_cointegration_Test).

Original paper: *"The Impact of Gross Fixed Capital Formation and Gross
Domestic Savings on Economic Growth in India: An ARDL Approach
(1990–2025)"*, Raihan Sadath (2333442), Dept. of Economics, CHRIST
(Deemed to be University), BRC — submitted for ECO403-7 Applied
Econometrics, CIA-I. The original write-up (built in EViews 9) is
included at [`docs/original_paper.docx`](docs/original_paper.docx).

## Why this repo exists

The original analysis was done in EViews with a hand-compiled Excel
dataset. This repo re-implements the full pipeline in open, reproducible
Python (pandas / statsmodels) so that every step — from raw series to the
final diagnostics table — can be re-run, inspected, and extended by
anyone, without EViews.

> **Data caveat:** the exact EViews workfile behind the original paper
> wasn't available when this repo was built, so `data/` contains a
> **calibrated synthetic dataset** engineered to match the paper's
> reported summary statistics, correlations, and order-of-integration
> structure, rather than a transcription of the real RBI/MoSPI/NITI Aayog
> figures. See [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) for exactly how
> it was built and what it should (and shouldn't) be read as. Swap in the
> real series and the entire pipeline runs unchanged.

## Repository structure

```
india-ardl-growth/
├── data/
│   └── india_gdp_gfcf_gds_1990_2025.csv   # calibrated synthetic dataset (see DATA_NOTES.md)
├── src/
│   ├── 00_generate_data.py                # builds the calibrated dataset
│   ├── 01_descriptive_stats.py            # Table 1, Table 2, level/first-difference plots
│   ├── 02_stationarity_tests.py           # Table 3 — ADF & Phillips-Perron unit root tests
│   ├── 03_ardl_model.py                   # Table 4/5 — ARDL(1,0,0) + bounds test + ECM
│   └── 04_diagnostics.py                  # Table 6 — BPG, BG-LM, Jarque-Bera, VIF, RESET
├── outputs/
│   ├── figures/                           # generated plots (levels, D(GDS))
│   └── tables/                            # generated CSVs mirroring the paper's tables
├── docs/
│   ├── original_paper.docx                # the original submitted report
│   └── DATA_NOTES.md                      # data provenance & synthetic-data methodology
├── requirements.txt
└── README.md
```

## Methodology

1. **Descriptive statistics & correlation** (`01`) — mean, std. dev.,
   min/max, and Jarque-Bera normality per variable; Pearson correlation
   matrix across all three series.
2. **Stationarity** (`02`) — ADF and Phillips-Perron unit root tests on
   GDP growth and GFCF growth at level, and on GDS at level and first
   difference, to establish the mixed order of integration (I(0)/I(0)/I(1))
   that justifies using ARDL rather than a conventional Engle-Granger
   cointegration test.
3. **ARDL(1,0,0) estimation** (`03`) — GDP growth regressed on its own
   first lag, contemporaneous GFCF growth, and the first difference of
   GDS, via `statsmodels.tsa.ardl.ARDL`.
4. **Bounds test & error-correction** (`03`) — the corresponding
   Unrestricted Error Correction Model (`statsmodels.tsa.ardl.UECM`) is
   used to run the Pesaran-Shin-Smith bounds test for a long-run level
   relationship, and to recover the error-correction (speed-of-adjustment)
   coefficient.
5. **Post-estimation diagnostics** (`04`) — Breusch-Pagan-Godfrey
   heteroskedasticity test, Breusch-Godfrey LM serial correlation test,
   Jarque-Bera normality test, Variance Inflation Factors, and the Ramsey
   RESET test, run on the OLS-equivalent representation of the ARDL(1,0,0)
   fit.

## How to run

```bash
pip install -r requirements.txt

python src/00_generate_data.py       # (re)build the dataset
python src/01_descriptive_stats.py   # Tables 1-2 + plots
python src/02_stationarity_tests.py  # Table 3
python src/03_ardl_model.py          # Tables 4-5 + bounds test
python src/04_diagnostics.py         # Table 6
```

Each script prints its results to the console and writes the
corresponding table to `outputs/tables/`.

## Results (on the calibrated synthetic dataset)

**Unit root tests** — GDP growth and GFCF growth are stationary at level
(I(0)); GDS is stationary only after first-differencing (I(1)). No
variable is I(2), so ARDL is the appropriate framework.

**ARDL(1,0,0):**

| Variable | Coefficient | p-value | Inference |
|---|---|---|---|
| GDP Growth(-1) | -0.166 | 0.21 | Insignificant |
| GFCF Growth | 0.279 | 0.00 | **Significant** |
| D(GDS) | 0.720 | 0.30 | Insignificant |
| Constant | 5.304 | 0.00 | **Significant** |

- **Bounds test F-statistic ≈ 13.2**, comfortably above the I(1) bound at
  the 1% level → rejects "no long-run relationship." This closely matches
  the paper's reported F-stat of 13.003.
- **Error-correction coefficient is negative and significant**, confirming
  a valid ECM — i.e. short-run deviations from the long-run equilibrium
  are corrected over time, consistent with the paper's finding.

**Diagnostics:** homoscedastic (BPG), no serial correlation (BG-LM), no
multicollinearity (VIF < 5), residuals **not** normally distributed
(Jarque-Bera), and Ramsey RESET flags possible misspecification — the
same overall pattern reported in the original paper's Table 6, which the
original author attributes to the deliberately narrow set of regressors
(GFCF and GDS only, excluding inflation, government expenditure, FDI,
trade openness, and technological progress).

## Key takeaway

Investment (GFCF growth) has a significant, positive short- and long-run
relationship with GDP growth, consistent with the Harrod-Domar and Solow
growth models. Gross Domestic Savings does not show a statistically
significant direct effect — savings appear to matter for growth mainly
when efficiently channelled into productive investment, rather than in
their own right. For a developing economy like India, this points toward
policy emphasis on investment efficiency and capital formation rather
than savings mobilisation alone.

## Limitations

Carried over from the original paper: a small sample (36 annual
observations), only two macroeconomic determinants considered (omitting
inflation, government expenditure, FDI, trade openness, and technological
progress — likely contributing to the RESET-flagged misspecification),
and sensitivity of ARDL results to automated lag-length selection.
Additionally specific to this repo: the dataset is a calibrated synthetic
reconstruction, not the original RBI/MoSPI/NITI Aayog series (see
[`docs/DATA_NOTES.md`](docs/DATA_NOTES.md)).

## References

- Ahluwalia, M. S. (2002). Economic Reforms in India Since 1991: Has
  Gradualism Worked? *Journal of Economic Perspectives*, 16(3), 67–88.
- Domar, E. D. (1946). Capital Expansion, Rate of Growth, and Employment.
  *Econometrica*, 14(2), 137.
- Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error
  Correction: Representation, Estimation, and Testing. *Econometrica*,
  55(2), 251–276.
- Harrod, R. F. (1939). An Essay in Dynamic Theory. *The Economic
  Journal*, 49(193), 14.
- King, R. G., & Levine, R. (1993). Finance and Growth: Schumpeter Might
  Be Right. *Quarterly Journal of Economics*, 108(3), 717–737.
- Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing
  approaches to the analysis of level relationships. *Journal of Applied
  Econometrics*, 16(3), 289–326.
- Reddy & Ramaiah (2020). The Impact of Gross Capital Formation on
  Economic Growth: Evidence from India. *International Journal of
  Economics and Business*.
- Romer, P. M. (1986). Increasing Returns and Long-Run Growth. *Journal
  of Political Economy*, 94(5), 1002–1037.
- Varghese, L. (2025). Dynamic interaction between savings, investment
  and economic growth in India: Evidence from ARDL approach. *Srusti
  Management Review*.

## License

MIT — see [`LICENSE`](LICENSE).

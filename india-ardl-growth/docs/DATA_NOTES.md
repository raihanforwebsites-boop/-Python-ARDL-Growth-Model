# Data Notes

## Where the original numbers came from

The underlying CIA-I paper ("*The Impact of Gross Fixed Capital Formation
and Gross Domestic Savings on Economic Growth in India: An ARDL Approach,
1990–2025*") was built on annual data compiled from:

- **RBI** — Handbook of Statistics on the Indian Economy
- **MoSPI** — National Account Statistics
- **NITI Aayog** — National Data and Analytics Platform

These were compiled in Excel and analysed in **EViews 9**. That raw EViews
workfile / underlying spreadsheet is not part of this repository — only the
written report (with summary tables and EViews screenshots) was available
when this repo was built.

## Why `data/india_gdp_gfcf_gds_1990_2025.csv` is synthetic

To make the whole pipeline (stationarity tests → ARDL → bounds test →
diagnostics) runnable and reproducible end-to-end without the original
workfile, `src/00_generate_data.py` **generates a calibrated synthetic
series** rather than transcribing real historical GDP/GFCF/GDS figures by
hand (which would risk silently introducing wrong numbers presented as
real data).

The generator searches over random seeds for a series whose:

- mean, standard deviation, min, and max for each variable, and
- pairwise Pearson correlations

closely match Table 1 and Table 2 of the original paper, while also
preserving the mixed order-of-integration structure the paper reports:
GDP growth and GFCF growth are constructed to be stationary at level
(I(0)), and GDS is constructed as a genuine random walk with drift so it
is non-stationary at level but stationary after first-differencing
(I(1)) — exactly the ARDL-justifying pattern in Section 3 of the paper.

**This is a stand-in dataset for reproducibility, not a transcription of
real Indian macroeconomic history.** In particular:

- Individual year-by-year values (e.g. the 2020 figure) should **not** be
  read as actual historical GDP/GFCF/GDS data — they were not built to
  reflect real single-year events (e.g. COVID-19).
- The qualitative conclusions the pipeline reproduces (GFCF growth has a
  significant positive short- and long-run relationship with GDP growth;
  GDS does not; a cointegrating long-run relationship exists; residuals
  are homoscedastic and free of serial correlation but not normally
  distributed; RESET flags possible misspecification) match the paper
  closely and are the point of this repository.
- If you have access to the real RBI/MoSPI/NITI Aayog series, drop a CSV
  with the same column names (`Year, GDP_Growth, GFCF_Growth,
  GDS_pct_GDP`) into `data/` and re-point the scripts at it — everything
  downstream (stationarity tests, ARDL, bounds test, diagnostics) will run
  unchanged on real data.

## Column definitions

| Column | Description |
|---|---|
| `Year` | Calendar year, 1990–2025 (36 annual observations) |
| `GDP_Growth` | Real GDP growth rate, % |
| `GFCF_Growth` | Gross Fixed Capital Formation growth rate, % |
| `GDS_pct_GDP` | Gross Domestic Savings, % of GDP |

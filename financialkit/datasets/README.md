# Bundled datasets

Small CSVs of real market data, read by `financialkit.datasets`. They are
written by `scripts/fetch_datasets.py`, which the author runs by hand,
offline from CI, and commits. Nothing in the package, the tests or the
gallery downloads anything. Each CSV's first line names its source.

Retrieved 2026-10-27.

| File | Content | Source | Terms |
|---|---|---|---|
| `shiller_monthly.csv` | S&P composite price, dividends, earnings, CPI and long rate, monthly from 1871-01 | Robert J. Shiller, *U.S. Stock Markets 1871-Present and CAPE Ratio*, `ie_data.xls` from <https://shillerdata.com/> | Posted publicly for research and teaching; cite Shiller (1981) and *Irrational Exuberance* |
| `ff_market_daily.csv` | Daily market excess return (`Mkt-RF`) and T-bill return, percent, from 1926-07-01 | Kenneth R. French, Data Library, *Fama/French 3 Factors [Daily]* | No licence stated; see below |
| `ff_factors_monthly.csv` | Monthly `Mkt-RF`, SMB, HML, RMW, CMA, momentum and RF, percent, from 1926-07 | French, *Fama/French 3 Factors*, *5 Factors (2x3)* and *Momentum Factor* | No licence stated; see below |
| `ff25_size_value_monthly.csv` | 25 value-weighted size by book-to-market portfolios, percent, monthly from 1963-07 | French, *25 Portfolios Formed on Size and Book-to-Market (5 x 5)* | No licence stated; see below |
| `treasury_yields_monthly.csv` | Constant-maturity Treasury yields, 3 months to 10 years, percent, last complete business day of each month from 1981-09 | Federal Reserve H.15 via FRED, series `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2`, `DGS3`, `DGS5`, `DGS7`, `DGS10` | U.S. government data, public domain |

Kenneth French's library is at
<https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>.
It is built from CRSP and Compustat data and states no licence. Only the
trimmed series the examples need are included, with credit to Eugene F.
Fama, Kenneth R. French and CRSP. Remove them if French asks.

To refresh: run `python scripts/fetch_datasets.py`, update the date above,
re-run the gallery (the examples assert on the numbers) and commit the
data in a commit of its own.

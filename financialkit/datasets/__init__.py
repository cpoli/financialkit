"""Small real datasets bundled with financialkit, loaded as numpy arrays.

The empirical lessons (fat tails, GARCH, excess volatility, the equity
premium, Fama-French, yield-curve PCA) are circular on data drawn from
the model being shown, so they run on these instead. Every loader reads a
CSV shipped in the package (nothing touches the network) and returns a
frozen dataclass of arrays, with dates as :class:`numpy.datetime64` and
returns and yields as decimals (0.01 is 1%), not percent.

* :func:`load_shiller` -- Robert Shiller's monthly S&P composite price,
  dividends, earnings, CPI and long rate since 1871.
* :func:`load_market_daily` -- the CRSP value-weighted market's daily
  excess return and the T-bill rate since 1926, from Kenneth French's
  data library.
* :func:`load_fama_french_factors` -- French's monthly factors: market,
  SMB, HML (since 1926), RMW and CMA (since 1963), momentum (since 1927)
  and the risk-free rate.
* :func:`load_fama_french_25` -- French's 25 value-weighted size by
  book-to-market portfolios, monthly since July 1963.
* :func:`load_treasury_yields` -- month-end constant-maturity Treasury
  yields from 3 months to 10 years (FRED, H.15), since September 1981.

``scripts/fetch_datasets.py`` writes the CSVs. The author runs it by hand
and commits the output; ``financialkit/datasets/README.md`` gives each
file's source, terms and retrieval date. Cite the original sources, not
financialkit, when you use the numbers.
"""

from financialkit import __version__
from financialkit.datasets.loaders import (
    FactorData,
    MarketReturns,
    PortfolioReturns,
    ShillerData,
    YieldCurveData,
    load_fama_french_25,
    load_fama_french_factors,
    load_market_daily,
    load_shiller,
    load_treasury_yields,
)

__all__ = [
    "__version__",
    "ShillerData",
    "MarketReturns",
    "FactorData",
    "PortfolioReturns",
    "YieldCurveData",
    "load_shiller",
    "load_market_daily",
    "load_fama_french_factors",
    "load_fama_french_25",
    "load_treasury_yields",
]

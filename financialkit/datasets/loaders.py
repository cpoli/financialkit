"""Loaders for the CSVs in ``financialkit/datasets/data/``.

Each CSV starts with one ``#`` line naming its source, then a header row.
Dates are stored as ``yyyymm`` or ``yyyymmdd`` integers; French's returns
and FRED's yields are stored in percent, as published, and converted to
decimals here. Missing values are empty fields, read as ``nan``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np

__all__ = [
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


def _read(name: str) -> np.ndarray:
    """Read a bundled CSV as a structured array with one field per column."""
    with files("financialkit.datasets").joinpath("data", name).open() as fh:
        fh.readline()  # the "# source" line
        return np.genfromtxt(fh, delimiter=",", names=True, dtype=np.float64)


def _columns(a: np.ndarray) -> tuple[str, ...]:
    """Field names after the leading date column."""
    assert a.dtype.names is not None
    return a.dtype.names[1:]


def _months(yyyymm: np.ndarray) -> np.ndarray:
    y = yyyymm.astype(np.int64)
    return ((y // 100 - 1970) * 12 + y % 100 - 1).astype("datetime64[M]")


def _days(yyyymmdd: np.ndarray) -> np.ndarray:
    d = yyyymmdd.astype(np.int64)
    return _months(d // 100).astype("datetime64[D]") + (d % 100 - 1)


@dataclass(frozen=True)
class ShillerData:
    """Shiller's monthly U.S. stock-market series (S&P composite, nominal).

    Prices are monthly averages of daily closes; dividends and earnings are
    at annual rates, interpolated from quarterly figures; the most recent
    months have a price but no dividend or earnings yet (``nan``).
    """

    dates: np.ndarray
    """ndarray of datetime64[M], shape (T,): Month, January 1871 onwards."""

    price: np.ndarray
    """ndarray, shape (T,): S&P composite index level."""

    dividend: np.ndarray
    """ndarray, shape (T,): Dividends per index unit, at an annual rate."""

    earnings: np.ndarray
    """ndarray, shape (T,): Earnings per index unit, at an annual rate."""

    cpi: np.ndarray
    """ndarray, shape (T,): Consumer price index."""

    long_rate: np.ndarray
    """ndarray, shape (T,): Ten-year Treasury yield (GS10, earlier a long government bond rate), as a decimal."""

    @property
    def real_price(self) -> np.ndarray:
        """ndarray, shape (T,): Price in dollars of the last month's CPI."""
        return self.price * self.cpi[-1] / self.cpi

    @property
    def real_dividend(self) -> np.ndarray:
        """ndarray, shape (T,): Dividend in dollars of the last month's CPI."""
        return self.dividend * self.cpi[-1] / self.cpi

    def annual(self, month: int = 1) -> ShillerData:
        """One observation a year, taken in the given calendar month (January by default)."""
        keep = (self.dates.astype(np.int64) % 12) == month - 1
        return ShillerData(*(getattr(self, f)[keep] for f in ("dates", "price", "dividend", "earnings", "cpi", "long_rate")))


@dataclass(frozen=True)
class MarketReturns:
    """Daily returns of the CRSP value-weighted market portfolio."""

    dates: np.ndarray
    """ndarray of datetime64[D], shape (T,): Trading day, 1 July 1926 onwards."""

    excess: np.ndarray
    """ndarray, shape (T,): Market return minus the T-bill return, ``Mkt-RF``."""

    risk_free: np.ndarray
    """ndarray, shape (T,): Daily T-bill return (published to 0.01%)."""

    @property
    def returns(self) -> np.ndarray:
        """ndarray, shape (T,): Total market return, ``excess + risk_free``."""
        return self.excess + self.risk_free


@dataclass(frozen=True)
class FactorData:
    """French's monthly factor returns; a factor is ``nan`` before it starts."""

    dates: np.ndarray
    """ndarray of datetime64[M], shape (T,): Month, July 1926 onwards."""

    mkt_rf: np.ndarray
    """ndarray, shape (T,): Market excess return."""

    smb: np.ndarray
    """ndarray, shape (T,): Small minus big (from the three-factor file)."""

    hml: np.ndarray
    """ndarray, shape (T,): High minus low book-to-market."""

    rmw: np.ndarray
    """ndarray, shape (T,): Robust minus weak profitability, from July 1963."""

    cma: np.ndarray
    """ndarray, shape (T,): Conservative minus aggressive investment, from July 1963."""

    mom: np.ndarray
    """ndarray, shape (T,): Momentum (winners minus losers), from January 1927."""

    risk_free: np.ndarray
    """ndarray, shape (T,): One-month T-bill return."""

    def window(self, start: str, end: str | None = None) -> FactorData:
        """The months from ``start`` to ``end`` inclusive, given as ``"YYYY-MM"``."""
        keep = _in_window(self.dates, start, end)
        return FactorData(*(getattr(self, f)[keep] for f in ("dates", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "risk_free")))


@dataclass(frozen=True)
class PortfolioReturns:
    """French's 25 value-weighted portfolios sorted on size and book-to-market."""

    dates: np.ndarray
    """ndarray of datetime64[M], shape (T,): Month, July 1963 onwards."""

    returns: np.ndarray
    """ndarray, shape (T, 25): Monthly returns; column ``5 * (i - 1) + (j - 1)`` is size quintile ``i`` and book-to-market quintile ``j`` (1 lowest)."""

    names: tuple[str, ...]
    """tuple of str: Column names ``"me1_bm1"`` to ``"me5_bm5"``."""

    def window(self, start: str, end: str | None = None) -> PortfolioReturns:
        """The months from ``start`` to ``end`` inclusive, given as ``"YYYY-MM"``."""
        keep = _in_window(self.dates, start, end)
        return PortfolioReturns(self.dates[keep], self.returns[keep], self.names)


@dataclass(frozen=True)
class YieldCurveData:
    """Month-end constant-maturity Treasury yields."""

    dates: np.ndarray
    """ndarray of datetime64[M], shape (T,): Month, September 1981 onwards."""

    tenors: np.ndarray
    """ndarray, shape (8,): Maturities in years, 0.25 to 10."""

    yields: np.ndarray
    """ndarray, shape (T, 8): Bond-equivalent yields as decimals, on the last business day of each month."""


def _in_window(dates: np.ndarray, start: str, end: str | None) -> np.ndarray:
    keep = dates >= np.datetime64(start, "M")
    if end is not None:
        keep &= dates <= np.datetime64(end, "M")
    return keep


def load_shiller() -> ShillerData:
    """Shiller's monthly S&P composite data, January 1871 onwards.

    Source: Robert J. Shiller, *U.S. Stock Markets 1871-Present and CAPE
    Ratio* (``ie_data.xls``, shillerdata.com), as used in Shiller (1981),
    "Do Stock Prices Move Too Much to be Justified by Subsequent Changes in
    Dividends?", *American Economic Review* 71(3), 421-436, and in
    *Irrational Exuberance* (2000).

    Returns
    -------
    ShillerData

    Examples
    --------
    >>> d = load_shiller()
    >>> str(d.dates[0]), float(d.price[0])
    ('1871-01', 4.44)
    """
    a = _read("shiller_monthly.csv")
    return ShillerData(_months(a["yyyymm"]), a["price"], a["dividend"], a["earnings"], a["cpi"], a["long_rate"] / 100)


def load_market_daily() -> MarketReturns:
    """Daily CRSP value-weighted market excess returns and T-bill returns, July 1926 onwards.

    Source: Kenneth R. French, Data Library, *Fama/French 3 Factors
    [Daily]*, built from CRSP; see Fama and French (1993), "Common Risk
    Factors in the Returns on Stocks and Bonds", *Journal of Financial
    Economics* 33(1), 3-56.

    Returns
    -------
    MarketReturns

    Examples
    --------
    >>> m = load_market_daily()
    >>> str(m.dates[0]), round(float(m.excess[0]), 6)
    ('1926-07-01', 0.0009)
    """
    a = _read("ff_market_daily.csv")
    return MarketReturns(_days(a["yyyymmdd"]), a["mkt_rf"] / 100, a["rf"] / 100)


def load_fama_french_factors() -> FactorData:
    """French's monthly market, size, value, profitability, investment and momentum factors.

    Source: Kenneth R. French, Data Library (*Fama/French 3 Factors*,
    *Fama/French 5 Factors (2x3)* and *Momentum Factor*), built from CRSP and
    Compustat; see Fama and French (1993), Carhart (1997), "On Persistence
    in Mutual Fund Performance", *Journal of Finance* 52(1), 57-82, and
    Fama and French (2015), "A Five-Factor Asset Pricing Model", *Journal
    of Financial Economics* 116(1), 1-22.

    Returns
    -------
    FactorData

    Examples
    --------
    >>> f = load_fama_french_factors()
    >>> str(f.dates[0]), bool(np.isnan(f.rmw[0])), str(f.dates[~np.isnan(f.rmw)][0])
    ('1926-07', True, '1963-07')
    """
    a = _read("ff_factors_monthly.csv")
    cols = (a[c] / 100 for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"))
    return FactorData(_months(a["yyyymm"]), *cols)


def load_fama_french_25() -> PortfolioReturns:
    """French's 25 value-weighted size by book-to-market portfolios, monthly from July 1963.

    Source: Kenneth R. French, Data Library, *25 Portfolios Formed on Size
    and Book-to-Market (5 x 5)*, the test assets of Fama and French (1993).

    Returns
    -------
    PortfolioReturns

    Examples
    --------
    >>> p = load_fama_french_25()
    >>> p.returns.shape[1], p.names[0], p.names[-1]
    (25, 'me1_bm1', 'me5_bm5')
    """
    a = _read("ff25_size_value_monthly.csv")
    names = _columns(a)
    return PortfolioReturns(_months(a["yyyymm"]), np.column_stack([a[n] for n in names]) / 100, names)


def load_treasury_yields() -> YieldCurveData:
    """Month-end constant-maturity Treasury yields, 3 months to 10 years, from September 1981.

    Source: Board of Governors of the Federal Reserve System, H.15
    Selected Interest Rates, series ``DGS3MO`` to ``DGS10``, via FRED
    (Federal Reserve Bank of St. Louis). The value for each month is the
    last business day on which every maturity was quoted.

    Returns
    -------
    YieldCurveData

    Examples
    --------
    >>> c = load_treasury_yields()
    >>> c.tenors.tolist()
    [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    """
    a = _read("treasury_yields_monthly.csv")
    names = _columns(a)
    tenors = np.array([float(n[1:-2]) / 12 if n.endswith("mo") else float(n[1:]) for n in names])
    return YieldCurveData(_months(a["yyyymm"]), tenors, np.column_stack([a[n] for n in names]) / 100)

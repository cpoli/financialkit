"""Equity valuation: Williams's dividend discount model, Gordon growth,
Modigliani-Miller and Shiller's variance bound.

Williams (1938) defined a share's investment value as the present value
of all its future dividends. Gordon (1959) summed the series when
dividends grow at a constant rate, and tied that rate to the retained
share of earnings. Modigliani and Miller (1958) showed that in a
frictionless market the value of a firm does not depend on how it is
financed, so leverage only shifts risk and return between debt and
equity. Shiller (1981) turned the dividend discount model into a test:
if prices are rational forecasts of discounted dividends, they cannot
be more volatile than the discounted dividends actually paid.

Rates in this module are effective per-period rates, as in the
original papers, rather than continuously compounded ones.

References: J. B. Williams, *The Theory of Investment Value*, Harvard
University Press (1938); F. Modigliani and M. H. Miller, "The Cost of
Capital, Corporation Finance and the Theory of Investment," *Am. Econ.
Rev.* 48 (1958) 261-297, and "Corporate Income Taxes and the Cost of
Capital: A Correction," *Am. Econ. Rev.* 53 (1963) 433-443;
M. J. Gordon, "Dividends, Earnings, and Stock Prices," *Rev. Econ.
Statist.* 41 (1959) 99-105; R. J. Shiller, "Do Stock Prices Move Too
Much to be Justified by Subsequent Changes in Dividends?" *Am. Econ.
Rev.* 71 (1981) 421-436.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from financialkit.asset_pricing.core.base import ExcessVolatilityResult
from financialkit.time_value import perpetuity_present_value

__all__ = [
    "dividend_discount_value",
    "gordon_growth_price",
    "implied_cost_of_equity",
    "sustainable_growth",
    "levered_cost_of_equity",
    "levered_firm_value",
    "wacc",
    "ex_post_rational_price",
    "ar1_dividend_price",
    "variance_bound",
]


def _out(x: np.ndarray) -> np.ndarray | float:
    return float(x) if x.ndim == 0 else x


def dividend_discount_value(dividends: ArrayLike, rate: float, terminal_growth: Optional[float] = None) -> float:
    """Williams's investment value: the present value of a share's dividends.

    Parameters
    ----------
    dividends : array_like, shape (n,)
        Dividends forecast for periods ``1 .. n``.
    rate : float
        Required return per period.
    terminal_growth : float, optional
        If given, dividends after period ``n`` grow at this rate forever,
        and their Gordon value at ``n`` is discounted back and added.
        If omitted, the share is worth only the ``n`` dividends shown.

    Returns
    -------
    float

    Examples
    --------
    Three dividends of 1, 1.1 and 1.2, then 3% growth, at a 10% required return:

    >>> round(dividend_discount_value([1.0, 1.1, 1.2], 0.10, terminal_growth=0.03), 4)
    15.9858
    """
    D = np.asarray(dividends, dtype=np.float64)
    n = D.size
    disc = (1.0 + rate) ** -np.arange(1, n + 1)
    value = float(D @ disc)
    if terminal_growth is not None:
        terminal = perpetuity_present_value(D[-1] * (1.0 + terminal_growth), rate, terminal_growth)
        value += float(terminal) * float(disc[-1])
    return value


def gordon_growth_price(next_dividend: ArrayLike, rate: ArrayLike, growth: ArrayLike) -> np.ndarray | float:
    """Gordon's price of a share whose dividends grow at a constant rate, :math:`D_1 / (r - g)`.

    Parameters
    ----------
    next_dividend : array_like
        Dividend paid one period from now, :math:`D_1`.
    rate : array_like
        Required return :math:`r`, greater than ``growth``.
    growth : array_like
        Dividend growth rate :math:`g`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> gordon_growth_price(2.0, 0.09, 0.04)
    40.0
    """
    return perpetuity_present_value(next_dividend, rate, growth)


def implied_cost_of_equity(price: ArrayLike, next_dividend: ArrayLike, growth: ArrayLike) -> np.ndarray | float:
    """The required return that the Gordon model reads off a price, :math:`D_1 / P + g`.

    Parameters
    ----------
    price, next_dividend, growth : array_like

    Returns
    -------
    float or ndarray
        Dividend yield plus growth.

    Examples
    --------
    >>> round(implied_cost_of_equity(40.0, 2.0, 0.04), 12)
    0.09
    """
    P = np.asarray(price, dtype=np.float64)
    return _out(np.asarray(next_dividend, dtype=np.float64) / P + np.asarray(growth, dtype=np.float64))


def sustainable_growth(return_on_equity: ArrayLike, payout_ratio: ArrayLike) -> np.ndarray | float:
    """Gordon's growth rate of a firm that reinvests its retained earnings, :math:`g = b \\cdot ROE`.

    Parameters
    ----------
    return_on_equity : array_like
        Return earned on reinvested equity.
    payout_ratio : array_like
        Share of earnings paid as dividends, so ``b = 1 - payout_ratio`` is retained.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(sustainable_growth(0.15, 0.6), 12)
    0.06
    """
    b = 1.0 - np.asarray(payout_ratio, dtype=np.float64)
    return _out(b * np.asarray(return_on_equity, dtype=np.float64))


def levered_cost_of_equity(asset_return: ArrayLike, debt_return: ArrayLike, debt_to_equity: ArrayLike, tax_rate: float = 0.0) -> np.ndarray | float:
    """Modigliani-Miller Proposition II: the expected return on levered equity.

    :math:`r_E = r_A + (r_A - r_D)(1 - t) D / E`.

    Parameters
    ----------
    asset_return : array_like
        Required return on the firm's assets (the unlevered cost of capital).
    debt_return : array_like
        Required return on its debt.
    debt_to_equity : array_like
        Market-value leverage :math:`D / E`.
    tax_rate : float
        Corporate tax rate; ``0`` gives the 1958 proposition, a positive
        value the 1963 correction with perpetual debt.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(levered_cost_of_equity(0.10, 0.05, 1.0), 12)
    0.15
    """
    rA = np.asarray(asset_return, dtype=np.float64)
    rD = np.asarray(debt_return, dtype=np.float64)
    return _out(rA + (rA - rD) * (1.0 - tax_rate) * np.asarray(debt_to_equity, dtype=np.float64))


def levered_firm_value(unlevered_value: ArrayLike, debt: ArrayLike, tax_rate: float = 0.0) -> np.ndarray | float:
    """Modigliani-Miller Proposition I: :math:`V_L = V_U + t D`.

    Without taxes the value of the firm does not depend on its debt; with
    a corporate tax and permanent debt the tax shield adds :math:`t D`.

    Parameters
    ----------
    unlevered_value : array_like
    debt : array_like
    tax_rate : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> levered_firm_value(100.0, 40.0), levered_firm_value(100.0, 40.0, tax_rate=0.25)
    (100.0, 110.0)
    """
    return _out(np.asarray(unlevered_value, dtype=np.float64) + tax_rate * np.asarray(debt, dtype=np.float64))


def wacc(equity_return: ArrayLike, debt_return: ArrayLike, debt_to_value: ArrayLike, tax_rate: float = 0.0) -> np.ndarray | float:
    """Weighted average cost of capital, :math:`(1 - L) r_E + L r_D (1 - t)` with :math:`L = D / V`.

    Parameters
    ----------
    equity_return, debt_return : array_like
    debt_to_value : array_like
        :math:`D / (D + E)`.
    tax_rate : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    With the equity return from Proposition II, the WACC is the asset return:

    >>> round(wacc(0.15, 0.05, 0.5), 12)
    0.1
    """
    L = np.asarray(debt_to_value, dtype=np.float64)
    rE = np.asarray(equity_return, dtype=np.float64)
    rD = np.asarray(debt_return, dtype=np.float64)
    return _out((1.0 - L) * rE + L * rD * (1.0 - tax_rate))


def ex_post_rational_price(dividends: ArrayLike, rate: float, terminal_price: float) -> np.ndarray:
    """Shiller's :math:`p^*_t`: the value of the dividends that were actually paid.

    :math:`p^*_t = (p^*_{t+1} + D_{t+1}) / (1 + r)`, run backwards from
    :math:`p^*_T` = ``terminal_price``.

    Parameters
    ----------
    dividends : array_like, shape (T,)
        Dividend paid at the end of each period ``0 .. T-1``; ``dividends[t + 1]``
        is received by the holder at date ``t``.
    rate : float
        Constant discount rate per period.
    terminal_price : float
        Value placed on the share at the last date (Shiller used the
        sample-average price).

    Returns
    -------
    ndarray, shape (T,)

    Examples
    --------
    >>> ex_post_rational_price([1.0, 1.0, 1.0], 0.0, 5.0).tolist()
    [7.0, 6.0, 5.0]
    """
    D = np.asarray(dividends, dtype=np.float64)
    p = np.empty_like(D)
    p[-1] = terminal_price
    for t in range(D.size - 2, -1, -1):
        p[t] = (p[t + 1] + D[t + 1]) / (1.0 + rate)
    return p


def ar1_dividend_price(dividends: ArrayLike, rate: float, rho: float, mean: float) -> np.ndarray | float:
    """The rational price when dividends follow an AR(1) around a constant mean.

    For :math:`D_{t+1} - \\bar D = \\rho (D_t - \\bar D) + \\varepsilon_{t+1}`,
    the expected present value of future dividends is
    :math:`\\bar D / r + \\rho (D_t - \\bar D) / (1 + r - \\rho)`.

    Parameters
    ----------
    dividends : array_like
        Current dividend :math:`D_t`.
    rate : float
        Discount rate per period.
    rho : float
        Autocorrelation of dividends, in ``[0, 1)``.
    mean : float
        Long-run dividend :math:`\\bar D`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(ar1_dividend_price(1.0, 0.05, 0.9, 1.0), 12)
    20.0
    """
    D = np.asarray(dividends, dtype=np.float64)
    return _out(mean / rate + rho * (D - mean) / (1.0 + rate - rho))


def variance_bound(price: ArrayLike, dividends: ArrayLike, rate: float, terminal_price: Optional[float] = None) -> ExcessVolatilityResult:
    """Compare the volatility of a price with that of its ex-post rational price.

    If :math:`p_t = E_t[p^*_t]`, then :math:`p^*_t = p_t + u_t` with a
    forecast error :math:`u_t` uncorrelated with :math:`p_t`, so
    :math:`\\sigma(p) \\le \\sigma(p^*)`.

    Parameters
    ----------
    price : array_like, shape (T,)
        Actual (detrended) price.
    dividends : array_like, shape (T,)
        Matching (detrended) dividends.
    rate : float
        Discount rate per period.
    terminal_price : float, optional
        Value of :math:`p^*` at the last date; the sample-average price by default.

    Returns
    -------
    ExcessVolatilityResult
    """
    P = np.asarray(price, dtype=np.float64)
    term = float(P.mean()) if terminal_price is None else float(terminal_price)
    p_star = ex_post_rational_price(dividends, rate, term)
    return ExcessVolatilityResult(price=P, rational_price=p_star, std_price=float(P.std()), std_rational=float(p_star.std()))

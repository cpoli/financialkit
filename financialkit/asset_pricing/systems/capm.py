"""The capital asset pricing model: Sharpe's security market line, Jensen's
alpha, Black's zero-beta CAPM and the Gibbons-Ross-Shanken test.

If every investor holds a mean-variance efficient portfolio and can
borrow and lend at the riskless rate, all of them hold the same risky
portfolio, the tangency portfolio, which must therefore be the market.
Its first-order condition is the security market line
:math:`E[r_i] = r_f + \\beta_i (E[r_m] - r_f)`. Jensen (1968) used the
intercept of a fund's regression on the market to measure skill. Black
(1972) dropped riskless borrowing: the line survives with a zero-beta
portfolio's return in place of :math:`r_f`. Gibbons, Ross and Shanken
(1989) gave the exact finite-sample F-test that all intercepts are zero.

The tangency portfolio is :func:`financialkit.risk.safety_first_portfolio`:
Roy's portfolio for a disaster level equal to the riskless rate.

References: W. F. Sharpe, "Capital Asset Prices: A Theory of Market
Equilibrium under Conditions of Risk," *J. Finance* 19 (1964) 425-442;
M. C. Jensen, "The Performance of Mutual Funds in the Period 1945-1964,"
*J. Finance* 23 (1968) 389-416; F. Black, "Capital Market Equilibrium
with Restricted Borrowing," *J. Business* 45 (1972) 444-455;
M. R. Gibbons, S. A. Ross and J. Shanken, "A Test of the Efficiency of
a Given Portfolio," *Econometrica* 57 (1989) 1121-1152.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.asset_pricing.core.base import CAPMEquilibrium, FactorRegressionResult, GRSResult, ZeroBetaResult
from financialkit.asset_pricing.systems.factors import time_series_regression
from financialkit.asset_pricing.utils.sorts import as_panel
from financialkit.risk import safety_first_portfolio

__all__ = [
    "capm_expected_return",
    "estimate_beta",
    "capm_equilibrium",
    "jensen_alpha",
    "zero_beta_capm",
    "grs_test",
]


def capm_expected_return(beta: ArrayLike, risk_free: ArrayLike, market_return: ArrayLike) -> np.ndarray | float:
    """The security market line, :math:`r_f + \\beta (E[r_m] - r_f)`.

    Parameters
    ----------
    beta, risk_free, market_return : array_like

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(capm_expected_return(1.5, 0.03, 0.08), 12)
    0.105
    """
    rf = np.asarray(risk_free, dtype=np.float64)
    out = rf + np.asarray(beta, dtype=np.float64) * (np.asarray(market_return, dtype=np.float64) - rf)
    return float(out) if out.ndim == 0 else out


def estimate_beta(returns: ArrayLike, market_returns: ArrayLike) -> np.ndarray | float:
    """Sample beta :math:`\\mathrm{Cov}(r_i, r_m) / \\mathrm{Var}(r_m)`.

    Parameters
    ----------
    returns : array_like, shape (T,) or (T, n)
    market_returns : array_like, shape (T,)

    Returns
    -------
    float or ndarray, shape (n,)

    Examples
    --------
    >>> estimate_beta([0.02, -0.04, 0.06], [0.01, -0.02, 0.03])
    2.0
    """
    R = np.asarray(returns, dtype=np.float64)
    m = np.asarray(market_returns, dtype=np.float64)
    dm = m - m.mean()
    dR = R - R.mean(axis=0)
    out = dm @ dR / (dm @ dm)
    return float(out) if np.ndim(out) == 0 else out


def capm_equilibrium(expected_returns: ArrayLike, cov: ArrayLike, risk_free: float) -> CAPMEquilibrium:
    """The market portfolio and betas of a mean-variance economy with a riskless asset.

    Parameters
    ----------
    expected_returns : array_like, shape (d,)
    cov : array_like, shape (d, d)
    risk_free : float
        Must lie below the minimum-variance portfolio's mean, so that
        the tangency portfolio is on the efficient branch.

    Returns
    -------
    CAPMEquilibrium
        Every asset lies on the security market line: ``sml_errors`` is zero.
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    C = np.asarray(cov, dtype=np.float64)
    w = safety_first_portfolio(mu, C, risk_free)
    var_m = float(w @ C @ w)
    return CAPMEquilibrium(
        market_weights=w,
        betas=C @ w / var_m,
        expected_returns=mu,
        risk_free=float(risk_free),
        market_return=float(w @ mu),
        market_vol=float(np.sqrt(var_m)),
    )


def jensen_alpha(excess_returns: ArrayLike, market_excess_returns: ArrayLike) -> FactorRegressionResult:
    """Jensen's alpha: the intercept of a regression of excess returns on the market's.

    Parameters
    ----------
    excess_returns : array_like, shape (T,) or (T, n)
        Fund (or asset) returns minus the riskless rate.
    market_excess_returns : array_like, shape (T,)

    Returns
    -------
    FactorRegressionResult
        ``alphas`` per period and their t-statistics ``alpha_t``.
    """
    return time_series_regression(excess_returns, market_excess_returns)


def zero_beta_capm(expected_returns: ArrayLike, cov: ArrayLike, zero_beta_rate: float) -> ZeroBetaResult:
    """Black's CAPM without a riskless asset.

    Any efficient portfolio :math:`m` prices every asset linearly,
    :math:`E[r_i] = E[r_z] + \\beta_{im} (E[r_m] - E[r_z])`, where
    :math:`z` is the minimum-variance portfolio uncorrelated with
    :math:`m`. Here the market is the efficient portfolio whose zero-beta
    rate is ``zero_beta_rate``: the frontier point where a tangent from
    that rate touches it.

    Parameters
    ----------
    expected_returns : array_like, shape (d,)
    cov : array_like, shape (d, d)
    zero_beta_rate : float
        Below the minimum-variance portfolio's mean.

    Returns
    -------
    ZeroBetaResult
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    C = np.asarray(cov, dtype=np.float64)
    w_m = safety_first_portfolio(mu, C, zero_beta_rate)
    d = mu.size
    # Black's zero-beta portfolio: minimise w'Cw subject to 1'w = 1 and w'C w_m = 0.
    A = np.column_stack([np.ones(d), C @ w_m])
    kkt = np.block([[2.0 * C, A], [A.T, np.zeros((2, 2))]])
    w_z = np.linalg.solve(kkt, np.concatenate([np.zeros(d), [1.0, 0.0]]))[:d]
    return ZeroBetaResult(
        market_weights=w_m,
        zero_beta_weights=w_z,
        zero_beta_rate=float(w_z @ mu),
        betas=C @ w_m / float(w_m @ C @ w_m),
        expected_returns=mu,
        market_return=float(w_m @ mu),
    )


def grs_test(excess_returns: ArrayLike, factors: ArrayLike) -> GRSResult:
    """Gibbons, Ross and Shanken's F-test that all time-series alphas are zero.

    .. math::

       F = \\frac{T - N - K}{N}\\,
           \\frac{\\hat\\alpha' \\hat\\Sigma^{-1} \\hat\\alpha}{1 + \\bar f' \\hat\\Omega^{-1} \\bar f}
           \\sim F_{N,\\, T - N - K}

    with :math:`\\hat\\Sigma` and :math:`\\hat\\Omega` the maximum-likelihood
    residual and factor covariances. The statistic measures how much the
    squared Sharpe ratio of the tangency portfolio improves when the test
    assets are added to the factors.

    Parameters
    ----------
    excess_returns : array_like, shape (T, n)
        Test assets, with ``n < T - k``.
    factors : array_like, shape (T,) or (T, k)
        Traded factor excess returns.

    Returns
    -------
    GRSResult
    """
    R = as_panel(excess_returns)
    F = as_panel(factors)
    T, N = R.shape
    K = F.shape[1]
    if T - N - K < 1:
        raise ValueError("the GRS test needs more periods than test assets plus factors")
    reg = time_series_regression(R, F)
    e = reg.residuals
    Sigma = e.T @ e / T
    f_bar = F.mean(axis=0)
    dF = F - f_bar
    Omega = dF.T @ dF / T
    a = reg.alphas
    quad = float(a @ np.linalg.solve(Sigma, a))
    theta2 = float(f_bar @ np.linalg.solve(Omega, f_bar))
    stat = (T - N - K) / N * quad / (1.0 + theta2)
    return GRSResult(statistic=stat, p_value=float(stats.f.sf(stat, N, T - N - K)), df1=N, df2=T - N - K, alphas=a)

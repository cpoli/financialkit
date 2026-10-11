"""Linear factor models: time-series regressions, Fama-MacBeth, Ross's APT,
Fama-French and Carhart.

A factor model says that expected excess returns are linear in the
assets' exposures to a few common factors,
:math:`E[r^e_i] = \\beta_i' \\lambda`. There are two ways to check it.
When the factors are themselves excess returns (the market, SMB, HML,
UMD), regress each asset on them over time: the intercepts are the
pricing errors, and the Gibbons-Ross-Shanken test
(:func:`~financialkit.asset_pricing.systems.capm.grs_test`) asks whether
they are jointly zero. Fama and MacBeth (1973) instead regress the
cross-section of returns on the betas each period and average the
slopes, which estimates the premia :math:`\\lambda` directly. Ross's
(1976) arbitrage pricing theory explains why the relation must hold:
an asset off the line could be combined with others into a portfolio
with no cost, no factor risk and a positive expected return.

References: E. F. Fama and J. D. MacBeth, "Risk, Return, and
Equilibrium: Empirical Tests," *J. Polit. Econ.* 81 (1973) 607-636;
S. A. Ross, "The Arbitrage Theory of Capital Asset Pricing," *J. Econ.
Theory* 13 (1976) 341-360; E. F. Fama and K. R. French, "Common Risk
Factors in the Returns on Stocks and Bonds," *J. Financ. Econ.* 33
(1993) 3-56; M. M. Carhart, "On Persistence in Mutual Fund
Performance," *J. Finance* 52 (1997) 57-82; J. H. Cochrane, *Asset
Pricing*, rev. ed., Princeton (2005), ch. 12.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from financialkit.asset_pricing.core.base import FactorRegressionResult, FamaMacBethResult
from financialkit.asset_pricing.utils.sorts import as_panel
from financialkit.time_series import ols

__all__ = [
    "time_series_regression",
    "fama_macbeth",
    "apt_expected_returns",
    "apt_arbitrage_portfolio",
]


def time_series_regression(excess_returns: ArrayLike, factors: ArrayLike) -> FactorRegressionResult:
    """Regress each asset's excess return on the factors, with an intercept.

    This is the test of a model whose factors are traded excess returns:
    the CAPM with the market, Fama and French's three factors, or
    Carhart's four. A single factor reproduces Jensen's alpha.

    Parameters
    ----------
    excess_returns : array_like, shape (T,) or (T, n)
        Excess returns of the test assets.
    factors : array_like, shape (T,) or (T, k)
        Factor returns.

    Returns
    -------
    FactorRegressionResult

    Examples
    --------
    >>> import numpy as np
    >>> f = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
    >>> res = time_series_regression(0.001 + 1.5 * f, f)
    >>> res.alphas.round(12).tolist(), res.betas.round(12).tolist()
    ([0.001], [[1.5]])
    """
    R = as_panel(excess_returns)
    F = as_panel(factors)
    T, n = R.shape
    X = np.column_stack([np.ones(T), F])
    fits = [ols(R[:, i], X) for i in range(n)]
    return FactorRegressionResult(
        alphas=np.array([f.coef[0] for f in fits]),
        betas=np.array([f.coef[1:] for f in fits]),
        alpha_se=np.array([f.std_errors[0] for f in fits]),
        r_squared=np.array([f.r_squared for f in fits]),
        residuals=np.column_stack([f.residuals for f in fits]),
    )


def fama_macbeth(excess_returns: ArrayLike, factors: Optional[ArrayLike] = None, betas: Optional[ArrayLike] = None) -> FamaMacBethResult:
    """Fama and MacBeth's two-pass estimate of factor risk premia.

    First pass: each asset's betas, from full-sample time-series
    regressions on ``factors`` (or supplied directly as ``betas``).
    Second pass: each period, a cross-sectional regression of the
    excess returns on a constant and the betas. The premia are the time
    averages of the slopes, with standard errors from their time-series
    variation, which absorbs cross-sectional correlation of the errors.

    Parameters
    ----------
    excess_returns : array_like, shape (T, n)
    factors : array_like, shape (T,) or (T, k), optional
    betas : array_like, shape (n,) or (n, k), optional
        Exactly one of ``factors`` and ``betas`` must be given.

    Returns
    -------
    FamaMacBethResult
        ``premia[0]`` is the average zero-beta excess return (zero if
        the model holds), ``premia[1:]`` the factor premia.
    """
    R = np.asarray(excess_returns, dtype=np.float64)
    if factors is not None and betas is None:
        B = time_series_regression(R, factors).betas
    elif betas is not None and factors is None:
        B = as_panel(betas)
    else:
        raise ValueError("give exactly one of factors and betas")
    T, n = R.shape
    X = np.column_stack([np.ones(n), B])
    gammas, *_ = np.linalg.lstsq(X, R.T, rcond=None)
    gammas = gammas.T
    return FamaMacBethResult(
        gammas=gammas,
        premia=gammas.mean(axis=0),
        std_errors=gammas.std(axis=0, ddof=1) / np.sqrt(T),
        betas=B,
    )


def apt_expected_returns(betas: ArrayLike, premia: ArrayLike, risk_free: float = 0.0) -> np.ndarray | float:
    """Ross's linear pricing relation :math:`E[r_i] = r_f + \\beta_i' \\lambda`.

    Parameters
    ----------
    betas : array_like, shape (k,) or (n, k)
        Factor exposures.
    premia : array_like, shape (k,)
        Factor risk premia :math:`\\lambda`.
    risk_free : float

    Returns
    -------
    float or ndarray, shape (n,)

    Examples
    --------
    >>> round(apt_expected_returns([1.2, 0.5], [0.05, 0.02], 0.03), 12)
    0.1
    """
    out = risk_free + np.asarray(betas, dtype=np.float64) @ np.asarray(premia, dtype=np.float64)
    return float(out) if np.ndim(out) == 0 else out


def apt_arbitrage_portfolio(expected_returns: ArrayLike, betas: ArrayLike) -> np.ndarray:
    """A zero-cost, zero-beta portfolio that earns any deviation from linear pricing.

    The weights are the residuals of the cross-sectional regression of
    expected returns on a constant and the betas. Being orthogonal to the
    constant, they sum to zero (no cost); being orthogonal to every beta,
    they carry no factor risk; and their expected return is the sum of
    squared residuals, positive unless the APT holds exactly. With many
    assets the idiosyncratic risk of this portfolio diversifies away,
    which is Ross's argument that the residuals must be small.

    Parameters
    ----------
    expected_returns : array_like, shape (n,)
    betas : array_like, shape (n,) or (n, k)

    Returns
    -------
    ndarray, shape (n,)
        Weights; their expected return is ``weights @ expected_returns``.

    Examples
    --------
    >>> import numpy as np
    >>> w = apt_arbitrage_portfolio([0.05, 0.08, 0.13], [0.0, 1.0, 2.0])
    >>> bool(np.isclose(w.sum(), 0.0, atol=1e-9)), bool(np.isclose(w @ [0.0, 1.0, 2.0], 0.0, atol=1e-9))
    (True, True)
    >>> bool(np.isclose(w @ [0.05, 0.08, 0.13], 6.6667e-05, rtol=1e-4))
    True
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    X = np.column_stack([np.ones(mu.size), as_panel(betas)])
    coef, *_ = np.linalg.lstsq(X, mu, rcond=None)
    return mu - X @ coef

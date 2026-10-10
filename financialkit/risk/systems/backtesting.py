"""VaR backtests: Kupiec's unconditional coverage, Christoffersen's
independence and conditional coverage, and the Basel traffic light.

A VaR model at level :math:`\\alpha` should be exceeded on a fraction
:math:`p = 1 - \\alpha` of days, and the exceptions should arrive
independently rather than in clusters. Each likelihood-ratio statistic
is asymptotically chi-squared under the null of a correct model.

References: P. H. Kupiec, "Techniques for Verifying the Accuracy of Risk
Measurement Models," *J. Derivatives* 3 (1995) 73-84; P. F.
Christoffersen, "Evaluating Interval Forecasts," *Int. Econ. Rev.* 39
(1998) 841-862; Basel Committee on Banking Supervision, *Supervisory
Framework for the Use of "Backtesting" in Conjunction with the Internal
Models Approach to Market Risk Capital Requirements* (1996).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import xlogy

from financialkit.risk.core.base import BacktestResult, TrafficLightResult
from financialkit.risk.utils.tail import check_alpha

__all__ = ["kupiec_test", "christoffersen_independence", "christoffersen_test", "basel_traffic_light"]

# Basel (1996), Table 2: plus factors for 5-9 exceptions in 250 days.
_PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def _hits(hits: ArrayLike) -> np.ndarray:
    h = np.asarray(hits).ravel()
    if h.size == 0:
        raise ValueError("hits is empty")
    return h.astype(bool)


def _bernoulli_loglik(x: float, n: float, p: float) -> float:
    return float(xlogy(n - x, 1.0 - p) + xlogy(x, p))


def kupiec_test(hits: ArrayLike, alpha: float = 0.99) -> BacktestResult:
    """Kupiec's proportion-of-failures test of unconditional coverage.

    .. math::

       LR_{uc} = -2 \\ln \\frac{(1 - p)^{n - x} p^x}{(1 - \\hat p)^{n - x} \\hat p^x},
       \\qquad \\hat p = x / n, \\quad LR_{uc} \\sim \\chi^2_1.

    Parameters
    ----------
    hits : array_like of bool, shape (n,)
        Exception indicator per day, e.g. from
        :func:`~financialkit.risk.utils.tail.var_exceptions`.
    alpha : float
        VaR confidence level.

    Returns
    -------
    BacktestResult

    Examples
    --------
    Twelve exceptions of a 99% VaR in 250 days are far too many.

    >>> res = kupiec_test([True] * 12 + [False] * 238)
    >>> round(res.statistic, 2), res.reject()
    (19.02, True)
    """
    h = _hits(hits)
    p = 1.0 - check_alpha(alpha)
    n, x = h.size, int(h.sum())
    lr = -2.0 * (_bernoulli_loglik(x, n, p) - _bernoulli_loglik(x, n, x / n))
    return BacktestResult(statistic=lr, p_value=float(stats.chi2.sf(lr, 1)), df=1, n_exceptions=x, n_obs=n, expected=n * p)


def christoffersen_independence(hits: ArrayLike, alpha: float = 0.99) -> BacktestResult:
    """Christoffersen's test that exceptions do not cluster.

    Fits a two-state Markov chain to the hit sequence, with
    :math:`\\pi_{01} = P(\\text{hit} \\mid \\text{no hit yesterday})` and
    :math:`\\pi_{11} = P(\\text{hit} \\mid \\text{hit yesterday})`, and
    tests :math:`\\pi_{01} = \\pi_{11}` against a single rate
    (:math:`\\chi^2_1`).

    Parameters
    ----------
    hits : array_like of bool, shape (n,)
    alpha : float
        VaR confidence level (only used to report the expected count).

    Returns
    -------
    BacktestResult
    """
    h = _hits(hits)
    prev, cur = h[:-1], h[1:]
    n00 = float(np.sum(~prev & ~cur))
    n01 = float(np.sum(~prev & cur))
    n10 = float(np.sum(prev & ~cur))
    n11 = float(np.sum(prev & cur))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    l_markov = _bernoulli_loglik(n01, n00 + n01, pi01) + _bernoulli_loglik(n11, n10 + n11, pi11)
    lr = -2.0 * (_bernoulli_loglik(n01 + n11, n00 + n01 + n10 + n11, pi) - l_markov)
    lr = max(lr, 0.0)
    p = 1.0 - check_alpha(alpha)
    return BacktestResult(statistic=lr, p_value=float(stats.chi2.sf(lr, 1)), df=1, n_exceptions=int(h.sum()), n_obs=h.size, expected=h.size * p)


def christoffersen_test(hits: ArrayLike, alpha: float = 0.99) -> BacktestResult:
    """Christoffersen's conditional-coverage test, :math:`LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2_2`.

    Parameters
    ----------
    hits : array_like of bool, shape (n,)
    alpha : float
        VaR confidence level.

    Returns
    -------
    BacktestResult
    """
    uc = kupiec_test(hits, alpha)
    ind = christoffersen_independence(hits, alpha)
    lr = uc.statistic + ind.statistic
    return BacktestResult(statistic=lr, p_value=float(stats.chi2.sf(lr, 2)), df=2, n_exceptions=uc.n_exceptions, n_obs=uc.n_obs, expected=uc.expected)


def basel_traffic_light(n_exceptions: int, n_obs: int = 250, alpha: float = 0.99) -> TrafficLightResult:
    """The Basel Committee's (1996) green, yellow and red zones.

    The zone is set by the binomial probability that an accurate model
    produces at most ``n_exceptions`` exceptions: green below 95%, red
    from 99.99%, yellow in between. For the regulatory setting (250 days,
    99% VaR) that is 0-4 green, 5-9 yellow and 10 or more red, and the
    capital multiplier is 3 plus the Committee's plus factor.

    Parameters
    ----------
    n_exceptions : int
    n_obs : int
        Days in the backtest.
    alpha : float
        VaR confidence level.

    Returns
    -------
    TrafficLightResult

    Examples
    --------
    >>> res = basel_traffic_light(6)
    >>> res.zone, res.multiplier
    ('yellow', 3.5)
    """
    p = 1.0 - check_alpha(alpha)
    cum = float(stats.binom.cdf(n_exceptions, n_obs, p))
    zone = "green" if cum < 0.95 else ("red" if cum >= 0.9999 else "yellow")
    if n_obs == 250 and np.isclose(alpha, 0.99):
        multiplier = 3.0 + (0.0 if zone == "green" else 1.0 if zone == "red" else _PLUS_FACTOR[n_exceptions])
    else:
        multiplier = float("nan")
    return TrafficLightResult(zone=zone, n_exceptions=int(n_exceptions), cumulative_probability=cum, multiplier=multiplier)

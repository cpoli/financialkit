"""VaR and ES backtests: Kupiec's unconditional coverage, Christoffersen's
independence and conditional coverage, the Basel traffic light, and
Acerbi and Szekely's expected-shortfall tests.

A VaR model at level :math:`\\alpha` should be exceeded on a fraction
:math:`p = 1 - \\alpha` of days, and the exceptions should arrive
independently rather than in clusters. Each likelihood-ratio statistic
is asymptotically chi-squared under the null of a correct model.

References: P. H. Kupiec, "Techniques for Verifying the Accuracy of Risk
Measurement Models," *J. Derivatives* 3 (1995) 73-84; P. F.
Christoffersen, "Evaluating Interval Forecasts," *Int. Econ. Rev.* 39
(1998) 841-862; Basel Committee on Banking Supervision, *Supervisory
Framework for the Use of "Backtesting" in Conjunction with the Internal
Models Approach to Market Risk Capital Requirements* (1996); C. Acerbi
and B. Szekely, "Backtesting Expected Shortfall," *Risk* 27(11) (2014)
76-81.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import xlogy

from financialkit.risk.core.base import BacktestResult, ESBacktestResult, TrafficLightResult
from financialkit.risk.utils.tail import check_alpha

__all__ = [
    "kupiec_test",
    "christoffersen_independence",
    "christoffersen_test",
    "basel_traffic_light",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "acerbi_szekely_test",
]

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


def _tail_terms(pnl: ArrayLike, var: ArrayLike, es: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(pnl, dtype=np.float64)
    v, e = np.broadcast_to(np.asarray(var, dtype=np.float64), x.shape), np.broadcast_to(np.asarray(es, dtype=np.float64), x.shape)
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("pnl is empty")
    if np.any(e <= 0):
        raise ValueError("es must be positive")
    hit = x < -v
    return np.where(hit, x / e, 0.0), hit


def acerbi_szekely_z1(pnl: ArrayLike, var: ArrayLike, es: ArrayLike) -> np.ndarray | float:
    """Acerbi and Szekely's first statistic: the size of the exceptions, given how many.

    .. math::

       Z_1 = \\frac{1}{N_T} \\sum_{t=1}^T \\frac{X_t I_t}{\\mathrm{ES}_t} + 1,
       \\qquad I_t = \\mathbf 1\\{X_t < -\\mathrm{VaR}_t\\},

    with :math:`N_T = \\sum_t I_t`. Under a correct ES its expectation is
    zero whatever the VaR; it is insensitive to the number of exceptions
    and undefined (``nan``) without any.

    Parameters
    ----------
    pnl : array_like, shape (..., T)
        Realised (or simulated) profits and losses, time on the last axis.
    var, es : array_like
        VaR and ES forecasts for each day, positive losses, broadcast
        against ``pnl``.

    Returns
    -------
    float or ndarray, shape (...)

    Examples
    --------
    Two exceptions, each 20% worse than its ES forecast:

    >>> round(acerbi_szekely_z1([-1.2, 0.5, -2.4, 0.1], [0.8, 0.8, 1.6, 0.8], [1.0, 1.0, 2.0, 1.0]), 12)
    -0.2
    """
    r, hit = _tail_terms(pnl, var, es)
    n = hit.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(n > 0, r.sum(axis=-1) / n + 1.0, np.nan)
    return float(z) if z.ndim == 0 else z


def acerbi_szekely_z2(pnl: ArrayLike, var: ArrayLike, es: ArrayLike, alpha: float = 0.975) -> np.ndarray | float:
    """Acerbi and Szekely's second statistic: the number and size of exceptions jointly.

    .. math::

       Z_2 = \\sum_{t=1}^T \\frac{X_t I_t}{T (1 - \\alpha)\\, \\mathrm{ES}_t} + 1 .

    Since :math:`\\mathrm{ES}_t = -E[X_t I_t] / (1 - \\alpha)` its
    expectation is zero under a correct model. Acerbi and Szekely found
    its critical values nearly the same for every distribution: for a
    97.5% ES over 250 days, -0.70 at the 5% level and -1.8 at 0.01%.

    Parameters
    ----------
    pnl : array_like, shape (..., T)
        Realised (or simulated) profits and losses, time on the last axis.
    var, es : array_like
        VaR and ES forecasts at level ``alpha``, positive losses.
    alpha : float
        Confidence level of the forecasts.

    Returns
    -------
    float or ndarray, shape (...)

    Examples
    --------
    Seven exceptions in 250 days where 6.25 are expected, each at its
    ES forecast: slightly too many.

    >>> import numpy as np
    >>> x = np.r_[np.full(7, -2.34), np.zeros(243)]
    >>> round(acerbi_szekely_z2(x, 1.96, 2.34), 6)
    -0.12
    """
    p = 1.0 - check_alpha(alpha)
    r, _ = _tail_terms(pnl, var, es)
    z = r.sum(axis=-1) / (r.shape[-1] * p) + 1.0
    return float(z) if z.ndim == 0 else z


def acerbi_szekely_test(
    pnl: ArrayLike, var: ArrayLike, es: ArrayLike, alpha: float = 0.975, test: str = "z2", simulated: ArrayLike | None = None
) -> ESBacktestResult:
    """Acerbi-Szekely backtest of expected shortfall, with a simulated p-value.

    The statistic is compared with its distribution under the model:
    ``simulated`` holds profit-and-loss paths drawn from the forecast
    distributions, and the p-value is the fraction of their statistics
    at or below the observed one (the test is one-sided, since only an
    underestimated risk matters to a regulator).

    Parameters
    ----------
    pnl : array_like, shape (T,)
        Realised profits and losses.
    var, es : array_like, shape (T,) or scalar
        VaR and ES forecasts at level ``alpha``, positive losses.
    alpha : float
        Confidence level of the forecasts.
    test : {"z1", "z2"}
    simulated : array_like, shape (n_paths, T), optional
        Paths simulated under the model; without them ``p_value`` is
        ``nan`` and the statistic can be read against Acerbi and
        Szekely's critical values.

    Returns
    -------
    ESBacktestResult
    """
    alpha = check_alpha(alpha)
    x = np.asarray(pnl, dtype=np.float64).ravel()
    if test == "z1":
        stat_fn = lambda y: acerbi_szekely_z1(y, var, es)
    elif test == "z2":
        stat_fn = lambda y: acerbi_szekely_z2(y, var, es, alpha)
    else:
        raise ValueError(f"test must be 'z1' or 'z2', got {test!r}")
    z = float(stat_fn(x))
    p_value = float("nan")
    if simulated is not None:
        sims = np.asarray(simulated, dtype=np.float64)
        if sims.ndim != 2 or sims.shape[1] != x.size:
            raise ValueError(f"simulated must have shape (n_paths, {x.size}), got {sims.shape}")
        zs = np.asarray(stat_fn(sims))
        # Z1 is conditional on at least one exception.
        zs = zs[np.isfinite(zs)]
        if np.isfinite(z) and zs.size:
            p_value = float(np.mean(zs <= z))
    n_exc = int(np.sum(x < -np.broadcast_to(np.asarray(var, dtype=np.float64), x.shape)))
    return ESBacktestResult(statistic=z, p_value=p_value, test=test, n_exceptions=n_exc, n_obs=x.size, expected=x.size * (1.0 - alpha))

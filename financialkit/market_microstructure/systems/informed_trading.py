"""The probability of informed trading, by maximum likelihood (PIN) and by volume buckets (VPIN).

David Easley, Nicholas Kiefer, Maureen O'Hara and Joseph Paperman (1996)
turned Glosten-Milgrom into an estimator. Each day an information event
happens with probability :math:`\\alpha`, bad news with probability
:math:`\\delta`; uninformed buys and sells arrive as Poisson streams with
rates :math:`\\varepsilon_b, \\varepsilon_s`, and on event days informed
traders add rate :math:`\\mu` to one side. Fitting the mixture to daily
buy and sell counts gives the probability that a trade is informed,

.. math::

   \\mathrm{PIN} = \\frac{\\alpha\\mu}{\\alpha\\mu + \\varepsilon_b + \\varepsilon_s}.

Easley, Lopez de Prado and O'Hara (2012) replaced the daily likelihood by
a running measure for high-frequency markets: group trading into buckets
of equal volume, split each bar's volume into buys and sells by *bulk
volume classification*, and average the absolute order imbalance over
the last :math:`n` buckets.

References: D. Easley, N. M. Kiefer, M. O'Hara and J. B. Paperman,
"Liquidity, Information, and Infrequently Traded Stocks," *J. Finance*
51 (1996) 1405-1436; D. Easley, M. M. Lopez de Prado and M. O'Hara, "Flow
Toxicity and Liquidity in a High-Frequency World," *Review of Financial
Studies* 25 (2012) 1457-1493; Y. Yan and S. Zhang, "An Improved Estimation
Method and Empirical Properties of the Probability of Informed Trading,"
*J. Banking Finance* 36 (2012) 454-467.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special, stats

from financialkit.market_microstructure.core.base import PINResult, VPINResult
from financialkit.time_series.utils.numerics import numerical_hessian

__all__ = ["pin_loglik", "fit_pin", "simulate_pin", "bulk_volume_classification", "vpin"]


def pin_loglik(params: ArrayLike, buys: ArrayLike, sells: ArrayLike) -> float:
    """Log-likelihood of daily buy and sell counts under the EKOP mixture.

    Parameters
    ----------
    params : array_like
        ``(alpha, delta, mu, eps_b, eps_s)``.
    buys, sells : array_like of int, shape (n_days,)

    Returns
    -------
    float
    """
    alpha, delta, mu, eb, es = np.asarray(params, dtype=np.float64).tolist()
    B = np.asarray(buys, dtype=np.float64)
    S = np.asarray(sells, dtype=np.float64)
    pb, ps = stats.poisson.logpmf(B, eb), stats.poisson.logpmf(S, es)
    with np.errstate(divide="ignore"):
        terms = np.stack(
            [
                np.log1p(-alpha) + pb + ps,
                np.log(alpha * delta) + pb + stats.poisson.logpmf(S, es + mu),
                np.log(alpha * (1 - delta)) + stats.poisson.logpmf(B, eb + mu) + ps,
            ]
        )
    return float(special.logsumexp(terms, axis=0).sum())


def simulate_pin(
    n_days: int, alpha: float, delta: float, mu: float, eps_b: float, eps_s: float, rng: np.random.Generator | int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Daily buy and sell counts from the EKOP model.

    Returns
    -------
    buys, sells : ndarray of int, shape (n_days,)
    """
    rng = np.random.default_rng(rng)
    event = rng.random(n_days) < alpha
    bad = rng.random(n_days) < delta
    buys = rng.poisson(eps_b + mu * (event & ~bad))
    sells = rng.poisson(eps_s + mu * (event & bad))
    return buys, sells


def _to_natural(z: np.ndarray) -> np.ndarray:
    return np.concatenate([special.expit(z[:2]), np.exp(z[2:])])


def fit_pin(buys: ArrayLike, sells: ArrayLike) -> PINResult:
    """Fit the EKOP model by maximum likelihood (:func:`scipy.optimize.minimize`).

    The probabilities are optimised on the logit scale and the rates on
    the log scale, from a small grid of starting points as Yan and Zhang
    (2012) recommend, because the likelihood has local maxima.

    Parameters
    ----------
    buys, sells : array_like of int, shape (n_days,)
        Daily counts of buyer- and seller-initiated trades.

    Returns
    -------
    PINResult
    """
    B = np.asarray(buys, dtype=np.float64)
    S = np.asarray(sells, dtype=np.float64)
    mb, ms = B.mean(), S.mean()
    imbalance = np.abs(B - S).mean()

    def nll(z):
        return -pin_loglik(_to_natural(z), B, S)

    best = None
    for a0 in (0.2, 0.5, 0.8):
        for d0 in (0.3, 0.7):
            mu0 = max(imbalance / a0, 1.0)
            eb0 = max(mb - a0 * (1 - d0) * mu0, 0.1 * mb + 1e-3)
            es0 = max(ms - a0 * d0 * mu0, 0.1 * ms + 1e-3)
            z0 = np.array([special.logit(a0), special.logit(d0), np.log(mu0), np.log(eb0), np.log(es0)])
            res = optimize.minimize(nll, z0, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
    assert best is not None
    theta = _to_natural(best.x)
    alpha, delta, mu, eb, es = (float(v) for v in theta)
    D = alpha * mu + eb + es
    pin = alpha * mu / D
    grad = np.array([mu * (eb + es), 0.0, alpha * (eb + es), -alpha * mu, -alpha * mu]) / D**2
    H = numerical_hessian(lambda p: -pin_loglik(p, B, S), theta)
    try:
        se = float(np.sqrt(grad @ np.linalg.solve(H, grad)))
    except np.linalg.LinAlgError:
        se = float("nan")
    return PINResult(alpha=alpha, delta=delta, mu=mu, eps_b=eb, eps_s=es, pin=pin, pin_se=se, loglik=-float(best.fun))


def bulk_volume_classification(price_changes: ArrayLike, volumes: ArrayLike, sigma: float | None = None) -> np.ndarray:
    """Buy volume of each bar by bulk volume classification, :math:`V \\Phi(\\Delta P / \\sigma_{\\Delta P})`.

    Parameters
    ----------
    price_changes : array_like, shape (n,)
        Price change over each bar.
    volumes : array_like, shape (n,)
        Volume traded in each bar.
    sigma : float, optional
        Scale of price changes; their sample standard deviation if omitted.

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> bulk_volume_classification([0.0, 1.0], [100.0, 100.0], sigma=1.0).round(2).tolist()
    [50.0, 84.13]
    """
    dp = np.asarray(price_changes, dtype=np.float64)
    if sigma is None and dp.size < 2:
        raise ValueError("need two or more bars to estimate sigma")
    s = float(dp.std(ddof=1)) if sigma is None else sigma
    return np.asarray(volumes, dtype=np.float64) * stats.norm.cdf(dp / s)


def vpin(price_changes: ArrayLike, volumes: ArrayLike, bucket_volume: float, window: int = 50, sigma: float | None = None) -> VPINResult:
    """Volume-synchronized probability of informed trading.

    Bars are classified by :func:`bulk_volume_classification`, their
    volume is poured into consecutive buckets of size `bucket_volume`
    (a bar may straddle buckets), and VPIN is the mean of
    :math:`|V^B_\\tau - V^S_\\tau| / V` over the last `window` buckets.

    Parameters
    ----------
    price_changes, volumes : array_like, shape (n,)
        Time bars.
    bucket_volume : float
    window : int
    sigma : float, optional
        Passed to :func:`bulk_volume_classification`.

    Returns
    -------
    VPINResult
    """
    v = np.asarray(volumes, dtype=np.float64)
    if np.any(v < 0) or bucket_volume <= 0:
        raise ValueError("volumes must be non-negative and bucket_volume positive")
    buy = bulk_volume_classification(price_changes, v, sigma)
    cum_v = np.concatenate([[0.0], np.cumsum(v)])
    cum_b = np.concatenate([[0.0], np.cumsum(buy)])
    n_buckets = int(cum_v[-1] // bucket_volume)
    if n_buckets < 1:
        raise ValueError("total volume is less than one bucket")
    edges = bucket_volume * np.arange(n_buckets + 1)
    bucket_buy = np.diff(np.interp(edges, cum_v, cum_b))
    bucket_sell = bucket_volume - bucket_buy
    imbalance = np.abs(bucket_buy - bucket_sell) / bucket_volume
    roll = np.full(n_buckets, np.nan)
    if n_buckets >= window:
        c = np.concatenate([[0.0], np.cumsum(imbalance)])
        roll[window - 1 :] = (c[window:] - c[:-window]) / window
    end = np.searchsorted(cum_v, edges[1:], side="left") - 1
    return VPINResult(vpin=roll, buy_volume=bucket_buy, sell_volume=bucket_sell, bucket_end=end)

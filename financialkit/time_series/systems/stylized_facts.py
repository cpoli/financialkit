"""Returns and their stylized facts: fat tails and the tail index.

Mandelbrot (1963) found that cotton price changes have far too many large
moves for a normal distribution, with tails that decay like a power law,
and that large changes cluster in time. Hill (1975) gave the estimator of
that power, the tail index :math:`\\alpha`, from the largest order
statistics. References: B. Mandelbrot, "The Variation of Certain
Speculative Prices," Journal of Business 36(4) (1963), 394-419; B. M.
Hill, "A Simple General Approach to Inference About the Tail of a
Distribution," Annals of Statistics 3(5) (1975), 1163-1174; R. Cont,
"Empirical Properties of Asset Returns: Stylized Facts and Statistical
Issues," Quantitative Finance 1(2) (2001), 223-236.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from financialkit.time_series.core.base import ReturnStatistics, TailIndexResult

__all__ = ["log_returns", "simple_returns", "return_statistics", "tail_survival", "hill_estimator"]


def log_returns(prices) -> np.ndarray:
    """Continuously compounded returns :math:`\\ln(P_t / P_{t-1})`.

    Examples
    --------
    >>> import numpy as np
    >>> log_returns([100.0, 100.0 * np.e]).round(12).tolist()
    [1.0]
    """
    p = np.asarray(prices, dtype=np.float64)
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    return np.diff(np.log(p))


def simple_returns(prices) -> np.ndarray:
    """Simple returns :math:`P_t / P_{t-1} - 1`.

    Examples
    --------
    >>> simple_returns([100.0, 110.0, 99.0]).round(12).tolist()
    [0.1, -0.1]
    """
    p = np.asarray(prices, dtype=np.float64)
    return p[1:] / p[:-1] - 1.0


def return_statistics(returns) -> ReturnStatistics:
    """Mean, volatility, skewness, excess kurtosis and the Jarque-Bera test.

    Parameters
    ----------
    returns : array_like

    Returns
    -------
    ReturnStatistics

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = return_statistics(rng.standard_t(3, 20_000))
    >>> bool(s.excess_kurtosis > 3), bool(s.jarque_bera_pvalue < 1e-6)
    (True, True)
    """
    r = np.asarray(returns, dtype=np.float64)
    jb = stats.jarque_bera(r)
    return ReturnStatistics(
        nobs=r.size,
        mean=float(r.mean()),
        std=float(r.std(ddof=1)),
        skewness=float(stats.skew(r)),
        excess_kurtosis=float(stats.kurtosis(r)),
        jarque_bera=float(jb.statistic),
        jarque_bera_pvalue=float(jb.pvalue),
    )


def tail_survival(x, side: str = "both") -> tuple[np.ndarray, np.ndarray]:
    """Empirical survival function :math:`P(|X| > u)` at the sorted sample points.

    On log-log axes a power-law tail :math:`P(X > u) \\sim u^{-\\alpha}` is
    a straight line of slope :math:`-\\alpha`, Mandelbrot's diagnostic.

    Parameters
    ----------
    x : array_like
    side : {"both", "upper", "lower"}
        Use :math:`|x|`, the positive values, or the magnitudes of the negative values.

    Returns
    -------
    u : ndarray
        Sorted magnitudes (ascending).
    prob : ndarray
        Fraction of the sample strictly above each `u`, i.e. ``(n - i) / n``
        shifted so the largest point has probability ``1/n``.
    """
    x = np.asarray(x, dtype=np.float64)
    u = {"both": np.abs(x), "upper": x[x > 0], "lower": -x[x < 0]}[side]
    u = np.sort(u)
    n = u.size
    return u, (n - np.arange(n)) / n


def hill_estimator(x, k=None, side: str = "upper") -> TailIndexResult:
    """Hill's estimator of the tail index :math:`\\alpha` from the `k` largest observations.

    .. math:: \\hat\\alpha_k = \\Bigl[\\frac1k \\sum_{i=1}^{k} \\ln X_{(i)} - \\ln X_{(k+1)}\\Bigr]^{-1}

    with :math:`X_{(1)} \\ge X_{(2)} \\ge \\dots` the descending order
    statistics. It is the maximum-likelihood estimator for an exact Pareto
    tail above :math:`X_{(k+1)}`, with asymptotic standard error
    :math:`\\alpha/\\sqrt k`. Plotting :math:`\\hat\\alpha_k` against `k`
    (the *Hill plot*) shows the trade-off: small `k` is noisy, large `k`
    reaches into the body of the distribution and is biased.

    Parameters
    ----------
    x : array_like
    k : int or array_like of int, optional
        Numbers of order statistics; defaults to ``2 .. n // 4``.
    side : {"upper", "lower", "both"}
        Which tail: positive values, negated negative values, or :math:`|x|`.

    Returns
    -------
    TailIndexResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> x = rng.pareto(3.0, 100_000) + 1.0  # exact Pareto tail, alpha = 3
    >>> res = hill_estimator(x, k=5000)
    >>> bool(abs(res.alpha[0] - 3.0) < 4 * res.std_error[0])
    True
    """
    x = np.asarray(x, dtype=np.float64)
    tail = {"upper": x[x > 0], "lower": -x[x < 0], "both": np.abs(x[x != 0])}[side]
    desc = np.sort(tail)[::-1]
    n = desc.size
    ks = np.arange(2, max(n // 4, 3)) if k is None else np.atleast_1d(np.asarray(k, dtype=np.int64))
    if np.any(ks < 1) or np.any(ks >= n):
        raise ValueError("k must lie in [1, n - 1] for the chosen tail")
    logs = np.log(desc)
    cum = np.cumsum(logs)
    inv_alpha = cum[ks - 1] / ks - logs[ks]
    alpha = 1.0 / inv_alpha
    return TailIndexResult(k=ks, alpha=alpha, std_error=alpha / np.sqrt(ks), threshold=desc[ks])

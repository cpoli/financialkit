"""Tests of the random walk: serial correlation, runs, portmanteau statistics and variance ratios.

Kendall (1953) found the serial correlations of weekly price changes to be
statistically indistinguishable from zero; Fama (1965) confirmed it with a
distribution-free runs test; Box and Pierce (1970) and Ljung and Box
(1978) pooled many autocorrelations into one portmanteau statistic; and Lo
and MacKinlay (1988) compared the variance of long-horizon returns with
that of short ones, which proved more powerful and rejected the random walk
for weekly U.S. stock indices. References: M. G. Kendall, "The Analysis of
Economic Time-Series -- Part I: Prices," Journal of the Royal Statistical
Society A 116(1) (1953), 11-34; E. F. Fama, "The Behavior of Stock-Market
Prices," Journal of Business 38(1) (1965), 34-105; G. M. Ljung and G. E.
P. Box, "On a Measure of Lack of Fit in Time Series Models," Biometrika
65(2) (1978), 297-303; A. W. Lo and A. C. MacKinlay, "Stock Market Prices
Do Not Follow Random Walks," Review of Financial Studies 1(1) (1988),
41-66; Campbell, Lo and MacKinlay, *The Econometrics of Financial Markets*
(Princeton, 1997), Ch. 2.
"""

from __future__ import annotations

import numpy as np
from scipy import fft, stats

from financialkit.time_series.core.base import TestResult, VarianceRatioResult

__all__ = ["autocovariance", "autocorrelation", "bartlett_band", "runs_test", "box_pierce", "ljung_box", "variance_ratio"]


def autocovariance(x, nlags: int) -> np.ndarray:
    """Sample autocovariances :math:`\\hat\\gamma_k = \\frac1n\\sum_{t=k}^{n-1}(x_t-\\bar x)(x_{t-k}-\\bar x)`, ``k = 0..nlags``.

    Computed with :func:`scipy.fft.rfft` (the divisor `n` keeps the
    sequence positive semidefinite).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if not 0 <= nlags < n:
        raise ValueError("nlags must lie in [0, n - 1]")
    d = x - x.mean()
    m = fft.next_fast_len(2 * n)
    f = fft.rfft(d, m)
    return fft.irfft(f * np.conj(f), m)[: nlags + 1] / n


def autocorrelation(x, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelations :math:`\\hat\\rho_k = \\hat\\gamma_k / \\hat\\gamma_0`, ``k = 0..nlags``.

    Kendall's (1953) serial correlations: for price changes of a random walk
    they are close to zero at every lag, while for price levels they are
    close to one.

    Examples
    --------
    >>> import numpy as np
    >>> rho = autocorrelation(np.random.default_rng(0).standard_normal(10_000), 3)
    >>> float(rho[0]), bool(np.all(np.abs(rho[1:]) < 4 / np.sqrt(10_000)))
    (1.0, True)
    """
    g = autocovariance(x, nlags)
    return g / g[0]


def bartlett_band(nobs: int, level: float = 0.95) -> float:
    """Half-width :math:`z_{1-\\alpha/2}/\\sqrt n` of the band that white-noise autocorrelations stay within.

    Examples
    --------
    >>> round(bartlett_band(400), 4)
    0.098
    """
    return float(stats.norm.ppf(0.5 + level / 2) / np.sqrt(nobs))


def runs_test(x) -> TestResult:
    """Fama's (1965) runs test on the signs of price changes.

    A *run* is a maximal sequence of changes with the same sign (+, - or 0).
    With :math:`n_i` changes of sign :math:`i` out of :math:`N`, the number of
    runs under independence has mean and variance

    .. math::

       m = \\frac{N(N+1) - \\sum_i n_i^2}{N}, \\qquad
       \\sigma_m^2 = \\frac{\\sum_i n_i^2\\bigl[\\sum_i n_i^2 + N(N+1)\\bigr]
                     - 2N\\sum_i n_i^3 - N^3}{N^2(N-1)}.

    Too few runs means positive dependence (trends), too many means
    reversals. The statistic is the normal approximation with a half-run
    continuity correction; with two signs it reduces to the Wald-Wolfowitz
    test.

    Returns
    -------
    TestResult
        Two-sided p-value; ``statistic`` is negative when there are fewer runs than expected.

    Examples
    --------
    >>> res = runs_test([1, 1, 1, 1, 1, -1, -1, -1, -1, -1] * 4)
    >>> bool(res.statistic < -3)
    True
    """
    s = np.sign(np.asarray(x, dtype=np.float64))
    N = s.size
    if N < 3:
        raise ValueError("need at least three observations")
    runs = 1 + int(np.count_nonzero(s[1:] != s[:-1]))
    counts = np.array([np.count_nonzero(s == v) for v in (-1.0, 0.0, 1.0)], dtype=np.float64)
    s2, s3 = np.sum(counts**2), np.sum(counts**3)
    m = (N * (N + 1) - s2) / N
    var = (s2 * (s2 + N * (N + 1)) - 2 * N * s3 - N**3) / (N**2 * (N - 1))
    if var <= 0:
        raise ValueError("all changes have the same sign")
    diff = runs - m
    diff = diff - np.sign(diff) * min(0.5, abs(diff))
    z = diff / np.sqrt(var)
    return TestResult(name="runs", statistic=float(z), pvalue=float(2 * stats.norm.sf(abs(z))), nobs=N)


def _portmanteau(x, lags: int, fitted_params: int, ljung: bool) -> TestResult:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    rho = autocorrelation(x, lags)[1:]
    k = np.arange(1, lags + 1)
    Q = n * (n + 2) * np.sum(rho**2 / (n - k)) if ljung else n * np.sum(rho**2)
    df = lags - fitted_params
    if df <= 0:
        raise ValueError("lags must exceed the number of fitted ARMA parameters")
    return TestResult(name="Ljung-Box" if ljung else "Box-Pierce", statistic=float(Q), pvalue=float(stats.chi2.sf(Q, df)), df=df, nobs=n)


def box_pierce(x, lags: int = 10, fitted_params: int = 0) -> TestResult:
    """Box-Pierce (1970) portmanteau statistic :math:`Q = n\\sum_{k=1}^h \\hat\\rho_k^2 \\sim \\chi^2_{h - p - q}`."""
    return _portmanteau(x, lags, fitted_params, ljung=False)


def ljung_box(x, lags: int = 10, fitted_params: int = 0) -> TestResult:
    """Ljung-Box (1978) portmanteau statistic.

    .. math:: Q = n(n+2)\\sum_{k=1}^{h} \\frac{\\hat\\rho_k^2}{n-k} \\;\\sim\\; \\chi^2_{h - p - q}

    The :math:`(n+2)/(n-k)` weights correct the Box-Pierce statistic's
    small-sample downward bias. Apply it to ARMA residuals with
    ``fitted_params = p + q``, or to squared returns to detect volatility
    clustering.

    Parameters
    ----------
    x : array_like
    lags : int
        Number of autocorrelations pooled, :math:`h`.
    fitted_params : int
        Degrees of freedom lost to estimation.

    Returns
    -------
    TestResult

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> e = np.random.default_rng(0).standard_normal(1000)
    >>> ljung_box(e, 10).rejects(), ljung_box(lfilter([1.0], [1.0, -0.3], e), 10).rejects()
    (False, True)
    """
    return _portmanteau(x, lags, fitted_params, ljung=True)


def variance_ratio(log_prices, q=(2, 4, 8, 16)) -> VarianceRatioResult:
    """Lo-MacKinlay (1988) variance-ratio test of the random walk.

    If log prices follow a random walk, the variance of `q`-period returns
    is `q` times that of one-period returns. With overlapping `q`-period
    returns and both variances bias-corrected,

    .. math::

       VR(q) = \\frac{\\hat\\sigma_c^2(q)}{\\hat\\sigma_a^2}, \\qquad
       z(q) = \\frac{VR(q) - 1}{\\sqrt{2(2q-1)(q-1)/(3qT)}},

    and the heteroskedasticity-consistent :math:`z^*(q) = \\sqrt T\\,(VR(q)-1)/\\sqrt{\\hat\\theta(q)}`
    with :math:`\\hat\\theta(q) = \\sum_{j=1}^{q-1} [2(q-j)/q]^2 \\hat\\delta(j)`
    stays valid under volatility clustering (Lo and MacKinlay 1988, eqs.
    (12)-(14) and Theorem 3). Since
    :math:`VR(q) - 1 \\approx 2\\sum_{k<q}(1-k/q)\\rho_k`, positive
    autocorrelation pushes the ratio above one.

    Parameters
    ----------
    log_prices : array_like, shape (T + 1,)
    q : int or sequence of int
        Horizons, each at least 2.

    Returns
    -------
    VarianceRatioResult

    Examples
    --------
    >>> import numpy as np
    >>> p = np.cumsum(np.random.default_rng(3).standard_normal(5001))
    >>> res = variance_ratio(p, q=[2, 4])
    >>> bool(np.all(np.abs(res.z) < 3))
    True
    """
    p = np.asarray(log_prices, dtype=np.float64)
    x = np.diff(p)
    T = x.size
    qs = np.atleast_1d(np.asarray(q, dtype=np.int64))
    if np.any(qs < 2) or np.any(qs >= T):
        raise ValueError("each q must lie in [2, T - 1]")
    mu = (p[-1] - p[0]) / T
    dev = x - mu
    var_a = np.sum(dev**2) / (T - 1)
    dev2 = dev**2
    denom = np.sum(dev2) ** 2
    ratio, z, z_rob = (np.empty(qs.size) for _ in range(3))
    for i, qq in enumerate(qs):
        m = qq * (T - qq + 1) * (1 - qq / T)
        var_c = np.sum((p[qq:] - p[:-qq] - qq * mu) ** 2) / m
        vr = var_c / var_a
        j = np.arange(1, qq)
        delta = np.array([T * np.sum(dev2[jj:] * dev2[:-jj]) / denom for jj in j])
        theta = np.sum((2 * (qq - j) / qq) ** 2 * delta)
        ratio[i] = vr
        z[i] = (vr - 1) / np.sqrt(2 * (2 * qq - 1) * (qq - 1) / (3 * qq * T))
        z_rob[i] = (vr - 1) / np.sqrt(theta / T)
    return VarianceRatioResult(q=qs, ratio=ratio, z=z, z_robust=z_rob, nobs=T)

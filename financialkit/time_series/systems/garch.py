"""Conditional heteroskedasticity: ARCH, GARCH, EGARCH, GJR-GARCH and RiskMetrics EWMA.

Engle (1982) let today's variance depend on yesterday's squared shock;
Bollerslev (1986) added yesterday's variance, so one lag captures the slow
decay of volatility clustering; Nelson (1991) modelled the log variance,
which needs no positivity constraints and responds asymmetrically to good
and bad news; Glosten, Jagannathan and Runkle (1993) obtained the same
leverage effect with an extra coefficient on negative shocks; and J.P.
Morgan's RiskMetrics (1994) fixed the GARCH weights at
:math:`\\lambda = 0.94` with no constant, an exponentially weighted moving
average. Every model is a :class:`~financialkit.time_series.core.base.VolatilityModel`,
estimated by Gaussian maximum likelihood (the likelihood recursion is
hand-written; the optimizer is :func:`scipy.optimize.minimize`).
References: R. F. Engle, Econometrica 50(4) (1982), 987-1007; T.
Bollerslev, Journal of Econometrics 31(3) (1986), 307-327; D. B. Nelson,
Econometrica 59(2) (1991), 347-370; L. R. Glosten, R. Jagannathan and D.
E. Runkle, Journal of Finance 48(5) (1993), 1779-1801; J.P. Morgan/Reuters,
*RiskMetrics -- Technical Document*, 4th ed. (1996), Ch. 5; Tsay,
*Analysis of Financial Time Series*, 3rd ed. (Wiley, 2010), Ch. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.signal import lfilter

from financialkit.time_series.core.base import TestResult, VolatilityModel
from financialkit.time_series.utils.numerics import lag_matrix, ols

__all__ = ["ARCH", "GARCH", "EGARCH", "GJRGARCH", "EWMA", "arch_lm_test"]

_ABS_MEAN = math.sqrt(2.0 / math.pi)  # E|z| for a standard normal z


def _check_v0(eps: np.ndarray, v0: Optional[float]) -> float:
    return float(np.mean(eps**2)) if v0 is None else float(v0)


def _forecast_mean_reverting(v_next: float, v_bar: float, persistence: float, h: int) -> np.ndarray:
    """:math:`\\bar\\sigma^2 + \\pi^{k-1}(\\sigma^2_{T+1} - \\bar\\sigma^2)`, ``k = 1..h``."""
    return v_bar + persistence ** np.arange(h) * (v_next - v_bar)


@dataclass(repr=False)
class ARCH(VolatilityModel):
    """Engle's (1982) ARCH(1): :math:`\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2`.

    Covariance stationary when :math:`\\alpha < 1`, with kurtosis
    :math:`3(1-\\alpha^2)/(1-3\\alpha^2)` for :math:`\\alpha^2 < 1/3`: the
    model fattens the tails of returns even with normal innovations.

    Parameters
    ----------
    omega, alpha : float
    mu : float
        Mean return.

    Examples
    --------
    >>> m = ARCH(omega=1e-4, alpha=0.5)
    >>> round(m.unconditional_variance, 8), round(m.kurtosis, 4)
    (0.0002, 9.0)
    """

    omega: float
    alpha: float
    mu: float = 0.0
    param_names = ("mu", "omega", "alpha")

    def news_impact(self, eps, var_prev):
        return self.omega + self.alpha * np.asarray(eps) ** 2 + 0.0 * np.asarray(var_prev)

    def variance(self, returns, v0: Optional[float] = None) -> np.ndarray:
        eps = np.asarray(returns, dtype=np.float64) - self.mu
        return np.r_[_check_v0(eps, v0), self.omega + self.alpha * eps[:-1] ** 2]

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1 - self.alpha) if self.alpha < 1 else math.inf

    @property
    def kurtosis(self) -> float:
        """Kurtosis of the returns (``inf`` when :math:`3\\alpha^2 \\ge 1`)."""
        a2 = self.alpha**2
        return 3 * (1 - a2) / (1 - 3 * a2) if 3 * a2 < 1 else math.inf

    @classmethod
    def _start(cls, r):
        return np.array([r.mean(), 0.7, 0.3])

    @classmethod
    def _bounds(cls):
        return [(None, None), (1e-6, 10.0), (0.0, 0.9999)]

    def _rescaled(self, s):
        return ARCH(omega=self.omega * s**2, alpha=self.alpha, mu=self.mu * s)


@dataclass(repr=False)
class GARCH(VolatilityModel):
    """Bollerslev's (1986) GARCH(1,1): :math:`\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2`.

    Equivalent to an ARCH(:math:`\\infty`) with geometrically declining
    weights. Covariance stationary when the persistence :math:`\\alpha +
    \\beta < 1`; shocks to the variance then decay at rate
    :math:`(\\alpha + \\beta)^h`, a half-life of
    :math:`\\ln\\frac12 / \\ln(\\alpha + \\beta)` periods.

    Parameters
    ----------
    omega, alpha, beta : float
    mu : float

    Examples
    --------
    >>> m = GARCH(omega=2e-6, alpha=0.08, beta=0.9)
    >>> round(m.unconditional_variance, 8), round(m.half_life, 1)
    (0.0001, 34.3)
    """

    omega: float
    alpha: float
    beta: float
    mu: float = 0.0
    param_names = ("mu", "omega", "alpha", "beta")

    def news_impact(self, eps, var_prev):
        return self.omega + self.alpha * np.asarray(eps) ** 2 + self.beta * np.asarray(var_prev)

    def variance(self, returns, v0: Optional[float] = None) -> np.ndarray:
        eps = np.asarray(returns, dtype=np.float64) - self.mu
        v0 = _check_v0(eps, v0)
        u = self.omega + self.alpha * eps[:-1] ** 2
        rest = lfilter([1.0], [1.0, -self.beta], u, zi=[self.beta * v0])[0]
        return np.r_[v0, rest]

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1 - self.persistence) if self.persistence < 1 else math.inf

    @property
    def half_life(self) -> float:
        """Periods for a variance shock to halve."""
        return math.log(0.5) / math.log(self.persistence)

    def forecast(self, returns, h: int) -> np.ndarray:
        """Variance forecasts 1..`h` periods after the end of `returns`.

        .. math:: E_T[\\sigma_{T+k}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{k-1}(\\sigma_{T+1}^2 - \\bar\\sigma^2)
        """
        r = np.asarray(returns, dtype=np.float64)
        v = self.variance(r)
        v_next = float(self.news_impact(r[-1] - self.mu, v[-1]))
        return _forecast_mean_reverting(v_next, self.unconditional_variance, self.persistence, h)

    @classmethod
    def _start(cls, r):
        return np.array([r.mean(), 0.05, 0.08, 0.87])

    @classmethod
    def _bounds(cls):
        return [(None, None), (1e-6, 10.0), (0.0, 1.0), (0.0, 1.0)]

    @classmethod
    def _constraints(cls):
        return [{"type": "ineq", "fun": lambda v: 0.9999 - v[2] - v[3]}]

    def _rescaled(self, s):
        return GARCH(omega=self.omega * s**2, alpha=self.alpha, beta=self.beta, mu=self.mu * s)


@dataclass(repr=False)
class GJRGARCH(VolatilityModel):
    """Glosten-Jagannathan-Runkle (1993) GARCH with a leverage term.

    .. math:: \\sigma_t^2 = \\omega + (\\alpha + \\gamma\\,\\mathbf 1\\{\\varepsilon_{t-1} < 0\\})\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2

    With symmetric innovations the persistence is :math:`\\alpha +
    \\gamma/2 + \\beta`. :math:`\\gamma > 0` makes bad news raise volatility
    more than good news of the same size.

    Examples
    --------
    >>> m = GJRGARCH(omega=1e-6, alpha=0.02, gamma=0.1, beta=0.9)
    >>> round(m.persistence, 2), float(m.news_impact(-0.01, 1e-4) > m.news_impact(0.01, 1e-4))
    (0.97, 1.0)
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    mu: float = 0.0
    param_names = ("mu", "omega", "alpha", "gamma", "beta")

    def news_impact(self, eps, var_prev):
        eps = np.asarray(eps)
        return self.omega + (self.alpha + self.gamma * (eps < 0)) * eps**2 + self.beta * np.asarray(var_prev)

    def variance(self, returns, v0: Optional[float] = None) -> np.ndarray:
        eps = np.asarray(returns, dtype=np.float64) - self.mu
        v0 = _check_v0(eps, v0)
        e = eps[:-1]
        u = self.omega + (self.alpha + self.gamma * (e < 0)) * e**2
        rest = lfilter([1.0], [1.0, -self.beta], u, zi=[self.beta * v0])[0]
        return np.r_[v0, rest]

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1 - self.persistence) if self.persistence < 1 else math.inf

    def forecast(self, returns, h: int) -> np.ndarray:
        """Variance forecasts 1..`h` periods ahead (persistence :math:`\\alpha + \\gamma/2 + \\beta`)."""
        r = np.asarray(returns, dtype=np.float64)
        v = self.variance(r)
        v_next = float(self.news_impact(r[-1] - self.mu, v[-1]))
        return _forecast_mean_reverting(v_next, self.unconditional_variance, self.persistence, h)

    @classmethod
    def _start(cls, r):
        return np.array([r.mean(), 0.05, 0.03, 0.08, 0.87])

    @classmethod
    def _bounds(cls):
        return [(None, None), (1e-6, 10.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]

    @classmethod
    def _constraints(cls):
        return [
            {"type": "ineq", "fun": lambda v: 0.9999 - v[2] - 0.5 * v[3] - v[4]},
            {"type": "ineq", "fun": lambda v: v[2] + v[3]},
        ]

    def _rescaled(self, s):
        return GJRGARCH(omega=self.omega * s**2, alpha=self.alpha, gamma=self.gamma, beta=self.beta, mu=self.mu * s)


@dataclass(repr=False)
class EGARCH(VolatilityModel):
    """Nelson's (1991) exponential GARCH(1,1).

    .. math::

       \\ln\\sigma_t^2 = \\omega + \\beta\\ln\\sigma_{t-1}^2
       + \\alpha\\bigl(|z_{t-1}| - E|z|\\bigr) + \\gamma z_{t-1},
       \\qquad z_t = \\varepsilon_t / \\sigma_t

    The log form keeps the variance positive with no sign restrictions, and
    :math:`\\gamma < 0` makes negative returns raise volatility more than
    positive ones. Stationary when :math:`|\\beta| < 1`.

    Examples
    --------
    >>> m = EGARCH(omega=-0.2, alpha=0.15, gamma=-0.08, beta=0.98)
    >>> bool(m.news_impact(-0.02, 1e-4) > m.news_impact(0.02, 1e-4))
    True
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    mu: float = 0.0
    param_names = ("mu", "omega", "alpha", "gamma", "beta")

    def news_impact(self, eps, var_prev):
        var_prev = np.asarray(var_prev, dtype=np.float64)
        z = np.asarray(eps) / np.sqrt(var_prev)
        return np.exp(self.omega + self.beta * np.log(var_prev) + self.alpha * (np.abs(z) - _ABS_MEAN) + self.gamma * z)

    def variance(self, returns, v0: Optional[float] = None) -> np.ndarray:
        eps = np.asarray(returns, dtype=np.float64) - self.mu
        out = np.empty_like(eps)
        lv = math.log(_check_v0(eps, v0))
        out[0] = math.exp(lv)
        w, a, g, b, c = self.omega, self.alpha, self.gamma, self.beta, _ABS_MEAN
        for t in range(1, eps.size):
            z = eps[t - 1] / math.sqrt(out[t - 1])
            lv = w + b * lv + a * (abs(z) - c) + g * z
            if lv > 700.0:  # overflow guard for wild trial parameters during fitting
                out[t:] = math.inf
                break
            out[t] = math.exp(lv)
        return out

    @property
    def unconditional_variance(self) -> float:
        """:math:`E[\\sigma_t^2]` for normal innovations (Nelson 1991, Theorem 2.1).

        .. math:: E[\\sigma_t^2] = e^{\\omega/(1-\\beta)} \\prod_{i \\ge 0} E\\bigl[e^{\\beta^i g(z)}\\bigr]

        with :math:`E[e^{a|z| + bz}] = e^{(a+b)^2/2}\\Phi(a+b) + e^{(a-b)^2/2}\\Phi(a-b)`.
        """
        if abs(self.beta) >= 1:
            return math.inf
        log_prod = self.omega / (1 - self.beta)
        i = 0
        while True:
            k = self.beta**i
            a, b = k * self.alpha, k * self.gamma
            m = math.exp((a + b) ** 2 / 2) * stats.norm.cdf(a + b) + math.exp((a - b) ** 2 / 2) * stats.norm.cdf(a - b)
            term = math.log(m) - a * _ABS_MEAN
            log_prod += term
            i += 1
            if abs(term) < 1e-14 or i > 10_000:
                return math.exp(log_prod)

    @classmethod
    def _start(cls, r):
        return np.array([r.mean(), 0.0, 0.15, -0.05, 0.95])

    @classmethod
    def _bounds(cls):
        return [(None, None), (-5.0, 5.0), (-1.0, 3.0), (-1.0, 1.0), (-0.9999, 0.9999)]

    def _rescaled(self, s):
        return EGARCH(omega=self.omega + (1 - self.beta) * math.log(s**2), alpha=self.alpha, gamma=self.gamma, beta=self.beta, mu=self.mu * s)

    def _param_scale(self, name, s):
        return s if name == "mu" else 1.0


@dataclass(repr=False)
class EWMA(VolatilityModel):
    """RiskMetrics (1994) exponentially weighted moving average variance.

    .. math:: \\sigma_t^2 = \\lambda\\,\\sigma_{t-1}^2 + (1 - \\lambda)\\,r_{t-1}^2

    an integrated GARCH(1,1) with :math:`\\omega = 0` and zero mean.
    RiskMetrics fixed :math:`\\lambda = 0.94` for daily and :math:`0.97`
    for monthly data across all assets; :meth:`fit` estimates it instead.
    With no mean reversion, the forecast at every horizon is today's
    variance.

    Parameters
    ----------
    lam : float
        Decay factor in (0, 1); the effective memory is about
        :math:`1/(1-\\lambda)` periods.

    Examples
    --------
    >>> EWMA(0.94).variance([0.01, -0.02, 0.0], v0=1e-4).round(8).tolist()
    [0.0001, 0.0001, 0.000118]
    """

    lam: float = 0.94
    param_names = ("lam",)

    def news_impact(self, eps, var_prev):
        return self.lam * np.asarray(var_prev) + (1 - self.lam) * np.asarray(eps) ** 2

    def variance(self, returns, v0: Optional[float] = None) -> np.ndarray:
        r = np.asarray(returns, dtype=np.float64)
        v0 = _check_v0(r, v0)
        rest = lfilter([1 - self.lam], [1.0, -self.lam], r[:-1] ** 2, zi=[self.lam * v0])[0]
        return np.r_[v0, rest]

    @property
    def unconditional_variance(self) -> float:
        return math.inf

    def forecast(self, returns, h: int) -> np.ndarray:
        """Flat variance forecast: :math:`E_T[\\sigma_{T+k}^2] = \\sigma_{T+1}^2` for every `k`."""
        r = np.asarray(returns, dtype=np.float64)
        v_next = float(self.news_impact(r[-1], self.variance(r)[-1]))
        return np.full(h, v_next)

    @classmethod
    def _start(cls, r):
        return np.array([0.94])

    @classmethod
    def _bounds(cls):
        return [(0.5, 0.9999)]

    def _rescaled(self, s):
        return EWMA(lam=self.lam)


def arch_lm_test(returns, lags: int = 5, demean: bool = True) -> TestResult:
    """Engle's (1982) Lagrange-multiplier test for ARCH effects.

    Regresses :math:`\\varepsilon_t^2` on a constant and
    :math:`\\varepsilon_{t-1}^2, \\dots, \\varepsilon_{t-q}^2`; under the
    null of no ARCH, :math:`T R^2 \\sim \\chi^2_q`.

    Parameters
    ----------
    returns : array_like
    lags : int
        :math:`q`.
    demean : bool

    Returns
    -------
    TestResult

    Examples
    --------
    >>> import numpy as np
    >>> arch_lm_test(np.random.default_rng(0).standard_normal(2000)).rejects()
    False
    >>> arch_lm_test(GARCH(0.05, 0.15, 0.8).simulate(2000, rng=0).returns).rejects()
    True
    """
    e = np.asarray(returns, dtype=np.float64)
    e = e - e.mean() if demean else e
    e2 = e**2
    X = np.column_stack([np.ones(e2.size - lags), lag_matrix(e2, lags)])
    res = ols(e2[lags:], X)
    T = e2.size - lags
    stat = T * res.r_squared
    return TestResult(name="ARCH-LM", statistic=float(stat), p_value=float(stats.chi2.sf(stat, lags)), df=lags, n_obs=T)

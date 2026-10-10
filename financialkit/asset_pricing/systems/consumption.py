"""Consumption-based asset pricing: Lucas's tree, Breeden's consumption
CAPM and Mehra and Prescott's equity premium puzzle.

Lucas (1978) priced assets in an exchange economy in which a
representative agent with power utility eats the fruit of a tree.
Equilibrium prices make the agent content to hold the tree, which gives
the Euler equation :math:`E_t[m_{t+1} R_{t+1}] = 1` with the stochastic
discount factor :math:`m_{t+1} = \\beta (C_{t+1}/C_t)^{-\\gamma}`.
Breeden (1979) collapsed Merton's intertemporal CAPM into a single
consumption beta: expected excess returns are proportional to the
covariance of returns with consumption growth. Mehra and Prescott
(1985) calibrated a two-state version of Lucas's economy to a century
of US consumption data and found that no reasonable risk aversion
produced more than a fraction of a percent of equity premium, against
about six percent in the data.

All returns in this module are gross.

References: R. E. Lucas Jr., "Asset Prices in an Exchange Economy,"
*Econometrica* 46 (1978) 1429-1445; D. T. Breeden, "An Intertemporal
Asset Pricing Model with Stochastic Consumption and Investment
Opportunities," *J. Financ. Econ.* 7 (1979) 265-296; R. Mehra and
E. C. Prescott, "The Equity Premium: A Puzzle," *J. Monet. Econ.* 15
(1985) 145-161; J. H. Cochrane, *Asset Pricing*, rev. ed., Princeton
(2005), ch. 1-2 and 21.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.asset_pricing.core.base import ConsumptionEconomy, ConsumptionMoments, EconomyPath

__all__ = ["LucasTree", "MehraPrescott", "ccapm_expected_excess_return", "implied_risk_aversion"]


class LucasTree(ConsumptionEconomy):
    """Lucas's tree with i.i.d. lognormal growth: every price in closed form.

    Log consumption growth is :math:`g \\sim N(\\mu, \\sigma^2)`, and the
    tree's dividend is consumption. The price-dividend ratio is constant,
    :math:`P/D = \\kappa / (1 - \\kappa)` with
    :math:`\\kappa = \\beta e^{(1-\\gamma)\\mu + (1-\\gamma)^2\\sigma^2/2}`,
    the riskless rate is :math:`\\log R_f = -\\log\\beta + \\gamma\\mu - \\gamma^2\\sigma^2/2`,
    and the log equity premium is :math:`\\log E[R_e] - \\log R_f = \\gamma\\sigma^2`.

    Parameters
    ----------
    beta : float
        Time-discount factor.
    gamma : float
        Relative risk aversion.
    mu : float
        Mean of log consumption growth per period.
    sigma : float
        Volatility of log consumption growth.

    Examples
    --------
    >>> m = LucasTree(beta=0.99, gamma=2.0, mu=0.02, sigma=0.036).moments()
    >>> round(m.risk_free - 1, 4), round(m.equity_premium, 4)
    (0.0486, 0.0027)
    """

    def __init__(self, beta: float, gamma: float, mu: float = 0.018, sigma: float = 0.036):
        super().__init__(beta, gamma)
        self.mu = float(mu)
        self.sigma = float(sigma)
        if self.kappa >= 1.0:
            raise ValueError("beta E[G^(1-gamma)] must be below 1, or the tree has infinite value")

    @property
    def kappa(self) -> float:
        """float: :math:`\\beta E[G^{1-\\gamma}]`, the discount applied to next period's dividend."""
        a = 1.0 - self.gamma
        return self.beta * float(np.exp(a * self.mu + 0.5 * a * a * self.sigma**2))

    def moments(self) -> ConsumptionMoments:
        g, mu, s2 = self.gamma, self.mu, self.sigma**2
        k = self.kappa
        sdf_mean = self.beta * float(np.exp(-g * mu + 0.5 * g * g * s2))
        mean_G = float(np.exp(mu + 0.5 * s2))
        return ConsumptionMoments(
            price_dividend=k / (1.0 - k),
            risk_free=1.0 / sdf_mean,
            equity_return=mean_G / k,
            equity_vol=mean_G / k * float(np.sqrt(np.expm1(s2))),
            sdf_mean=sdf_mean,
            sdf_vol=sdf_mean * float(np.sqrt(np.expm1(g * g * s2))),
        )

    def simulate(self, n_periods: int, rng: np.random.Generator | int | None = None) -> EconomyPath:
        rng = np.random.default_rng(rng)
        G = np.exp(self.mu + self.sigma * rng.standard_normal(n_periods))
        return EconomyPath(
            growth=G,
            sdf=self.beta * G**-self.gamma,
            equity_return=G / self.kappa,
            risk_free=np.full(n_periods, self.moments().risk_free),
        )


class MehraPrescott(ConsumptionEconomy):
    """Mehra and Prescott's two-state Markov economy.

    Gross consumption growth is :math:`\\lambda_1 = 1 + \\mu + \\delta` or
    :math:`\\lambda_2 = 1 + \\mu - \\delta`, and the chain stays in its
    state with probability :math:`\\phi`. The price-dividend ratios
    :math:`w_i` solve :math:`w_i = \\beta \\sum_j \\phi_{ij} \\lambda_j^{1-\\gamma}(w_j + 1)`,
    and the riskless return in state :math:`i` is
    :math:`1 / (\\beta \\sum_j \\phi_{ij} \\lambda_j^{-\\gamma})`.
    The defaults are their 1889-1978 calibration: mean growth 1.8%,
    standard deviation 3.6% and first-order autocorrelation -0.14.

    Parameters
    ----------
    beta, gamma : float
    mu : float
        Mean consumption growth.
    delta : float
        Half the gap between the two growth states (their standard deviation).
    phi : float
        Probability of staying in the current state; the autocorrelation is ``2 phi - 1``.

    Examples
    --------
    >>> m = MehraPrescott(beta=0.99, gamma=10.0).moments()
    >>> round(m.risk_free - 1, 3), round(m.equity_premium, 4)
    (0.131, 0.0269)
    """

    def __init__(self, beta: float, gamma: float, mu: float = 0.018, delta: float = 0.036, phi: float = 0.43):
        super().__init__(beta, gamma)
        if not 0.0 < phi < 1.0:
            raise ValueError(f"phi must lie in (0, 1), got {phi}")
        self.mu, self.delta, self.phi = float(mu), float(delta), float(phi)
        self.growth = np.array([1.0 + mu + delta, 1.0 + mu - delta])
        self.transition = np.array([[phi, 1.0 - phi], [1.0 - phi, phi]])
        self.stationary = np.array([0.5, 0.5])
        A = beta * self.transition * self.growth ** (1.0 - gamma)
        if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
            raise ValueError("the tree has infinite value at these parameters")
        self.price_dividend = np.linalg.solve(np.eye(2) - A, A.sum(axis=1))
        """ndarray, shape (2,): Price-dividend ratio in each state."""
        self.risk_free = 1.0 / (beta * self.transition @ self.growth**-gamma)
        """ndarray, shape (2,): Gross riskless return from each state."""

    @property
    def growth_autocorr(self) -> float:
        """float: First-order autocorrelation of consumption growth, ``2 phi - 1``."""
        return 2.0 * self.phi - 1.0

    def _returns(self) -> np.ndarray:
        w = self.price_dividend
        return self.growth[None, :] * (w[None, :] + 1.0) / w[:, None]

    def moments(self) -> ConsumptionMoments:
        P = self.stationary[:, None] * self.transition
        R = self._returns()
        m = np.broadcast_to(self.beta * self.growth**-self.gamma, P.shape)
        ER = float((P * R).sum())
        Em = float((P * m).sum())
        return ConsumptionMoments(
            price_dividend=float(self.stationary @ self.price_dividend),
            risk_free=float(self.stationary @ self.risk_free),
            equity_return=ER,
            equity_vol=float(np.sqrt((P * (R - ER) ** 2).sum())),
            sdf_mean=Em,
            sdf_vol=float(np.sqrt((P * (m - Em) ** 2).sum())),
        )

    def simulate(self, n_periods: int, rng: np.random.Generator | int | None = None) -> EconomyPath:
        rng = np.random.default_rng(rng)
        u = rng.random(n_periods + 1)
        s = np.empty(n_periods + 1, dtype=np.int64)
        s[0] = int(u[0] < 0.5)
        for t in range(1, n_periods + 1):
            s[t] = s[t - 1] if u[t] < self.phi else 1 - s[t - 1]
        prev, nxt = s[:-1], s[1:]
        G = self.growth[nxt]
        return EconomyPath(
            growth=G,
            sdf=self.beta * G**-self.gamma,
            equity_return=self._returns()[prev, nxt],
            risk_free=self.risk_free[prev],
        )


def ccapm_expected_excess_return(consumption_beta: ArrayLike, consumption_vol: float, gamma: float) -> np.ndarray | float:
    """Breeden's consumption CAPM, :math:`E[R_i] - R_f \\approx \\gamma\\, \\beta_{i,c}\\, \\sigma_c^2`.

    The first-order form of :math:`E[R_i] - R_f = -R_f \\mathrm{Cov}(m, R_i)`
    with :math:`m \\approx \\beta(1 - \\gamma \\Delta c)`: the premium is
    risk aversion times the covariance of the return with consumption
    growth. It is exact for log returns in :class:`LucasTree`.

    Parameters
    ----------
    consumption_beta : array_like
        :math:`\\mathrm{Cov}(R_i, \\Delta c) / \\mathrm{Var}(\\Delta c)`, for
        example from :func:`~financialkit.asset_pricing.systems.capm.estimate_beta`.
    consumption_vol : float
        Volatility of consumption growth.
    gamma : float
        Relative risk aversion.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(ccapm_expected_excess_return(2.0, 0.02, 5.0), 12)
    0.004
    """
    out = gamma * np.asarray(consumption_beta, dtype=np.float64) * consumption_vol**2
    return float(out) if out.ndim == 0 else out


def implied_risk_aversion(equity_premium: float, equity_vol: float, consumption_vol: float, correlation: float = 1.0) -> float:
    """The risk aversion that the consumption CAPM needs to explain a premium.

    :math:`\\gamma = \\dfrac{E[R_e] - R_f}{\\rho\\, \\sigma_e\\, \\sigma_c}`:
    the Sharpe ratio divided by the volatility of consumption growth,
    scaled up by the imperfect correlation between the two.

    Parameters
    ----------
    equity_premium, equity_vol : float
    consumption_vol : float
    correlation : float
        Correlation of equity returns with consumption growth.

    Returns
    -------
    float

    Examples
    --------
    Mehra and Prescott's sample (premium 6.18%, equity volatility 16.54%,
    consumption volatility 3.57%), with perfect correlation:

    >>> round(implied_risk_aversion(0.0618, 0.1654, 0.0357), 1)
    10.5
    """
    return float(equity_premium / (correlation * equity_vol * consumption_vol))

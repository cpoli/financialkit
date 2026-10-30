"""Consumption-based asset pricing: Lucas's tree, Breeden's consumption
CAPM, Hansen and Singleton's estimated Euler equation, Mehra and
Prescott's equity premium puzzle, and two resolutions, Campbell and
Cochrane's habit and Bansal and Yaron's long-run risk.

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
about six percent in the data. Hansen and Singleton (1982) had already
estimated the Euler equation itself by GMM, without solving the model.

Campbell and Cochrane (1999) kept power utility but applied it to
consumption above a slow-moving habit: near the habit the agent is
very risk averse, so the premium is large and countercyclical while the
riskless rate stays constant. Bansal and Yaron (2004) kept consumption
growth almost i.i.d. but added a small, persistent predictable
component and stochastic volatility; with Epstein-Zin preferences and
an elasticity of intertemporal substitution above one, news about
long-run growth carries a large price.

All returns in this module are gross.

References: R. E. Lucas Jr., "Asset Prices in an Exchange Economy,"
*Econometrica* 46 (1978) 1429-1445; D. T. Breeden, "An Intertemporal
Asset Pricing Model with Stochastic Consumption and Investment
Opportunities," *J. Financ. Econ.* 7 (1979) 265-296; L. P. Hansen and
K. J. Singleton, "Generalized Instrumental Variables Estimation of
Nonlinear Rational Expectations Models," *Econometrica* 50 (1982)
1269-1286; R. Mehra and E. C. Prescott, "The Equity Premium: A Puzzle,"
*J. Monet. Econ.* 15 (1985) 145-161; J. Y. Campbell and J. H. Cochrane,
"By Force of Habit: A Consumption-Based Explanation of Aggregate Stock
Market Behavior," *J. Polit. Econ.* 107 (1999) 205-251; R. Bansal and
A. Yaron, "Risks for the Long Run: A Potential Resolution of Asset
Pricing Puzzles," *J. Finance* 59 (2004) 1481-1509; J. H. Cochrane,
*Asset Pricing*, rev. ed., Princeton (2005), ch. 1-2 and 21.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import lfilter

from financialkit.asset_pricing.core.base import ConsumptionEconomy, ConsumptionMoments, EconomyPath, GMMResult
from financialkit.asset_pricing.systems.gmm import gmm
from financialkit.asset_pricing.utils.sorts import as_panel
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER

__all__ = [
    "LucasTree",
    "MehraPrescott",
    "CampbellCochrane",
    "BansalYaron",
    "ccapm_expected_excess_return",
    "implied_risk_aversion",
    "hansen_singleton",
]


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


def _annual_moments(path: EconomyPath, price_dividend: float, periods_per_year: int) -> ConsumptionMoments:
    """Moments of a simulated path after compounding returns and discount factors to years."""
    n = path.sdf.size // periods_per_year * periods_per_year

    def years(x: np.ndarray) -> np.ndarray:
        return x[:n].reshape(-1, periods_per_year).prod(axis=1)

    R, Rf, m = years(path.equity_return), years(path.risk_free), years(path.sdf)
    return ConsumptionMoments(
        price_dividend=price_dividend,
        risk_free=float(Rf.mean()),
        equity_return=float(R.mean()),
        equity_vol=float(R.std()),
        sdf_mean=float(m.mean()),
        sdf_vol=float(m.std()),
    )


class CampbellCochrane(ConsumptionEconomy):
    """Campbell and Cochrane's habit-formation economy, solved on a grid and simulated period by period.

    Log consumption growth is i.i.d. :math:`N(g, \\sigma^2)`, and the agent
    has power utility over consumption above a habit :math:`X`. The log
    surplus ratio :math:`s = \\log((C - X)/C)` follows

    .. math::

       s_{t+1} = (1 - \\phi)\\bar s + \\phi s_t + \\lambda(s_t)(\\Delta c_{t+1} - g),
       \\qquad \\lambda(s) = \\frac{\\sqrt{1 - 2(s - \\bar s)}}{\\bar S} - 1,

    with :math:`\\bar S = \\sigma\\sqrt{\\gamma / (1 - \\phi)}` and
    :math:`\\lambda = 0` above :math:`s_{\\max} = \\bar s + (1 - \\bar S^2)/2`.
    This sensitivity keeps the riskless rate constant. The discount
    factor is :math:`m_{t+1} = \\beta e^{-\\gamma(\\Delta c_{t+1} + \\Delta s_{t+1})}`,
    and the price-consumption ratio solves the linear equation
    :math:`P/C(s) = E_s[m\\,G\\,(1 + P/C(s'))]` on a grid in :math:`s`,
    with Gauss-Hermite quadrature over the shock and linear
    interpolation between the nodes. The consumption claim is the
    equity.

    Parameters
    ----------
    gamma : float
        Curvature of utility over surplus consumption.
    mu, sigma : float
        Annual mean and volatility of log consumption growth.
    phi : float
        Annual persistence of the surplus ratio.
    risk_free : float
        Annual log riskless rate; it sets :math:`\\beta`.
    periods_per_year : int
        Model steps per year; Campbell and Cochrane simulate monthly.
    n_grid : int
        Grid points in :math:`s` below :math:`s_{\\max}`.

    Notes
    -----
    The defaults are Campbell and Cochrane's calibration (their Table 1):
    :math:`g = 1.89\\%`, :math:`\\sigma = 1.50\\%`, :math:`r_f = 0.94\\%`,
    :math:`\\phi = 0.87` and :math:`\\gamma = 2`. ``beta``, ``mu``,
    ``sigma`` and ``phi`` are stored per period; the methods report
    annual figures.

    Examples
    --------
    >>> cc = CampbellCochrane()
    >>> round(cc.S_bar, 3), round(float(np.exp(cc.s_max)), 3)
    (0.057, 0.094)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        mu: float = 0.0189,
        sigma: float = 0.015,
        phi: float = 0.87,
        risk_free: float = 0.0094,
        periods_per_year: int = 12,
        n_grid: int = 400,
    ):
        if not 0.0 < phi < 1.0:
            raise ValueError(f"phi must lie in (0, 1), got {phi}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        h = int(periods_per_year)
        self.periods_per_year = h
        self.mu = mu / h
        self.sigma = float(sigma / np.sqrt(h))
        self.phi = phi ** (1.0 / h)
        self.risk_free_rate = float(risk_free)
        """float: Annual log riskless rate, constant while :math:`s \\le s_{\\max}`."""
        super().__init__(float(np.exp(gamma * self.mu - 0.5 * gamma * (1.0 - self.phi) - risk_free / h)), gamma)
        self.S_bar = self.sigma * float(np.sqrt(gamma / (1.0 - self.phi)))
        """float: Steady-state surplus consumption ratio."""
        self.s_bar = float(np.log(self.S_bar))
        self.s_max = self.s_bar + 0.5 * (1.0 - self.S_bar**2)
        """float: Largest log surplus ratio at which the sensitivity is positive."""
        lower = np.linspace(0.0, 1.0, n_grid) ** 2 * -(10.0 + self.s_max - self.s_bar) + self.s_max
        self.grid = np.unique(np.concatenate([lower, np.linspace(self.s_max, self.s_max + 0.3, 20)]))
        """ndarray: Nodes in the log surplus ratio."""
        self._pc, self._rf = self._solve()

    def sensitivity(self, s: ArrayLike) -> np.ndarray:
        """The sensitivity :math:`\\lambda(s)` of the surplus ratio to consumption shocks.

        Parameters
        ----------
        s : array_like
            Log surplus consumption ratio.

        Returns
        -------
        ndarray
        """
        s = np.asarray(s, dtype=np.float64)
        lam = np.sqrt(np.maximum(1.0 - 2.0 * (s - self.s_bar), 0.0)) / self.S_bar - 1.0
        return np.where(s <= self.s_max, lam, 0.0)

    def _next(self, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (1.0 - self.phi) * self.s_bar + self.phi * s + self.sensitivity(s) * v

    def _sdf(self, s: np.ndarray, s_next: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.beta * np.exp(-self.gamma * (self.mu + v + s_next - s))

    def _solve(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.grid
        n = s.size
        x, w = np.polynomial.hermite_e.hermegauss(30)
        w = w / w.sum()
        v = self.sigma * x[None, :]
        sn = self._next(s[:, None], v)
        rf = 1.0 / (w * self._sdf(s[:, None], sn, v)).sum(axis=1)
        # The claim is priced with s' reflected at the grid's ends, which the
        # simulated economy practically never reaches.
        sn = np.clip(sn, s[0], s[-1])
        K = w * self._sdf(s[:, None], sn, v) * np.exp(self.mu + v)
        k = np.clip(np.searchsorted(s, sn) - 1, 0, n - 2)
        frac = (sn - s[k]) / (s[k + 1] - s[k])
        rows = np.broadcast_to(np.arange(n)[:, None], k.shape)
        A = np.zeros((n, n))
        np.add.at(A, (rows, k), K * (1.0 - frac))
        np.add.at(A, (rows, k + 1), K * frac)
        return np.linalg.solve(np.eye(n) - A, K.sum(axis=1)), rf

    def price_consumption(self, s: ArrayLike) -> np.ndarray:
        """Price of the consumption claim over a year's consumption, at log surplus ratio ``s``.

        Parameters
        ----------
        s : array_like

        Returns
        -------
        ndarray
        """
        return np.interp(np.asarray(s, dtype=np.float64), self.grid, self._pc) / self.periods_per_year

    def max_sharpe_ratio(self, s: ArrayLike) -> np.ndarray:
        """Annualised conditional :math:`\\sigma_t(m)/E_t(m) = \\sqrt{e^{\\gamma^2\\sigma^2(1 + \\lambda(s))^2} - 1}`.

        It is the highest Sharpe ratio any asset can offer, and it
        rises as the surplus ratio falls: risk premia are countercyclical.

        Parameters
        ----------
        s : array_like

        Returns
        -------
        ndarray
        """
        b = self.gamma * self.sigma * (1.0 + self.sensitivity(s))
        return np.sqrt(np.expm1(b * b) * self.periods_per_year)

    def surplus_path(self, n_periods: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Log surplus ratios ``s_0 .. s_n`` from the steady state; the same ``rng`` gives :meth:`simulate`'s path.

        Parameters
        ----------
        n_periods : int
        rng : numpy.random.Generator or int or None

        Returns
        -------
        ndarray, shape (n_periods + 1,)
        """
        return self._simulate(n_periods, rng)[0]

    def _simulate(self, n_periods: int, rng: np.random.Generator | int | None) -> tuple[np.ndarray, np.ndarray]:
        v = self.sigma * np.random.default_rng(rng).standard_normal(n_periods)
        s = np.empty(n_periods + 1)
        s[0] = self.s_bar
        a, b, sb, smax, Sb = 1.0 - self.phi, self.phi, self.s_bar, self.s_max, self.S_bar
        for t in range(n_periods):
            st = s[t]
            lam = np.sqrt(max(1.0 - 2.0 * (st - sb), 0.0)) / Sb - 1.0 if st <= smax else 0.0
            s[t + 1] = a * sb + b * st + lam * v[t]
        return s, v

    def simulate(self, n_periods: int, rng: np.random.Generator | int | None = None) -> EconomyPath:
        s, v = self._simulate(n_periods, rng)
        G = np.exp(self.mu + v)
        pc = np.interp(s, self.grid, self._pc)
        return EconomyPath(
            growth=G,
            sdf=self._sdf(s[:-1], s[1:], v),
            equity_return=G * (1.0 + pc[1:]) / pc[:-1],
            risk_free=np.interp(s[:-1], self.grid, self._rf),
        )

    def moments(self, n_years: int = 5000, rng: np.random.Generator | int | None = 0) -> ConsumptionMoments:
        """Annual moments of a long simulation, with monthly returns compounded within each year.

        Parameters
        ----------
        n_years : int
        rng : numpy.random.Generator or int or None
            Seeded by default, so the moments are reproducible.

        Returns
        -------
        ConsumptionMoments
        """
        n = n_years * self.periods_per_year
        s, _ = self._simulate(n, rng)
        path = self.simulate(n, rng)
        return _annual_moments(path, float(self.price_consumption(s).mean()), self.periods_per_year)


class BansalYaron(ConsumptionEconomy):
    """Bansal and Yaron's long-run risk economy, solved by Campbell-Shiller log-linearisation.

    Monthly log consumption and dividend growth share a small persistent
    component :math:`x_t`, and both have stochastic volatility:

    .. math::

       g_{t+1} &= \\mu + x_t + \\sigma_t \\eta_{t+1}, \\qquad
       x_{t+1} = \\rho x_t + \\varphi_e \\sigma_t e_{t+1}, \\\\
       \\sigma^2_{t+1} &= \\bar\\sigma^2 + \\nu(\\sigma^2_t - \\bar\\sigma^2) + \\sigma_w w_{t+1}, \\qquad
       g_{d,t+1} = \\mu_d + \\Phi x_t + \\varphi_d \\sigma_t u_{t+1}.

    The agent has Epstein-Zin preferences with risk aversion
    :math:`\\gamma` and elasticity of intertemporal substitution
    :math:`\\psi`, so :math:`m_{t+1} = \\theta\\log\\delta - \\frac{\\theta}{\\psi} g_{t+1} + (\\theta - 1) r_{a,t+1}`
    with :math:`\\theta = (1-\\gamma)/(1 - 1/\\psi)`. The log
    price-consumption ratio is :math:`A_0 + A_1 x_t + A_2 \\sigma^2_t` and
    the log price-dividend ratio :math:`A_{0,m} + A_{1,m} x_t + A_{2,m}\\sigma^2_t`,
    with the linearisation constants solved as a fixed point. The equity
    premium is

    .. math::

       E_t[r_{m,t+1} - r_{f,t}] = \\beta_{m,e}\\lambda_{e}\\sigma_t^2 + \\beta_{m,w}\\lambda_{w}\\sigma_w^2 - \\tfrac12 \\mathrm{Var}_t(r_{m,t+1}).

    Parameters
    ----------
    beta : float
        Monthly time-discount factor :math:`\\delta`.
    gamma : float
        Relative risk aversion.
    psi : float
        Elasticity of intertemporal substitution, not 1.
    mu, rho, phi_e, sigma, nu, sigma_w : float
        Consumption growth: mean, persistence and loading of the long-run
        component, mean volatility, persistence and volatility of the variance.
    mu_d, Phi, phi_d : float
        Dividend growth: mean, leverage on the long-run component, volatility scale.

    Notes
    -----
    The defaults are Bansal and Yaron's monthly calibration (their
    Table I, with :math:`\\gamma = 10` and :math:`\\psi = 1.5`). Simulated
    variances are truncated at zero, as in their paper.

    Examples
    --------
    >>> by = BansalYaron()
    >>> round(by.A1, 1), round(12 * by.equity_premium(), 3)
    (14.6, 0.042)
    """

    def __init__(
        self,
        beta: float = 0.998,
        gamma: float = 10.0,
        psi: float = 1.5,
        mu: float = 0.0015,
        rho: float = 0.979,
        phi_e: float = 0.044,
        sigma: float = 0.0078,
        nu: float = 0.987,
        sigma_w: float = 0.23e-5,
        mu_d: float = 0.0015,
        Phi: float = 3.0,
        phi_d: float = 4.5,
    ):
        super().__init__(beta, gamma)
        if psi <= 0.0 or psi == 1.0:
            raise ValueError(f"psi must be positive and not 1, got {psi}")
        if not (abs(rho) < 1.0 and abs(nu) < 1.0):
            raise ValueError("rho and nu must lie in (-1, 1)")
        self.psi, self.mu, self.rho, self.phi_e, self.sigma = float(psi), float(mu), float(rho), float(phi_e), float(sigma)
        self.nu, self.sigma_w, self.mu_d, self.Phi, self.phi_d = float(nu), float(sigma_w), float(mu_d), float(Phi), float(phi_d)
        self.theta = (1.0 - self.gamma) / (1.0 - 1.0 / self.psi)
        self._solve()

    @staticmethod
    def _kappas(z: float) -> tuple[float, float]:
        k1 = np.exp(z) / (1.0 + np.exp(z))
        return np.log1p(np.exp(z)) - k1 * z, k1

    @staticmethod
    def _fixed_point(update, z: float = 5.0) -> float:
        for _ in range(DEFAULT_MAX_ITER):
            with np.errstate(all="ignore"):
                new = update(z)
            if not np.isfinite(new):
                break
            if abs(new - z) < DEFAULT_ATOL:
                return float(new)
            z = new
        raise ValueError("no stationary price-dividend ratio at these parameters")

    def _solve(self) -> None:
        g, psi, th, s2, sw = self.gamma, self.psi, self.theta, self.sigma**2, self.sigma_w**2
        ld = float(np.log(self.beta))

        def wealth(z: float) -> float:
            k0, k1 = self._kappas(z)
            self.A1 = (1.0 - 1.0 / psi) / (1.0 - k1 * self.rho)
            self.A2 = 0.5 * th * ((1.0 - 1.0 / psi) ** 2 + (k1 * self.A1 * self.phi_e) ** 2) / (1.0 - k1 * self.nu)
            self.A0 = (ld + k0 + (1.0 - 1.0 / psi) * self.mu + k1 * self.A2 * s2 * (1.0 - self.nu) + 0.5 * th * (k1 * self.A2) ** 2 * sw) / (1.0 - k1)
            return self.A0 + self.A2 * s2

        self.z_bar = self._fixed_point(wealth)
        """float: Mean log price-consumption ratio (monthly consumption)."""
        k0, k1 = self._kappas(self.z_bar)
        self.kappa1 = k1
        self.lambda_e = (1.0 - th) * k1 * self.A1 * self.phi_e
        """float: Price of long-run growth risk."""
        self.lambda_w = (1.0 - th) * k1 * self.A2
        """float: Price of volatility risk."""
        a0 = k0 + k1 * self.A0 + k1 * self.A2 * s2 * (1.0 - self.nu) - self.A0 + self.mu
        a2 = k1 * self.A2 * self.nu - self.A2
        mean_m = th * ld - th / psi * self.mu + (th - 1.0) * a0
        self._rf0 = -mean_m - 0.5 * self.lambda_w**2 * sw
        self._rf2 = -(th - 1.0) * a2 - 0.5 * (g * g + self.lambda_e**2)

        def market(z: float) -> float:
            k0m, k1m = self._kappas(z)
            self.A1m = (self.Phi - 1.0 / psi) / (1.0 - k1m * self.rho)
            be = k1m * self.A1m * self.phi_e
            H = g * g + (be - self.lambda_e) ** 2 + self.phi_d**2
            self.A2m = ((1.0 - th) * self.A2 * (1.0 - k1 * self.nu) + 0.5 * H) / (1.0 - k1m * self.nu)
            bw = k1m * self.A2m
            self.A0m = (mean_m + k0m + k1m * self.A2m * s2 * (1.0 - self.nu) + self.mu_d + 0.5 * (bw - self.lambda_w) ** 2 * sw) / (1.0 - k1m)
            return self.A0m + self.A2m * s2

        self.z_bar_m = self._fixed_point(market)
        """float: Mean log price-dividend ratio (monthly dividend)."""
        self.kappa1_m = self._kappas(self.z_bar_m)[1]
        self.beta_e = self.kappa1_m * self.A1m * self.phi_e
        """float: Exposure of the market return to long-run growth news."""
        self.beta_w = self.kappa1_m * self.A2m
        """float: Exposure of the market return to volatility news."""
        for name in ("A0", "A1", "A2", "A0m", "A1m", "A2m", "kappa1", "kappa1_m", "lambda_e", "lambda_w", "beta_e", "beta_w", "_rf0", "_rf2"):
            setattr(self, name, float(getattr(self, name)))

    def risk_free_rate(self, x: ArrayLike = 0.0, sigma2: Optional[ArrayLike] = None) -> np.ndarray | float:
        """Monthly log riskless rate in state ``(x, sigma2)``; the mean variance by default.

        Parameters
        ----------
        x : array_like
            Long-run growth component.
        sigma2 : array_like, optional
            Conditional variance.

        Returns
        -------
        float or ndarray
        """
        s2 = self.sigma**2 if sigma2 is None else np.asarray(sigma2, dtype=np.float64)
        out = self._rf0 + np.asarray(x, dtype=np.float64) / self.psi + self._rf2 * s2
        return float(out) if np.ndim(out) == 0 else out

    def equity_premium(self, sigma2: Optional[ArrayLike] = None) -> np.ndarray | float:
        """Monthly expected log excess return of the dividend claim, :math:`E_t[r_m - r_f]`.

        Parameters
        ----------
        sigma2 : array_like, optional
            Conditional variance; the mean variance by default.

        Returns
        -------
        float or ndarray
        """
        s2 = self.sigma**2 if sigma2 is None else np.asarray(sigma2, dtype=np.float64)
        sw = self.sigma_w**2
        var = (self.beta_e**2 + self.phi_d**2) * s2 + self.beta_w**2 * sw
        out = self.beta_e * self.lambda_e * s2 + self.beta_w * self.lambda_w * sw - 0.5 * var
        return float(out) if np.ndim(out) == 0 else out

    def _states(self, n_periods: int, rng: np.random.Generator | int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eta, e, w, u = np.random.default_rng(rng).standard_normal((4, n_periods))
        s2 = self.sigma**2 + lfilter([1.0], [1.0, -self.nu], np.concatenate([[0.0], self.sigma_w * w]))
        sd = np.sqrt(np.maximum(s2, 0.0))
        x = lfilter([1.0], [1.0, -self.rho], np.concatenate([[0.0], self.phi_e * sd[:-1] * e]))
        return x, np.maximum(s2, 0.0), np.stack([eta, u])

    def simulate(self, n_periods: int, rng: np.random.Generator | int | None = None) -> EconomyPath:
        x, s2, (eta, u) = self._states(n_periods, rng)
        sd = np.sqrt(s2[:-1])
        g = self.mu + x[:-1] + sd * eta
        gd = self.mu_d + self.Phi * x[:-1] + self.phi_d * sd * u
        k0, k1 = self._kappas(self.z_bar)
        k0m, k1m = self._kappas(self.z_bar_m)
        z = self.A0 + self.A1 * x + self.A2 * s2
        zm = self.A0m + self.A1m * x + self.A2m * s2
        ra = k0 + k1 * z[1:] - z[:-1] + g
        rm = k0m + k1m * zm[1:] - zm[:-1] + gd
        m = self.theta * np.log(self.beta) - self.theta / self.psi * g + (self.theta - 1.0) * ra
        return EconomyPath(
            growth=np.exp(g),
            sdf=np.exp(m),
            equity_return=np.exp(rm),
            risk_free=np.exp(self.risk_free_rate(x[:-1], s2[:-1])),
        )

    def moments(self, n_years: int = 5000, rng: np.random.Generator | int | None = 0) -> ConsumptionMoments:
        """Annual moments of a long monthly simulation, with returns compounded within each year.

        Parameters
        ----------
        n_years : int
        rng : numpy.random.Generator or int or None
            Seeded by default, so the moments are reproducible.

        Returns
        -------
        ConsumptionMoments
            ``price_dividend`` is the price over a year's dividends.
        """
        n = 12 * n_years
        x, s2, _ = self._states(n, rng)
        pd = float(np.exp(self.A0m + self.A1m * x + self.A2m * s2).mean()) / 12.0
        return _annual_moments(self.simulate(n, rng), pd, 12)


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


def hansen_singleton(
    growth: ArrayLike,
    returns: ArrayLike,
    instruments: Optional[ArrayLike] = None,
    theta0: ArrayLike = (0.99, 2.0),
    weighting: str | ArrayLike = "two-step",
    lags: int = 0,
) -> GMMResult:
    """Hansen and Singleton's GMM estimate of the power-utility Euler equation.

    The moment conditions are
    :math:`E[(\\beta G_{t+1}^{-\\gamma} R_{i,t+1} - 1)\\, z_{k,t}] = 0` for every
    asset ``i`` and instrument ``k``: the pricing error must be
    unpredictable by anything known when the return begins. No model of
    consumption or returns is needed, only the Euler equation.

    Parameters
    ----------
    growth : array_like, shape (T,)
        Gross consumption growth :math:`C_{t+1}/C_t`.
    returns : array_like, shape (T,) or (T, N)
        Gross returns over the same periods.
    instruments : array_like, shape (T,) or (T, K), optional
        Variables known at the start of each period, aligned with the
        rows of ``growth``; a constant only by default. Include a
        column of ones to keep the unconditional moments.
    theta0 : array_like
        Starting ``(beta, gamma)``.
    weighting : str or array_like
        As in :func:`~financialkit.asset_pricing.systems.gmm.gmm`.
    lags : int
        Newey-West lags.

    Returns
    -------
    GMMResult
        ``params`` is ``(beta, gamma)``; with ``N K > 2`` the J statistic
        tests the overidentifying restrictions.

    Examples
    --------
    Data simulated from a Lucas tree with ``beta = 0.97`` and ``gamma = 4``:

    >>> p = LucasTree(0.97, 4.0, 0.02, 0.1).simulate(5000, rng=0)
    >>> res = hansen_singleton(p.growth, np.column_stack([p.equity_return, p.risk_free]))
    >>> res.params.round(2).tolist()
    [0.97, 4.0]
    """
    G = np.asarray(growth, dtype=np.float64)
    R = as_panel(returns)
    Z = np.ones((G.size, 1)) if instruments is None else as_panel(instruments)
    if R.shape[0] != G.size or Z.shape[0] != G.size:
        raise ValueError("growth, returns and instruments need the same number of rows")

    def moments(theta: np.ndarray) -> np.ndarray:
        e = theta[0] * G[:, None] ** -theta[1] * R - 1.0
        return (e[:, :, None] * Z[:, None, :]).reshape(G.size, -1)

    return gmm(moments, theta0, weighting=weighting, lags=lags)

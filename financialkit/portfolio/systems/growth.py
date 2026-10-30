"""Growth-optimal and dynamic strategies: Kelly (1956), Merton (1969), CPPI (1987) and Garleanu-Pedersen (2013).

Kelly showed that a gambler who bets the fraction
:math:`f^* = p/a - q/b` of wealth on a favourable bet maximizes the
long-run growth rate of wealth, :math:`E\\log(1 + f X)`. Merton solved
the continuous-time version for an investor with constant relative risk
aversion :math:`\\gamma`: hold the constant fraction
:math:`\\pi^* = (\\mu - r)/(\\gamma\\sigma^2)` in the stock, which is
Kelly's rule at :math:`\\gamma = 1`. Black and Jones's constant
proportion portfolio insurance instead holds a multiple :math:`m` of the
cushion above a floor, so the floor is protected under continuous
trading while upside is kept. Garleanu and Pedersen added predictable
returns and quadratic transaction costs to the mean-variance problem:
the optimal position moves a constant fraction of the way towards an
aim portfolio that weights slow signals more than fast ones.

References: J. L. Kelly, Jr., "A New Interpretation of Information
Rate," *Bell System Tech. J.* 35 (1956) 917-926; R. C. Merton,
"Lifetime Portfolio Selection under Uncertainty: The Continuous-Time
Case," *Rev. Econ. Stat.* 51 (1969) 247-257; F. Black and R. Jones,
"Simplifying Portfolio Insurance," *J. Portfolio Mgmt.* 14 (1987) 48-51;
F. Black and A. Perold, "Theory of Constant Proportion Portfolio
Insurance," *J. Econ. Dyn. Control* 16 (1992) 403-426; N. Garleanu and
L. H. Pedersen, "Dynamic Trading with Predictable Returns and Transaction
Costs," *J. Finance* 68 (2013) 2309-2340.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.integrators import gbm_paths
from financialkit.portfolio.core.base import CPPIResult, DynamicTradingPolicy
from financialkit.portfolio.systems.mean_variance import two_fund_allocation

__all__ = [
    "kelly_fraction",
    "kelly_growth_rate",
    "kelly_wealth_paths",
    "kelly_portfolio",
    "merton_fraction",
    "merton_certainty_equivalent",
    "merton_consumption_rate",
    "merton_wealth_paths",
    "cppi",
    "cppi_continuous",
    "garleanu_pedersen",
]


def kelly_fraction(p: ArrayLike, b: ArrayLike = 1.0, a: ArrayLike = 1.0) -> np.ndarray | float:
    """Kelly's growth-optimal fraction of wealth to stake on a binary bet.

    A stake :math:`f` returns :math:`+b f` with probability :math:`p` and
    :math:`-a f` otherwise; :math:`f^* = p/a - (1 - p)/b`. A negative
    value means the bet is unfavourable (take the other side, or abstain).

    Parameters
    ----------
    p : array_like
        Probability of winning.
    b : array_like
        Net odds received on a win, per unit staked.
    a : array_like
        Fraction of the stake lost on a loss.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Kelly's even-money bet with a 60% edge:

    >>> round(kelly_fraction(0.6), 12)
    0.2
    """
    p = np.asarray(p, dtype=np.float64)
    out = p / np.asarray(a, dtype=np.float64) - (1.0 - p) / np.asarray(b, dtype=np.float64)
    return float(out) if out.ndim == 0 else out


def kelly_growth_rate(f: ArrayLike, p: ArrayLike, b: ArrayLike = 1.0, a: ArrayLike = 1.0) -> np.ndarray | float:
    """Expected log-growth per bet, :math:`p\\log(1 + bf) + (1 - p)\\log(1 - af)`.

    Parameters
    ----------
    f : array_like
        Fraction staked, below ``1 / a``.
    p, b, a : array_like
        As in :func:`kelly_fraction`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(kelly_growth_rate(0.2, 0.6), 6)
    0.020136
    """
    f, p = np.asarray(f, dtype=np.float64), np.asarray(p, dtype=np.float64)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p * np.log1p(b * f) + (1.0 - p) * np.log1p(-a * f)
    return float(out) if out.ndim == 0 else out


def kelly_wealth_paths(
    f: float,
    p: float,
    n_bets: int,
    n_paths: int,
    b: float = 1.0,
    a: float = 1.0,
    w0: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Wealth of a gambler who stakes the fixed fraction ``f`` on repeated bets.

    Parameters
    ----------
    f, p, b, a : float
        Fraction staked and the bet, as in :func:`kelly_growth_rate`.
    n_bets, n_paths : int
    w0 : float
        Initial wealth.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    ndarray, shape (n_paths, n_bets + 1)
    """
    wins = np.random.default_rng(rng).random((n_paths, n_bets)) < p
    factors = np.where(wins, 1.0 + b * f, 1.0 - a * f)
    return w0 * np.concatenate([np.ones((n_paths, 1)), np.cumprod(factors, axis=1)], axis=1)


def kelly_portfolio(mu: ArrayLike, cov: ArrayLike, r: float = 0.0) -> np.ndarray:
    """Growth-optimal risky weights for lognormal assets, :math:`\\Sigma^{-1}(\\mu - r)`.

    The continuous-time Kelly portfolio is the logarithmic investor's
    (:math:`\\gamma = 1`) Merton/Tobin allocation; whatever it leaves over
    is held in cash (or borrowed, if negative).

    Parameters
    ----------
    mu : array_like, shape (d,)
        Instantaneous expected returns.
    cov : array_like, shape (d, d)
        Instantaneous covariance.
    r : float
        Riskless rate.

    Returns
    -------
    ndarray, shape (d,)

    Examples
    --------
    >>> kelly_portfolio([0.08], [[0.04]], 0.02)
    array([1.5])
    """
    return two_fund_allocation(mu, cov, r, 1.0).weights


def merton_fraction(mu: ArrayLike, r: ArrayLike, sigma: ArrayLike, gamma: ArrayLike) -> np.ndarray | float:
    """Merton's optimal constant fraction of wealth in the stock, :math:`(\\mu - r)/(\\gamma\\sigma^2)`.

    Parameters
    ----------
    mu : array_like
        Expected return of the stock.
    r : array_like
        Riskless rate.
    sigma : array_like
        Volatility of the stock.
    gamma : array_like
        Relative risk aversion (``1`` is log utility, i.e. Kelly).

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(merton_fraction(0.08, 0.02, 0.2, 3.0), 12)
    0.5
    """
    out = (np.asarray(mu, dtype=np.float64) - np.asarray(r, dtype=np.float64)) / (
        np.asarray(gamma, dtype=np.float64) * np.asarray(sigma, dtype=np.float64) ** 2
    )
    return float(out) if out.ndim == 0 else out


def merton_certainty_equivalent(pi: ArrayLike, mu: float, r: float, sigma: float, gamma: float) -> np.ndarray | float:
    """Certainty-equivalent growth rate of holding the constant fraction ``pi`` in the stock.

    Wealth is then geometric Brownian motion with drift
    :math:`r + \\pi(\\mu - r)` and volatility :math:`\\pi\\sigma`, and a CRRA
    investor values it like a sure return of
    :math:`r + \\pi(\\mu - r) - \\tfrac{\\gamma}{2}\\pi^2\\sigma^2` per year,
    maximized at :func:`merton_fraction`.

    Parameters
    ----------
    pi : array_like
        Fraction of wealth in the stock.
    mu, r, sigma, gamma : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(merton_certainty_equivalent(0.5, 0.08, 0.02, 0.2, 3.0), 6)
    0.035
    """
    pi = np.asarray(pi, dtype=np.float64)
    out = r + pi * (mu - r) - 0.5 * gamma * pi**2 * sigma**2
    return float(out) if out.ndim == 0 else out


def merton_consumption_rate(mu: float, r: float, sigma: float, gamma: float, rho: float) -> float:
    """Merton's optimal consumption-to-wealth ratio over an infinite horizon.

    With time preference :math:`\\rho` and Sharpe ratio
    :math:`\\theta = (\\mu - r)/\\sigma`, the investor consumes the constant
    fraction :math:`\\nu = [\\rho - (1 - \\gamma)(r + \\theta^2 / (2\\gamma))]/\\gamma`
    of wealth per year; it must be positive for the problem to be well posed.

    Parameters
    ----------
    mu, r, sigma, gamma : float
    rho : float
        Subjective discount rate.

    Returns
    -------
    float

    Examples
    --------
    Log utility consumes at the rate of time preference.

    >>> merton_consumption_rate(0.08, 0.02, 0.2, 1.0, 0.03)
    0.03
    """
    theta = (mu - r) / sigma
    nu = (rho - (1.0 - gamma) * (r + theta**2 / (2.0 * gamma))) / gamma
    if nu <= 0:
        raise ValueError("no optimum: the consumption rate is not positive (raise rho or gamma)")
    return float(nu)


def merton_wealth_paths(
    pi: float,
    mu: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    consumption: float = 0.0,
    w0: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact wealth paths of the constant-mix strategy, through :func:`financialkit.integrators.gbm_paths`.

    Parameters
    ----------
    pi : float
        Fraction of wealth held in the stock, rebalanced continuously.
    mu, r, sigma : float
        Stock drift and volatility, riskless rate.
    T : float
        Horizon in years.
    n_steps, n_paths : int
    consumption : float
        Fraction of wealth consumed per year.
    w0 : float
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
    wealth : ndarray, shape (n_paths, n_steps + 1)
    """
    return gbm_paths(w0, r + pi * (mu - r) - consumption, abs(pi) * sigma, T, n_steps, n_paths, rng=rng)


def cppi(
    prices: ArrayLike,
    times: ArrayLike,
    floor: float = 0.8,
    multiplier: float = 3.0,
    r: float = 0.0,
    max_leverage: float = np.inf,
    w0: float = 1.0,
) -> CPPIResult:
    """Constant proportion portfolio insurance rebalanced on a discrete grid.

    At each date the strategy holds :math:`E = \\min(m\\,\\max(W - F, 0), \\ell W)`
    in the risky asset and the rest at the riskless rate, where the floor
    :math:`F_t = F_0 e^{rt}` grows like a zero-coupon bond. Between dates
    the risky asset can fall by more than :math:`1/m`, and wealth then
    drops through the floor: that is gap risk.

    Parameters
    ----------
    prices : array_like, shape (n_paths, n_steps + 1)
        Risky asset paths, e.g. from :func:`financialkit.integrators.gbm_paths`.
    times : array_like, shape (n_steps + 1,)
    floor : float
        Initial floor as a fraction of ``w0``.
    multiplier : float
        Exposure per unit of cushion, :math:`m`.
    r : float
        Riskless rate.
    max_leverage : float
        Cap :math:`\\ell` on exposure as a multiple of wealth (``1`` forbids borrowing).
    w0 : float
        Initial wealth.

    Returns
    -------
    CPPIResult

    Examples
    --------
    >>> import numpy as np
    >>> res = cppi([[100.0, 90.0, 99.0]], [0.0, 0.5, 1.0], floor=0.8, multiplier=2.0)
    >>> np.round(res.wealth, 4)
    array([[1.   , 0.96 , 0.992]])
    """
    S = np.atleast_2d(np.asarray(prices, dtype=np.float64))
    t = np.asarray(times, dtype=np.float64)
    if S.shape[1] != t.size:
        raise ValueError("prices must have one column per entry of times")
    F = floor * w0 * np.exp(r * t)
    growth = np.exp(r * np.diff(t))
    W = np.empty_like(S)
    E = np.empty((S.shape[0], t.size - 1))
    W[:, 0] = w0
    for k in range(t.size - 1):
        E[:, k] = multiplier * np.maximum(W[:, k] - F[k], 0.0)
        if np.isfinite(max_leverage):
            E[:, k] = np.minimum(E[:, k], max_leverage * np.maximum(W[:, k], 0.0))
        W[:, k + 1] = E[:, k] * S[:, k + 1] / S[:, k] + (W[:, k] - E[:, k]) * growth[k]
    return CPPIResult(times=t, prices=S, wealth=W, floor=F, exposure=E, multiplier=float(multiplier))


def cppi_continuous(
    prices: ArrayLike, times: ArrayLike, sigma: float, floor: float = 0.8, multiplier: float = 3.0, r: float = 0.0, w0: float = 1.0
) -> np.ndarray:
    """Black and Perold's closed form for CPPI under continuous rebalancing.

    For geometric Brownian motion the cushion is a power of the price,
    :math:`C_t = C_0 (S_t/S_0)^m \\exp\\left((1 - m)(r + \\tfrac12 m\\sigma^2)t\\right)`,
    so wealth :math:`F_t + C_t` never falls below the floor.

    Parameters
    ----------
    prices : array_like, shape (n_paths, n_steps + 1)
    times : array_like, shape (n_steps + 1,)
    sigma : float
        Volatility of the risky asset.
    floor, multiplier, r, w0 : float
        As in :func:`cppi` (no leverage cap).

    Returns
    -------
    ndarray, shape (n_paths, n_steps + 1)
        Wealth along each path.
    """
    S = np.atleast_2d(np.asarray(prices, dtype=np.float64))
    t = np.asarray(times, dtype=np.float64)
    F = floor * w0 * np.exp(r * t)
    C = (1.0 - floor) * w0 * (S / S[:, :1]) ** multiplier * np.exp((1.0 - multiplier) * (r + 0.5 * multiplier * sigma**2) * t)
    return F + C


def garleanu_pedersen(B: ArrayLike, cov: ArrayLike, Phi: ArrayLike, gamma: float, lam: float, rho: float = 0.0) -> DynamicTradingPolicy:
    """Garleanu and Pedersen's optimal trading policy with predictable returns and transaction costs.

    Excess returns are predicted by factors that mean-revert,
    :math:`r_{t+1} = Bf_t + u_{t+1}` with :math:`\\mathrm{Var}(u) = \\Sigma`
    and :math:`\\Delta f_{t+1} = -\\Phi f_t + \\varepsilon_{t+1}`, and trading
    :math:`\\Delta x_t` costs :math:`\\tfrac12\\Delta x_t'\\Lambda\\Delta x_t` with
    :math:`\\Lambda = \\lambda\\Sigma`. Maximizing the discounted sum of
    mean-variance utility net of costs gives (Garleanu and Pedersen 2013,
    Propositions 2 and 4)

    .. math::

       x_t = \\left(1 - \\frac{a}{\\lambda}\\right)x_{t-1} + \\frac{a}{\\lambda}\\,
       (\\gamma\\Sigma)^{-1}B\\left(I + \\frac{a}{\\gamma}\\Phi\\right)^{-1}f_t,
       \\qquad a = \\frac{-(\\gamma\\bar\\rho + \\lambda\\rho)
       + \\sqrt{(\\gamma\\bar\\rho + \\lambda\\rho)^2 + 4\\gamma\\lambda\\bar\\rho^2}}{2\\bar\\rho},

    with :math:`\\bar\\rho = 1 - \\rho`: trade partially towards an aim
    portfolio that discounts each signal by its speed of decay.

    Parameters
    ----------
    B : array_like, shape (d, k)
        Loadings of next period's excess returns on the predictors.
    cov : array_like, shape (d, d)
        Covariance :math:`\\Sigma` of the return innovations.
    Phi : array_like, shape (k,) or (k, k)
        Mean-reversion speeds of the predictors per period (a vector is a
        diagonal matrix); a half-life of :math:`h` periods is
        :math:`\\phi = 1 - 2^{-1/h}`.
    gamma : float
        Risk aversion.
    lam : float
        Transaction cost per unit of traded variance.
    rho : float
        Discount rate per period, in :math:`[0, 1)`.

    Returns
    -------
    DynamicTradingPolicy

    Examples
    --------
    >>> p = garleanu_pedersen([[1.0]], [[1.0]], [0.5], gamma=1.0, lam=2.0)
    >>> p.trade_rate, np.round(p.aim_loadings, 4)
    (0.5, array([[0.6667]]))
    """
    if gamma <= 0 or lam <= 0:
        raise ValueError(f"gamma and lam must be positive, got {gamma} and {lam}")
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    Bm = np.atleast_2d(np.asarray(B, dtype=np.float64))
    C = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    P = np.asarray(Phi, dtype=np.float64)
    P = np.diag(np.atleast_1d(P)) if P.ndim < 2 else P
    d, k = Bm.shape
    if C.shape != (d, d) or P.shape != (k, k):
        raise ValueError(f"cov must have shape ({d}, {d}) and Phi ({k},) or ({k}, {k}), got {C.shape} and {P.shape}")
    rb = 1.0 - rho
    c = gamma * rb + lam * rho
    # The root above, rationalized so that it does not cancel when lam is small.
    a = 2.0 * gamma * lam * rb / (c + np.sqrt(c**2 + 4.0 * gamma * lam * rb**2))
    markowitz = np.linalg.solve(gamma * C, Bm)
    return DynamicTradingPolicy(
        trade_rate=float(a / lam),
        aim_loadings=markowitz @ np.linalg.inv(np.eye(k) + a / gamma * P),
        markowitz_loadings=markowitz,
        gamma=float(gamma),
        lam=float(lam),
        rho=float(rho),
    )

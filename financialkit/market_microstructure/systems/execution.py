"""Optimal execution: the square-root impact law, Bertsimas-Lo, Almgren-Chriss and Obizhaeva-Wang.

Trading a large order moves the price against the trader. Practitioners
found that the impact of a whole order grows like the square root of its
size relative to daily volume (Loeb, 1983; the BARRA market impact model
of Torre, 1997), not linearly. Dimitris Bertsimas and Andrew Lo (1998)
posed execution as dynamic programming: with linear permanent impact a
risk-neutral trader should split the order evenly (TWAP), and a
predictive signal tilts the schedule. Robert Almgren and Neil Chriss
(2000) added temporary impact and risk aversion: the efficient
trajectories trade off expected cost against its variance and are
hyperbolic sines, from TWAP to immediate liquidation. Anna Obizhaeva
and Jiang Wang (2013) derived impact from the order book itself: a trade
eats into the book, which then refills at a finite rate, so the impact
is transient and the optimal schedule mixes blocks with steady trading.

References: T. F. Loeb, "Trading Cost: The Critical Link between
Investment Information and Results," *Financial Analysts J.* 39 (1983)
39-44; N. Torre, *BARRA Market Impact Model Handbook* (BARRA, 1997); D.
Bertsimas and A. W. Lo, "Optimal Control of Execution Costs," *J.
Financial Markets* 1 (1998) 1-50; R. Almgren and N. Chriss, "Optimal
Execution of Portfolio Transactions," *J. Risk* 3 (2000) 5-39; A. A.
Obizhaeva and J. Wang, "Optimal Trading Strategy and Supply/Demand
Dynamics," *J. Financial Markets* 16 (2013) 1-32.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.market_microstructure.core.base import BertsimasLoResult, ExecutionResult, TransientImpactResult
from financialkit.time_series.utils.numerics import OLSResult, ols

__all__ = ["square_root_impact", "fit_impact_law", "bertsimas_lo", "almgren_chriss", "execution_cost", "obizhaeva_wang", "transient_impact_cost"]


def square_root_impact(quantity: ArrayLike, daily_volume: ArrayLike, sigma: ArrayLike, Y: float = 1.0) -> np.ndarray:
    """Square-root law :math:`I = Y \\sigma \\sqrt{Q / V}` for the impact of a metaorder.

    Parameters
    ----------
    quantity : array_like
        Shares in the order, :math:`Q`.
    daily_volume : array_like
        Average daily volume, :math:`V`.
    sigma : array_like
        Daily volatility (relative impact) or its price equivalent.
    Y : float
        Order-one prefactor, measured empirically.

    Returns
    -------
    ndarray

    Examples
    --------
    Trading 1% of daily volume in a 2%-volatility stock costs about 20 bp.

    >>> float(square_root_impact(1e4, 1e6, 0.02))
    0.002
    """
    return Y * np.asarray(sigma, dtype=np.float64) * np.sqrt(np.asarray(quantity, dtype=np.float64) / np.asarray(daily_volume, dtype=np.float64))


def fit_impact_law(participation: ArrayLike, impact: ArrayLike) -> OLSResult:
    """Fit :math:`I = Y q^\\delta` by least squares on logs.

    Parameters
    ----------
    participation : array_like, shape (n,)
        :math:`q = Q / V`, positive.
    impact : array_like, shape (n,)
        Measured impact (divided by volatility), positive.

    Returns
    -------
    OLSResult
        ``coef = [log Y, delta]``; the square-root law says ``delta = 0.5``.

    Examples
    --------
    >>> q = [0.001, 0.01, 0.1]
    >>> fit_impact_law(q, square_root_impact(q, 1.0, 1.0, Y=0.7)).coef.round(6).tolist()
    [-0.356675, 0.5]
    """
    q = np.asarray(participation, dtype=np.float64)
    i = np.asarray(impact, dtype=np.float64)
    if np.any(q <= 0) or np.any(i <= 0):
        raise ValueError("participation and impact must be positive")
    return ols(np.log(i), np.column_stack([np.ones(q.size), np.log(q)]))


def bertsimas_lo(
    shares: float,
    n_periods: int,
    theta: float,
    gamma: float = 0.0,
    rho: float = 0.0,
    sigma_eta: float = 0.0,
    x0: float = 0.0,
    signals: ArrayLike | None = None,
    p0: float = 0.0,
) -> BertsimasLoResult:
    """Bertsimas and Lo's optimal schedule for buying `shares` over `n_periods`.

    The execution price follows
    :math:`P_k = P_{k-1} + \\theta S_k + \\gamma X_k + \\varepsilon_k` with a
    serially correlated signal :math:`X_k = \\rho X_{k-1} + \\eta_k`, and the
    trader minimises :math:`E[\\sum_k P_k S_k]` subject to
    :math:`\\sum_k S_k = \\bar S`. The value function is quadratic, and
    backward induction gives

    .. math::

       \\delta_{w,k} = 1 - \\frac{\\theta}{2a_{k+1}}, \\quad
       \\delta_{x,k} = \\frac{\\rho b_{k+1}}{2a_{k+1}}, \\quad
       a_k = \\theta - \\frac{\\theta^2}{4a_{k+1}}, \\quad
       b_k = \\rho\\left(\\gamma + \\frac{\\theta b_{k+1}}{2a_{k+1}}\\right),

    from :math:`a = \\theta`, :math:`b = \\gamma\\rho` in the last period.
    With :math:`\\gamma = 0` the policy is :math:`S_k = W_k / (T - k)`, an even
    split, and the expected cost is :math:`P_0\\bar S + \\theta \\bar S^2 (T + 1) / (2T)`.

    Parameters
    ----------
    shares : float
        :math:`\\bar S`.
    n_periods : int
        :math:`T`.
    theta : float
        Permanent impact per share.
    gamma, rho, sigma_eta : float
        Signal loading, persistence and innovation volatility.
    x0 : float
        Signal observed before the first trade, :math:`X_{-1}`.
    signals : array_like, shape (n_periods,), optional
        Realised :math:`X_{k-1}` before each trade; the expected path
        :math:`\\rho^k x_0` if omitted.
    p0 : float
        Price before trading.

    Returns
    -------
    BertsimasLoResult

    Examples
    --------
    >>> res = bertsimas_lo(1000.0, 4, theta=1e-4)
    >>> res.trades.round(9).tolist(), round(res.expected_cost, 6)
    ([250.0, 250.0, 250.0, 250.0], 62.5)
    """
    T = int(n_periods)
    if T < 1 or theta <= 0:
        raise ValueError("need n_periods >= 1 and theta > 0")
    a, b, c, d = np.empty(T), np.empty(T), np.empty(T), np.empty(T)
    dw, dx = np.empty(T), np.empty(T)
    a[-1], b[-1], c[-1], d[-1], dw[-1], dx[-1] = theta, gamma * rho, 0.0, 0.0, 1.0, 0.0
    for k in range(T - 2, -1, -1):
        an, bn, cn = a[k + 1], b[k + 1], c[k + 1]
        dw[k] = 1.0 - theta / (2.0 * an)
        dx[k] = rho * bn / (2.0 * an)
        a[k] = theta - theta**2 / (4.0 * an)
        b[k] = rho * (gamma + theta * bn / (2.0 * an))
        c[k] = rho**2 * (cn - bn**2 / (4.0 * an))
        d[k] = d[k + 1] + cn * sigma_eta**2
    sig = x0 * rho ** np.arange(T) if signals is None else np.asarray(signals, dtype=np.float64)
    if sig.shape != (T,):
        raise ValueError("signals must have shape (n_periods,)")
    trades = np.empty(T)
    W = shares
    for k in range(T):
        trades[k] = dw[k] * W + dx[k] * sig[k]
        W -= trades[k]
    cost = p0 * shares + a[0] * shares**2 + b[0] * x0 * shares + c[0] * x0**2 + d[0]
    return BertsimasLoResult(delta_w=dw, delta_x=dx, a=a, b=b, c=c, d=d, trades=trades, expected_cost=float(cost))


def _temp_eta(eta: float, gamma: float, tau: float) -> float:
    eta_t = eta - 0.5 * gamma * tau
    if eta_t <= 0:
        raise ValueError("need eta > gamma * tau / 2")
    return eta_t


def execution_cost(holdings: ArrayLike, T: float, sigma: float, eta: float, gamma: float = 0.0, epsilon: float = 0.0) -> tuple[float, float]:
    """Expected shortfall and variance of any schedule in the Almgren-Chriss model.

    .. math::

       E = \\tfrac12\\gamma X^2 + \\varepsilon\\sum|n_k| + \\frac{\\tilde\\eta}{\\tau}\\sum n_k^2, \\qquad
       V = \\sigma^2\\tau\\sum_{k=1}^N x_k^2, \\qquad \\tilde\\eta = \\eta - \\tfrac12\\gamma\\tau.

    Parameters
    ----------
    holdings : array_like, shape (N + 1,)
        :math:`x_0 = X, \\dots, x_N = 0` on an even grid over ``[0, T]``.
    T, sigma, eta, gamma, epsilon : float
        Horizon, price volatility, temporary and permanent impact, and the
        fixed half-spread cost.

    Returns
    -------
    expected_cost, variance : float

    Examples
    --------
    >>> execution_cost([10.0, 5.0, 0.0], T=2.0, sigma=1.0, eta=0.1)
    (5.0, 25.0)
    """
    x = np.asarray(holdings, dtype=np.float64)
    tau = T / (x.size - 1)
    n = -np.diff(x)
    E = 0.5 * gamma * x[0] ** 2 + epsilon * np.abs(n).sum() + _temp_eta(eta, gamma, tau) / tau * (n @ n)
    return float(E), float(sigma**2 * tau * (x[1:] @ x[1:]))


def almgren_chriss(
    X: float, T: float, n_steps: int, sigma: float, eta: float, risk_aversion: float, gamma: float = 0.0, epsilon: float = 0.0
) -> ExecutionResult:
    """Almgren-Chriss efficient liquidation of `X` shares over `[0, T]`.

    Minimising :math:`E + \\lambda V` gives

    .. math::

       x_j = X\\,\\frac{\\sinh(\\kappa (T - t_j))}{\\sinh(\\kappa T)}, \\qquad
       \\cosh(\\kappa\\tau) = 1 + \\frac{\\lambda\\sigma^2\\tau^2}{2\\tilde\\eta},

    so :math:`1/\\kappa` is the half-life of the trade: a risk-neutral trader
    (:math:`\\lambda = 0`) sells evenly, and a very risk-averse one sells
    almost everything at once.

    Parameters
    ----------
    X : float
        Shares to sell.
    T : float
        Horizon.
    n_steps : int
        Number of trading intervals :math:`N`.
    sigma : float
        Price volatility per unit time (price units).
    eta : float
        Temporary impact per unit of trading rate.
    risk_aversion : float
        :math:`\\lambda \\ge 0`.
    gamma, epsilon : float
        Permanent impact and fixed cost per share.

    Returns
    -------
    ExecutionResult

    Examples
    --------
    >>> almgren_chriss(10.0, 1.0, 5, sigma=0.3, eta=0.05, risk_aversion=0.0).holdings.round(6).tolist()
    [10.0, 8.0, 6.0, 4.0, 2.0, 0.0]
    """
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    tau = T / n_steps
    eta_t = _temp_eta(eta, gamma, tau)
    t = np.linspace(0.0, T, n_steps + 1)
    kappa = np.arccosh(1.0 + 0.5 * risk_aversion * sigma**2 * tau**2 / eta_t) / tau
    x = X * (1.0 - t / T) if kappa * T < 1e-10 else X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    E, V = execution_cost(x, T, sigma, eta, gamma, epsilon)
    return ExecutionResult(times=t, holdings=x, trades=-np.diff(x), expected_cost=E, variance=V)


def _book_deviations(trades: np.ndarray, T: float, kappa: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
    N = trades.size - 1
    if N == 0:
        return np.zeros(1), kappa * trades
    k = np.arange(N + 1)
    lag = np.subtract.outer(k, k)
    decay = np.where(lag > 0, np.exp(-rho * (T / N) * np.maximum(lag, 0)), 0.0)
    before = kappa * (decay @ trades)
    return before, before + kappa * trades


def transient_impact_cost(trades: ArrayLike, T: float, kappa: float, rho: float, lam: float = 0.0) -> float:
    """Cost of any schedule of block purchases in the Obizhaeva-Wang book.

    A purchase of :math:`x_n` shares at :math:`t_n = nT/N` walks up a book of
    density :math:`1/\\kappa`, raising the ask's displacement from :math:`D_n^-`
    to :math:`D_n^+ = D_n^- + \\kappa x_n`, which then decays at rate :math:`\\rho`:
    :math:`D_{n+1}^- = e^{-\\rho T/N} D_n^+`. A fraction :math:`\\lambda` of
    impact is permanent and costs :math:`\\lambda X^2/2` whatever the schedule, so

    .. math::

       C = \\sum_{n=0}^N x_n\\left(D_n^- + \\tfrac12\\kappa x_n\\right) + \\tfrac12\\lambda X^2,

    measured above the ask before trading.

    Parameters
    ----------
    trades : array_like, shape (N + 1,)
        Shares bought at the evenly spaced dates :math:`t_0 = 0, \\dots, t_N = T`.
    T : float
        Horizon.
    kappa : float
        Transient impact per share, the inverse of the book's density.
    rho : float
        Resilience: the rate at which the book refills.
    lam : float
        Permanent impact per share.

    Returns
    -------
    float

    Examples
    --------
    A single block pays half the impact it causes; the same block split in
    two, with time for the book to refill in between, pays less.

    >>> transient_impact_cost([100.0], T=1.0, kappa=0.01, rho=2.0)
    50.0
    >>> round(transient_impact_cost([50.0, 50.0], T=1.0, kappa=0.01, rho=2.0), 6)
    28.383382
    """
    x = np.asarray(trades, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("trades must be a non-empty 1-D array")
    before, _ = _book_deviations(x, T, kappa, rho)
    return float(x @ (before + 0.5 * kappa * x) + 0.5 * lam * x.sum() ** 2)


def obizhaeva_wang(X: float, T: float, n_steps: int, kappa: float, rho: float, lam: float = 0.0) -> TransientImpactResult:
    """Obizhaeva and Wang's optimal purchase of `X` shares at ``n_steps + 1`` dates.

    Minimising :func:`transient_impact_cost` is a quadratic programme whose
    matrix :math:`a^{|n - m|}`, :math:`a = e^{-\\rho T/N}`, has a tridiagonal
    inverse, so the solution is explicit: equal blocks at the two ends and
    smaller equal trades in between,

    .. math::

       x_0 = x_N = \\frac{X}{2 + (N - 1)(1 - a)}, \\qquad x_n = (1 - a)\\,x_0 \\quad (0 < n < N),

    which keep the book's displacement constant between the blocks. As
    :math:`N \\to \\infty` this tends to blocks of :math:`X / (\\rho T + 2)`
    with trading at the rate :math:`\\rho X / (\\rho T + 2)` in between, and
    the cost to :math:`\\kappa X^2 / (\\rho T + 2) + \\lambda X^2 / 2`.

    Parameters
    ----------
    X : float
        Shares to buy (negative to sell).
    T : float
        Horizon.
    n_steps : int
        Number of intervals :math:`N \\ge 1` between the trading dates.
    kappa, rho, lam : float
        Transient impact per share, resilience and permanent impact per share.

    Returns
    -------
    TransientImpactResult

    Examples
    --------
    >>> res = obizhaeva_wang(100.0, 1.0, 1, kappa=0.01, rho=2.0)
    >>> res.trades.tolist(), round(res.expected_cost, 6)
    ([50.0, 50.0], 28.383382)
    """
    N = int(n_steps)
    if N < 1 or T <= 0 or kappa <= 0 or rho < 0:
        raise ValueError("need n_steps >= 1, T > 0, kappa > 0 and rho >= 0")
    a = np.exp(-rho * T / N)
    trades = np.full(N + 1, (1.0 - a) * X)
    trades[[0, -1]] = X
    trades /= 2.0 + (N - 1) * (1.0 - a)
    before, after = _book_deviations(trades, T, kappa, rho)
    cost = kappa * X**2 * (1.0 + a) / (2.0 * (2.0 + (N - 1) * (1.0 - a))) + 0.5 * lam * X**2
    return TransientImpactResult(times=np.linspace(0.0, T, N + 1), trades=trades, deviation_before=before, deviation_after=after, expected_cost=float(cost))

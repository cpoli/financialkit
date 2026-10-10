"""Dealers and market makers: Garman's order flow, Ho-Stoll inventory and Avellaneda-Stoikov.

Mark Garman (1976) modelled buy and sell orders as Poisson streams whose
rates depend on the ask and the bid, and showed that a dealer who sets
no spread runs out of stock or cash with certainty. Thomas Ho and Hans
Stoll (1981) made the dealer risk averse: inventory is risky, so a dealer
who is long shades both quotes down to sell more and buy less, while the
width of the spread pays for the order flow. Marco Avellaneda and Sasha
Stoikov (2008) solved the same trade-off in continuous time for a mid
price that moves as Brownian motion and fills that arrive at rate
:math:`A e^{-k\\delta}` at distance :math:`\\delta` from the mid.

References: M. B. Garman, "Market Microstructure," *J. Financial
Economics* 3 (1976) 257-275; T. Ho and H. R. Stoll, "Optimal Dealer
Pricing under Transactions and Return Uncertainty," *J. Financial
Economics* 9 (1981) 47-73; M. Avellaneda and S. Stoikov, "High-Frequency
Trading in a Limit Order Book," *Quantitative Finance* 8 (2008) 217-224;
M. O'Hara, *Market Microstructure Theory* (Blackwell, 1995), Ch. 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.integrators import brownian_increments
from financialkit.market_microstructure.core.base import GarmanResult, MarketMakingResult, QuotingStrategy

__all__ = [
    "garman_ruin_probability",
    "simulate_garman",
    "ho_stoll_quotes",
    "reservation_price",
    "avellaneda_stoikov_spread",
    "AvellanedaStoikov",
    "SymmetricQuotes",
    "simulate_market_making",
]


def garman_ruin_probability(inventory: ArrayLike, buy_rate: float, sell_rate: float) -> np.ndarray:
    """Probability that Garman's dealer eventually runs out of stock.

    Customers buy (the dealer sells one share) at rate `buy_rate` and sell
    at rate `sell_rate`, so the inventory is a gambler's-ruin random walk:

    .. math::

       P(\\text{ruin} \\mid I_0) = \\min\\left(1, \\left(\\lambda_a / \\lambda_b\\right)^{I_0}\\right).

    Parameters
    ----------
    inventory : array_like of int
        Starting inventory :math:`I_0 \\ge 0`.
    buy_rate : float
        :math:`\\lambda_a`, the arrival rate of buy orders at the ask.
    sell_rate : float
        :math:`\\lambda_b`, the arrival rate of sell orders at the bid.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> garman_ruin_probability([0, 3, 10], 2.0, 3.0).round(4).tolist()
    [1.0, 0.2963, 0.0173]
    >>> float(garman_ruin_probability(50, 1.0, 1.0))
    1.0
    """
    I = np.asarray(inventory, dtype=np.float64)
    if np.any(I < 0) or buy_rate <= 0 or sell_rate <= 0:
        raise ValueError("need inventory >= 0 and positive rates")
    return np.minimum(1.0, (buy_rate / sell_rate) ** I)


def simulate_garman(
    stock: int,
    cash: float,
    ask: float,
    bid: float,
    buy_rate: float,
    sell_rate: float,
    n_events: int,
    n_paths: int = 1,
    rng: np.random.Generator | int | None = None,
) -> GarmanResult:
    """Simulate Garman's dealer event by event until stock or cash runs out.

    A buy order (rate :math:`\\lambda_a`) takes one share at the ask; a sell
    order (rate :math:`\\lambda_b`) delivers one at the bid. The dealer fails
    when the stock is zero or the cash cannot pay the bid. Both inventories
    drift upward only if :math:`\\lambda_b > \\lambda_a` and
    :math:`a\\lambda_a > b\\lambda_b`, which needs :math:`a > b`: a spread.

    Parameters
    ----------
    stock : int
        Starting shares.
    cash : float
        Starting cash; ``np.inf`` isolates the stock inventory.
    ask, bid : float
        Fixed quotes.
    buy_rate, sell_rate : float
        :math:`\\lambda_a(a)` and :math:`\\lambda_b(b)` at those quotes.
    n_events : int
        Orders simulated per path.
    n_paths : int
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    GarmanResult
        Paths are frozen after failure.
    """
    rng = np.random.default_rng(rng)
    total = buy_rate + sell_rate
    buys = rng.random((n_paths, n_events)) < buy_rate / total
    waits = rng.exponential(1.0 / total, (n_paths, n_events))
    times = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(waits, axis=1)], axis=1)
    d_stock = np.where(buys, -1.0, 1.0)
    d_cash = np.where(buys, ask, -bid)
    s = stock + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(d_stock, axis=1)], axis=1)
    c = cash + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(d_cash, axis=1)], axis=1)
    # Failure: no stock to sell, or not enough cash to pay the next bid.
    dead = (s <= 0) | (c < bid)
    failed = dead.any(axis=1)
    first = np.where(failed, dead.argmax(axis=1), n_events)
    after = np.arange(n_events + 1) > first[:, None]
    s = np.where(after, s[np.arange(n_paths), first][:, None], s)
    c = np.where(after, c[np.arange(n_paths), first][:, None], c)
    t_fail = np.where(failed, times[np.arange(n_paths), np.minimum(first, n_events)], np.inf)
    return GarmanResult(times=times, stock=s, cash=c, failed=failed, failure_time=t_fail)


def ho_stoll_quotes(
    value: ArrayLike, inventory: ArrayLike, alpha: float, beta: float, risk_aversion: float, sigma: float, tau: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal bid and ask of a risk-averse dealer, a one-period Ho-Stoll model.

    Orders arrive at :math:`\\lambda = \\alpha - \\beta\\delta` on each side,
    :math:`\\delta` being the distance of the quote from the true value
    :math:`p`. The dealer maximises expected trading profit minus
    :math:`\\tfrac{\\gamma}{2}\\sigma^2\\tau` times the squared end-of-period
    inventory. The first-order conditions give Ho and Stoll's two results:

    .. math::

       \\delta_a + \\delta_b = \\frac{\\alpha}{\\beta}, \\qquad
       \\frac{a + b}{2} = p - \\frac{\\kappa I}{2(1 + \\kappa\\beta)}, \\quad \\kappa = \\gamma\\sigma^2\\tau:

    the spread does not depend on inventory, and the quotes slide down
    when the dealer is long.

    Parameters
    ----------
    value : array_like
        True value :math:`p`.
    inventory : array_like
        Dealer's inventory :math:`I`.
    alpha, beta : float
        Intercept and slope of the order-arrival rate.
    risk_aversion : float
        :math:`\\gamma \\ge 0`.
    sigma : float
        Volatility of the value, per unit time.
    tau : float
        Holding horizon.

    Returns
    -------
    bid, ask : ndarray

    Examples
    --------
    >>> b, a = ho_stoll_quotes(100.0, [-10, 0, 10], alpha=10.0, beta=20.0, risk_aversion=0.1, sigma=0.2)
    >>> (a - b).round(6).tolist(), ((a + b) / 2).round(4).tolist()
    ([0.5, 0.5, 0.5], [100.0185, 100.0, 99.9815])
    """
    if beta <= 0 or risk_aversion < 0:
        raise ValueError("need beta > 0 and risk_aversion >= 0")
    kappa = risk_aversion * sigma**2 * tau
    skew = -kappa * np.asarray(inventory, dtype=np.float64) / (1.0 + kappa * beta)
    half = 0.5 * alpha / beta
    p = np.asarray(value, dtype=np.float64)
    return p - half + 0.5 * skew, p + half + 0.5 * skew


def reservation_price(mid: ArrayLike, inventory: ArrayLike, risk_aversion: float, sigma: float, time_left: ArrayLike) -> np.ndarray:
    """Avellaneda-Stoikov indifference price :math:`r = s - q\\gamma\\sigma^2 (T - t)`.

    Examples
    --------
    >>> reservation_price(100.0, [-2, 0, 2], 0.1, 2.0, 0.5).tolist()
    [100.4, 100.0, 99.6]
    """
    return np.asarray(mid, dtype=np.float64) - np.asarray(inventory, dtype=np.float64) * risk_aversion * sigma**2 * np.asarray(time_left, dtype=np.float64)


def avellaneda_stoikov_spread(risk_aversion: float, sigma: float, time_left: ArrayLike, k: float) -> np.ndarray:
    """Optimal total spread :math:`\\gamma\\sigma^2 (T - t) + \\frac{2}{\\gamma}\\ln(1 + \\gamma / k)`.

    As :math:`\\gamma \\to 0` it tends to :math:`2/k`, the spread that
    maximises expected profit from fills at rate :math:`A e^{-k\\delta}`.

    Examples
    --------
    >>> round(float(avellaneda_stoikov_spread(0.1, 2.0, 1.0, 1.5)), 4)
    1.6908
    """
    if risk_aversion <= 0 or k <= 0:
        raise ValueError("need risk_aversion > 0 and k > 0")
    return risk_aversion * sigma**2 * np.asarray(time_left, dtype=np.float64) + 2.0 / risk_aversion * np.log1p(risk_aversion / k)


class AvellanedaStoikov(QuotingStrategy):
    """Avellaneda-Stoikov's inventory strategy: the optimal spread around the reservation price.

    Parameters
    ----------
    risk_aversion : float
        :math:`\\gamma`.
    sigma : float
        Mid-price volatility in price units per square root of time.
    k : float
        Decay of the fill intensity with distance from the mid.
    T : float
        Terminal time.
    """

    def __init__(self, risk_aversion: float, sigma: float, k: float, T: float):
        self.risk_aversion, self.sigma, self.k, self.T = risk_aversion, sigma, k, T

    def quotes(self, mid: np.ndarray, inventory: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        r = reservation_price(mid, inventory, self.risk_aversion, self.sigma, self.T - t)
        half = 0.5 * avellaneda_stoikov_spread(self.risk_aversion, self.sigma, self.T - t, self.k)
        return r - half, r + half


class SymmetricQuotes(QuotingStrategy):
    """Quotes a fixed half-spread either side of the mid, ignoring inventory.

    Parameters
    ----------
    half_spread : float or callable
        A number, or ``f(t)`` returning one.
    """

    def __init__(self, half_spread):
        self.half_spread = half_spread

    def quotes(self, mid: np.ndarray, inventory: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        h = self.half_spread(t) if callable(self.half_spread) else self.half_spread
        return mid - h, mid + h


def simulate_market_making(
    strategy: QuotingStrategy,
    s0: float,
    sigma: float,
    T: float,
    n_steps: int,
    A: float,
    k: float,
    n_paths: int = 1,
    rng: np.random.Generator | int | None = None,
) -> MarketMakingResult:
    """Simulate a market maker as in Avellaneda and Stoikov's Section 4.

    The mid price is arithmetic Brownian motion (increments from
    :func:`financialkit.integrators.brownian_increments`). In each step of
    length :math:`dt` the ask is lifted with probability
    :math:`A e^{-k\\delta_a} dt` and the bid hit with probability
    :math:`A e^{-k\\delta_b} dt`, one share each.

    Parameters
    ----------
    strategy : QuotingStrategy
    s0 : float
        Initial mid price.
    sigma : float
        Mid-price volatility.
    T : float
        Horizon.
    n_steps : int
        Choose it so that :math:`A\\,dt` stays well below one.
    A, k : float
        Fill-intensity parameters.
    n_paths : int
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    MarketMakingResult
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    dW = brownian_increments(T, n_steps, n_paths, rng=rng)
    u = rng.random((n_paths, n_steps, 2))
    mid = s0 + sigma * np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    q = np.zeros((n_paths, n_steps + 1))
    x = np.zeros((n_paths, n_steps + 1))
    bid = np.empty((n_paths, n_steps))
    ask = np.empty((n_paths, n_steps))
    times = np.linspace(0.0, T, n_steps + 1)
    for i in range(n_steps):
        b, a = strategy.quotes(mid[:, i], q[:, i], times[i])
        bid[:, i], ask[:, i] = b, a
        sold = u[:, i, 0] < np.minimum(1.0, A * np.exp(-k * (a - mid[:, i])) * dt)
        bought = u[:, i, 1] < np.minimum(1.0, A * np.exp(-k * (mid[:, i] - b)) * dt)
        q[:, i + 1] = q[:, i] - sold + bought
        x[:, i + 1] = x[:, i] + sold * a - bought * b
    return MarketMakingResult(times=times, mid=mid, bid=bid, ask=ask, inventory=q, cash=x)

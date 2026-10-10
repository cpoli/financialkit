"""Asymmetric information: Kyle's insider and Glosten-Milgrom's dealer.

Albert Kyle (1985) let one insider who knows the asset's value trade
against noise traders through a market maker who sees only total order
flow. In the linear equilibrium the price moves :math:`\\lambda` per
unit of flow, the insider trades just enough to hide behind the noise,
and half the private information enters the price. Lawrence Glosten and
Paul Milgrom (1985) showed that a competitive dealer facing traders who
may be informed must quote a spread even with no costs or inventory:
the ask is the expected value given a buy and the bid given a sell, and
the quotes converge to the truth as trades reveal it.

References: A. S. Kyle, "Continuous Auctions and Insider Trading,"
*Econometrica* 53 (1985) 1315-1335; L. R. Glosten and P. R. Milgrom,
"Bid, Ask and Transaction Prices in a Specialist Market with
Heterogeneously Informed Traders," *J. Financial Economics* 14 (1985)
71-100.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.market_microstructure.core.base import GlostenMilgromResult, KyleResult

__all__ = ["kyle_equilibrium", "simulate_kyle", "glosten_milgrom_quotes", "simulate_glosten_milgrom"]


def kyle_equilibrium(sigma_v: float, sigma_u: float, p0: float = 0.0) -> KyleResult:
    """Kyle's single-auction equilibrium :math:`\\lambda = \\sigma_v / (2\\sigma_u)`, :math:`\\beta = \\sigma_u / \\sigma_v`.

    The value is :math:`v \\sim N(p_0, \\sigma_v^2)`, noise flow
    :math:`u \\sim N(0, \\sigma_u^2)`, the insider submits
    :math:`x = \\beta(v - p_0)` and the market maker sets
    :math:`p = p_0 + \\lambda(x + u)`.

    Parameters
    ----------
    sigma_v : float
        Standard deviation of the value.
    sigma_u : float
        Standard deviation of the noise order flow.
    p0 : float
        Prior mean of the value.

    Returns
    -------
    KyleResult

    Examples
    --------
    Doubling the noise halves the price impact and doubles the insider's
    profit.

    >>> [(r.lam, r.insider_profit) for r in (kyle_equilibrium(2.0, 1.0), kyle_equilibrium(2.0, 2.0))]
    [(1.0, 1.0), (0.5, 2.0)]
    """
    if sigma_v <= 0 or sigma_u <= 0:
        raise ValueError("sigma_v and sigma_u must be positive")
    return KyleResult(
        lam=sigma_v / (2.0 * sigma_u),
        beta=sigma_u / sigma_v,
        insider_profit=0.5 * sigma_v * sigma_u,
        posterior_variance=0.5 * sigma_v**2,
        sigma_v=sigma_v,
        sigma_u=sigma_u,
        p0=p0,
    )


def simulate_kyle(
    eq: KyleResult, n_auctions: int, beta: float | None = None, rng: np.random.Generator | int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw independent single-period Kyle markets.

    Parameters
    ----------
    eq : KyleResult
        Supplies :math:`\\sigma_v`, :math:`\\sigma_u`, :math:`p_0` and the market maker's :math:`\\lambda`.
    n_auctions : int
    beta : float, optional
        Insider's intensity; the equilibrium value if omitted. Any other
        value shows why it is optimal.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    value, order_flow, price : ndarray, shape (n_auctions,)
        :math:`v`, total flow :math:`x + u` and the clearing price.
    """
    rng = np.random.default_rng(rng)
    b = eq.beta if beta is None else beta
    v = eq.p0 + eq.sigma_v * rng.standard_normal(n_auctions)
    y = b * (v - eq.p0) + eq.sigma_u * rng.standard_normal(n_auctions)
    return v, y, eq.p0 + eq.lam * y


def glosten_milgrom_quotes(belief: ArrayLike, alpha: float, v_low: float, v_high: float, eta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Zero-profit bid and ask when a fraction `alpha` of traders know the value.

    The value is :math:`V_H` with probability :math:`\\theta` (the belief)
    and :math:`V_L` otherwise. An informed trader buys when it is high
    and sells when it is low; an uninformed one buys with probability
    :math:`\\eta`. Then

    .. math::

       a = E[V \\mid \\text{buy}] = V_L + (V_H - V_L)\\,
       \\frac{\\theta(\\alpha + (1-\\alpha)\\eta)}{\\theta\\alpha + (1-\\alpha)\\eta},

    and the bid is the same with a sell.

    Parameters
    ----------
    belief : array_like
        :math:`\\theta = P(V = V_H)`.
    alpha : float
        Probability that a trader is informed, in ``[0, 1)``.
    v_low, v_high : float
    eta : float
        Probability that an uninformed trader buys.

    Returns
    -------
    bid, ask : ndarray

    Examples
    --------
    >>> b, a = glosten_milgrom_quotes(0.5, 0.2, 100.0, 110.0)
    >>> float(b), float(a)
    (104.0, 106.0)
    """
    if not (0.0 <= alpha < 1.0 and 0.0 < eta < 1.0):
        raise ValueError("need 0 <= alpha < 1 and 0 < eta < 1")
    th = np.asarray(belief, dtype=np.float64)
    buy_h, buy_l = alpha + (1 - alpha) * eta, (1 - alpha) * eta
    sell_h, sell_l = (1 - alpha) * (1 - eta), alpha + (1 - alpha) * (1 - eta)
    th_buy = th * buy_h / (th * buy_h + (1 - th) * buy_l)
    th_sell = th * sell_h / (th * sell_h + (1 - th) * sell_l)
    dv = v_high - v_low
    return v_low + th_sell * dv, v_low + th_buy * dv


def simulate_glosten_milgrom(
    n_trades: int,
    alpha: float,
    v_low: float,
    v_high: float,
    belief0: float = 0.5,
    eta: float = 0.5,
    true_value: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> GlostenMilgromResult:
    """A sequence of Glosten-Milgrom trades with Bayesian quote updates.

    Parameters
    ----------
    n_trades : int
    alpha, v_low, v_high, eta : float
        As in :func:`glosten_milgrom_quotes`.
    belief0 : float
        Prior probability of the high value.
    true_value : float, optional
        ``v_low`` or ``v_high``; drawn from the prior if omitted.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    GlostenMilgromResult
        After a buy the belief becomes the one that priced the ask, and
        after a sell the one that priced the bid.
    """
    rng = np.random.default_rng(rng)
    if true_value is None:
        true_value = v_high if rng.random() < belief0 else v_low
    if true_value not in (v_low, v_high):
        raise ValueError("true_value must be v_low or v_high")
    informed = rng.random(n_trades) < alpha
    noise_buy = rng.random(n_trades) < eta
    trades = np.where(informed, 1.0 if true_value == v_high else -1.0, np.where(noise_buy, 1.0, -1.0))
    belief = np.empty(n_trades + 1)
    belief[0] = belief0
    bid = np.empty(n_trades)
    ask = np.empty(n_trades)
    dv = v_high - v_low
    for i in range(n_trades):
        bid[i], ask[i] = glosten_milgrom_quotes(belief[i], alpha, v_low, v_high, eta)
        belief[i + 1] = ((ask[i] if trades[i] > 0 else bid[i]) - v_low) / dv
    return GlostenMilgromResult(bid=bid, ask=ask, belief=belief, trades=trades, informed=informed, v_low=v_low, v_high=v_high, true_value=float(true_value))

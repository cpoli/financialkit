"""Abstract base classes and result containers for financialkit.market_microstructure.

Sign conventions, domain-wide: a trade sign or order flow is ``+1`` (or
positive) for a buyer-initiated trade and ``-1`` (negative) for a
seller-initiated one; a dealer's inventory is positive when long.
Spreads are in price units unless a ``relative`` flag says otherwise.

Closed forms (Kyle's lambda, Glosten-Milgrom quotes, the
Avellaneda-Stoikov reservation price, the square-root law) return plain
floats or broadcasting arrays. Simulations and estimators return a small
dataclass that the visualizers plot. :class:`QuotingStrategy` is the
interface the market-making simulator calls each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = [
    "KyleResult",
    "GlostenMilgromResult",
    "GarmanResult",
    "MarketMakingResult",
    "ExecutionResult",
    "BertsimasLoResult",
    "TradeImpactResult",
    "PINResult",
    "VPINResult",
    "HawkesResult",
    "OrderBookResult",
    "TransientImpactResult",
    "QuotingStrategy",
]


@dataclass
class KyleResult:
    """Kyle's (1985) single-auction linear equilibrium."""

    lam: float
    """Price impact :math:`\\lambda = \\sigma_v / (2\\sigma_u)`: price change per unit of order flow."""

    beta: float
    """Insider's trading intensity :math:`\\beta = \\sigma_u / \\sigma_v`, so ``x = beta (v - p0)``."""

    insider_profit: float
    """Insider's expected profit :math:`\\sigma_v \\sigma_u / 2`, equal to the noise traders' expected loss."""

    posterior_variance: float
    """:math:`\\operatorname{Var}(v \\mid p) = \\sigma_v^2 / 2`: half the private information is revealed."""

    sigma_v: float
    """Standard deviation of the asset's liquidation value."""

    sigma_u: float
    """Standard deviation of the noise traders' order flow."""

    p0: float
    """Prior mean of the value, the price before trading."""


@dataclass
class GlostenMilgromResult:
    """A simulated sequence of Glosten-Milgrom trades and quotes."""

    bid: np.ndarray
    """ndarray, shape (n,): Bid quoted before each trade, :math:`E[V \\mid \\text{history}, \\text{sell}]`."""

    ask: np.ndarray
    """ndarray, shape (n,): Ask quoted before each trade, :math:`E[V \\mid \\text{history}, \\text{buy}]`."""

    belief: np.ndarray
    """ndarray, shape (n + 1,): Dealer's probability that the value is high, before and after each trade."""

    trades: np.ndarray
    """ndarray, shape (n,): ``+1`` buy or ``-1`` sell."""

    informed: np.ndarray
    """ndarray of bool, shape (n,): Whether each trader was informed."""

    v_low: float
    """Low value."""

    v_high: float
    """High value."""

    true_value: float
    """The value the informed traders know."""


@dataclass
class GarmanResult:
    """Garman's (1976) dealer under Poisson order flow, simulated."""

    times: np.ndarray
    """ndarray, shape (n_paths, n_events + 1): Event times, starting at 0."""

    stock: np.ndarray
    """ndarray, shape (n_paths, n_events + 1): Dealer's inventory of shares after each event."""

    cash: np.ndarray
    """ndarray, shape (n_paths, n_events + 1): Dealer's cash after each event."""

    failed: np.ndarray
    """ndarray of bool, shape (n_paths,): Whether stock or cash ran out within the simulated events."""

    failure_time: np.ndarray
    """ndarray, shape (n_paths,): Time of failure, ``inf`` for survivors."""


@dataclass
class MarketMakingResult:
    """A simulated market maker quoting around a Brownian mid price."""

    times: np.ndarray
    """ndarray, shape (n_steps + 1,): Time grid."""

    mid: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Mid price."""

    bid: np.ndarray
    """ndarray, shape (n_paths, n_steps): Bid quoted during each step."""

    ask: np.ndarray
    """ndarray, shape (n_paths, n_steps): Ask quoted during each step."""

    inventory: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Shares held."""

    cash: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Cash account."""

    @property
    def pnl(self) -> np.ndarray:
        """ndarray, shape (n_paths,): Terminal profit, cash plus inventory marked at the mid."""
        return self.cash[:, -1] + self.inventory[:, -1] * self.mid[:, -1]


@dataclass
class ExecutionResult:
    """An execution schedule for selling (or buying) a block of shares."""

    times: np.ndarray
    """ndarray, shape (N + 1,): Trading dates :math:`t_0 = 0, \\dots, t_N = T`."""

    holdings: np.ndarray
    """ndarray, shape (N + 1,): Shares still to trade at each date, from `X` down to 0."""

    trades: np.ndarray
    """ndarray, shape (N,): Shares traded in each interval, :math:`n_k = x_{k-1} - x_k`."""

    expected_cost: float
    """Expected implementation shortfall."""

    variance: float
    """Variance of the implementation shortfall."""


@dataclass
class BertsimasLoResult:
    """Bertsimas and Lo's (1998) dynamic-programming execution policy.

    Before period ``k`` (0-based) with :math:`W_k` shares left and last
    signal :math:`X_{k-1}`, the optimal order is
    :math:`S_k = \\delta_{w,k} W_k + \\delta_{x,k} X_{k-1}`, and the expected
    remaining cost is
    :math:`P_{k-1} W_k + a_k W_k^2 + b_k X_{k-1} W_k + c_k X_{k-1}^2 + d_k`.
    """

    delta_w: np.ndarray
    """ndarray, shape (T,): Fraction of the remaining shares traded each period."""

    delta_x: np.ndarray
    """ndarray, shape (T,): Shares traded per unit of the last signal."""

    a: np.ndarray
    """ndarray, shape (T,): Quadratic cost coefficient of the value function."""

    b: np.ndarray
    """ndarray, shape (T,): Signal-shares coefficient of the value function."""

    c: np.ndarray
    """ndarray, shape (T,): Squared-signal coefficient of the value function."""

    d: np.ndarray
    """ndarray, shape (T,): Constant of the value function."""

    trades: np.ndarray
    """ndarray, shape (T,): Orders executed along the given signal path."""

    expected_cost: float
    """Expected total cost from the start, :math:`E[\\sum_k P_k S_k]`."""


@dataclass
class TradeImpactResult:
    """Hasbrouck's (1991) vector autoregression of quote revisions and trades."""

    coef_r: np.ndarray
    """ndarray, shape (1 + 2 * lags,): Return equation, columns ``[x_t, r_{t-1..t-p}, x_{t-1..t-p}]`` (no intercept)."""

    coef_x: np.ndarray
    """ndarray, shape (2 * lags,): Trade equation, columns ``[r_{t-1..t-p}, x_{t-1..t-p}]``."""

    impulse: np.ndarray
    """ndarray, shape (horizon + 1,): Cumulative quote revision after a unit trade innovation."""

    permanent_impact: float
    """Long-run price impact of a unit trade innovation, the last entry of :attr:`impulse`."""

    trade_share: float
    """Fraction of the efficient price's variance explained by trades, Hasbrouck's :math:`R^2_w`."""

    innovation_cov: np.ndarray
    """ndarray, shape (2, 2): Covariance of the return and trade innovations."""


@dataclass
class PINResult:
    """Easley-Kiefer-O'Hara-Paperman model fitted by maximum likelihood."""

    alpha: float
    """Probability of an information event on a day."""

    delta: float
    """Probability that an information event is bad news."""

    mu: float
    """Arrival rate of informed traders on event days."""

    eps_b: float
    """Arrival rate of uninformed buyers."""

    eps_s: float
    """Arrival rate of uninformed sellers."""

    pin: float
    """Probability of informed trading :math:`\\alpha\\mu / (\\alpha\\mu + \\varepsilon_b + \\varepsilon_s)`."""

    pin_se: float
    """Delta-method standard error of :attr:`pin` from the observed information."""

    loglik: float
    """Maximised log-likelihood."""


@dataclass
class VPINResult:
    """Volume-synchronized probability of informed trading."""

    vpin: np.ndarray
    """ndarray, shape (n_buckets,): Rolling mean order imbalance; ``nan`` until `window` buckets are full."""

    buy_volume: np.ndarray
    """ndarray, shape (n_buckets,): Buy volume classified into each bucket."""

    sell_volume: np.ndarray
    """ndarray, shape (n_buckets,): Sell volume classified into each bucket."""

    bucket_end: np.ndarray
    """ndarray of int, shape (n_buckets,): Index of the bar in which each bucket filled."""

    @property
    def imbalance(self) -> np.ndarray:
        """ndarray, shape (n_buckets,): :math:`|V^B - V^S| / V` for each bucket."""
        return np.abs(self.buy_volume - self.sell_volume) / (self.buy_volume + self.sell_volume)


@dataclass
class HawkesResult:
    """A Hawkes process with exponential kernel fitted by maximum likelihood.

    The intensity is :math:`\\lambda(t) = \\mu + \\sum_{t_i < t} \\alpha e^{-\\beta (t - t_i)}`.
    """

    mu: float
    """Baseline (immigrant) intensity."""

    alpha: float
    """Jump in the intensity after each event."""

    beta: float
    """Decay rate of the excitation."""

    loglik: float
    """Maximised log-likelihood."""

    @property
    def branching_ratio(self) -> float:
        """:math:`n = \\alpha / \\beta`: expected number of events each event triggers directly."""
        return self.alpha / self.beta

    @property
    def mean_intensity(self) -> float:
        """Stationary event rate :math:`\\mu / (1 - n)`, ``inf`` when :math:`n \\ge 1`."""
        n = self.branching_ratio
        return self.mu / (1.0 - n) if n < 1 else float("inf")


@dataclass
class OrderBookResult:
    """A simulated Cont-Stoikov-Talreja limit order book, recorded after every event.

    Prices are in ticks. Event codes: 0 limit buy, 1 limit sell, 2 market
    buy, 3 market sell, 4 cancelled buy, 5 cancelled sell.
    """

    times: np.ndarray
    """ndarray, shape (n_events + 1,): Event times, starting at 0."""

    events: np.ndarray
    """ndarray of int, shape (n_events,): Code of each event."""

    best_bid: np.ndarray
    """ndarray of int, shape (n_events + 1,): Best bid; ``-1`` when the bid side is empty."""

    best_ask: np.ndarray
    """ndarray of int, shape (n_events + 1,): Best ask; ``n_ticks`` when the ask side is empty."""

    bid_size: np.ndarray
    """ndarray of int, shape (n_events + 1,): Orders queued at the best bid."""

    ask_size: np.ndarray
    """ndarray of int, shape (n_events + 1,): Orders queued at the best ask."""

    mean_depth: np.ndarray
    """ndarray, shape (n_levels,): Time-averaged orders at ``i = 1..n_levels`` ticks from the opposite best quote, both sides pooled."""

    @property
    def mid(self) -> np.ndarray:
        """ndarray, shape (n_events + 1,): Mid price in ticks."""
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> np.ndarray:
        """ndarray of int, shape (n_events + 1,): Bid-ask spread in ticks."""
        return self.best_ask - self.best_bid


@dataclass
class TransientImpactResult:
    """Obizhaeva and Wang's (2013) optimal schedule in a book that recovers exponentially."""

    times: np.ndarray
    """ndarray, shape (N + 1,): Trading dates :math:`t_0 = 0, \\dots, t_N = T`."""

    trades: np.ndarray
    """ndarray, shape (N + 1,): Shares bought at each date, summing to `X`."""

    deviation_before: np.ndarray
    """ndarray, shape (N + 1,): Displacement :math:`D_n^-` of the ask from its steady state just before each trade."""

    deviation_after: np.ndarray
    """ndarray, shape (N + 1,): Displacement :math:`D_n^+ = D_n^- + \\kappa x_n` just after each trade."""

    expected_cost: float
    """Cost above the pre-trade ask, :math:`\\sum_n x_n (D_n^- + \\kappa x_n / 2) + \\lambda X^2 / 2`."""


class QuotingStrategy(ABC):
    """A market maker's rule for its bid and ask."""

    @abstractmethod
    def quotes(self, mid: np.ndarray, inventory: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Bid and ask given the mid price and inventory of every path at time `t`.

        Parameters
        ----------
        mid : ndarray, shape (n_paths,)
        inventory : ndarray, shape (n_paths,)
        t : float

        Returns
        -------
        bid, ask : ndarray, shape (n_paths,)
        """

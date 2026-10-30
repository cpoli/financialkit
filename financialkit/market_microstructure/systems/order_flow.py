"""Order flow: Hawkes clustering, the Cont-Stoikov-Talreja book and order-flow imbalance.

Orders do not arrive as a Poisson stream: they cluster, and one order
makes the next more likely. Alan Hawkes (1971) wrote down the
self-exciting point process whose intensity jumps at every event and
decays between them; it was later fitted to trades and quotes (Bowsher,
2007; Bacry, Mastromatteo and Muzy, 2015). Rama Cont, Sasha Stoikov and
Rishi Talreja (2010) modelled the whole limit order book as a
continuous-time Markov chain of limit orders, market orders and
cancellations, in which the book's shape and the probability that the
next price move is up follow from the order rates. Cont, Arseniy Kukanov
and Stoikov (2014) found that over short intervals the price change is
linear in the order-flow imbalance at the best quotes, which explains it
far better than the imbalance of trades alone.

References: A. G. Hawkes, "Spectra of Some Self-Exciting and Mutually
Exciting Point Processes," *Biometrika* 58 (1971) 83-90; A. G. Hawkes
and D. Oakes, "A Cluster Process Representation of a Self-Exciting
Process," *J. Applied Probability* 11 (1974) 493-503; T. Ozaki,
"Maximum Likelihood Estimation of Hawkes' Self-Exciting Point
Processes," *Annals of the Institute of Statistical Mathematics* 31
(1979) 145-155; Y. Ogata, "Statistical Models for Earthquake
Occurrences and Residual Analysis for Point Processes," *J. American
Statistical Association* 83 (1988) 9-27; R. Cont, S. Stoikov and R.
Talreja, "A Stochastic Model for Order Book Dynamics," *Operations
Research* 58 (2010) 549-563; R. Cont, A. Kukanov and S. Stoikov, "The
Price Impact of Order Book Events," *J. Financial Econometrics* 12
(2014) 47-88.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from financialkit.market_microstructure.core.base import HawkesResult, OrderBookResult
from financialkit.time_series.utils.numerics import OLSResult, ols

__all__ = [
    "simulate_hawkes",
    "hawkes_intensity",
    "hawkes_loglik",
    "hawkes_compensator",
    "fit_hawkes",
    "simulate_cst",
    "cst_mid_up_probability",
    "order_flow_imbalance",
    "fit_ofi_impact",
]


def _hawkes_params(mu: float, alpha: float, beta: float) -> None:
    if mu <= 0 or alpha < 0 or beta <= 0:
        raise ValueError("need mu > 0, alpha >= 0 and beta > 0")


def _excitation(times: np.ndarray, beta: float) -> np.ndarray:
    """:math:`A_i = \\sum_{j < i} e^{-\\beta (t_i - t_j)}` by Ozaki's recursion."""
    A = [0.0] * times.size
    t = times.tolist()
    for i in range(1, len(t)):
        A[i] = math.exp(-beta * (t[i] - t[i - 1])) * (1.0 + A[i - 1])
    return np.asarray(A)


def _event_times(times: ArrayLike) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or np.any(np.diff(t) < 0):
        raise ValueError("times must be a sorted 1-D array")
    return t


def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Event times of a Hawkes process with exponential kernel on ``[0, T]``.

    The intensity is :math:`\\lambda(t) = \\mu + \\sum_{t_i < t} \\alpha e^{-\\beta(t - t_i)}`.
    The simulation uses Hawkes and Oakes's cluster representation:
    immigrants arrive as a Poisson process of rate :math:`\\mu`, and every
    event has a Poisson(:math:`n`) number of children, :math:`n = \\alpha/\\beta`,
    each after an exponential delay of rate :math:`\\beta`. One generation
    is drawn at a time, all of it at once.

    Parameters
    ----------
    mu : float
        Baseline intensity.
    alpha : float
        Jump of the intensity at each event.
    beta : float
        Decay rate, with :math:`\\alpha < \\beta` so that the process is stationary.
    T : float
        Horizon.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray
        Sorted event times.

    Examples
    --------
    >>> t = simulate_hawkes(1.0, 0.5, 1.0, 1000.0, rng=0)
    >>> bool(1700 < t.size < 2300)  # mean mu T / (1 - alpha / beta) = 2000
    True
    """
    _hawkes_params(mu, alpha, beta)
    if alpha >= beta:
        raise ValueError("need alpha < beta: the process explodes otherwise")
    rng = np.random.default_rng(rng)
    generation = rng.uniform(0.0, T, rng.poisson(mu * T))
    events = [generation]
    while generation.size:
        parents = np.repeat(generation, rng.poisson(alpha / beta, generation.size))
        children = parents + rng.exponential(1.0 / beta, parents.size)
        generation = children[children < T]
        events.append(generation)
    return np.sort(np.concatenate(events))


def hawkes_intensity(times: ArrayLike, t: ArrayLike, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional intensity :math:`\\lambda(t)` given the events strictly before `t`.

    Parameters
    ----------
    times : array_like, shape (n,)
        Sorted event times.
    t : array_like
        Times at which to evaluate the intensity.
    mu, alpha, beta : float
        Baseline, jump and decay.

    Returns
    -------
    ndarray, same shape as `t`

    Examples
    --------
    >>> hawkes_intensity([1.0], [0.5, 2.0], mu=0.5, alpha=1.0, beta=np.log(2.0)).tolist()
    [0.5, 1.0]
    """
    ev = _event_times(times)
    s = np.asarray(t, dtype=np.float64)
    if ev.size == 0:
        return np.full(s.shape, float(mu))
    k = np.searchsorted(ev, s, side="left") - 1
    B = 1.0 + _excitation(ev, beta)
    last = np.maximum(k, 0)
    excite = np.where(k >= 0, B[last] * np.exp(-beta * (s - ev[last])), 0.0)
    return mu + alpha * excite


def hawkes_loglik(times: ArrayLike, T: float, mu: float, alpha: float, beta: float) -> float:
    """Exact log-likelihood of events on ``[0, T]`` with no history before 0.

    .. math::

       \\ell = \\sum_i \\log(\\mu + \\alpha A_i) - \\mu T - \\frac{\\alpha}{\\beta}\\sum_i \\left(1 - e^{-\\beta(T - t_i)}\\right),
       \\qquad A_i = e^{-\\beta(t_i - t_{i-1})}(1 + A_{i-1}),

    Ozaki's recursion, which makes it :math:`O(n)` instead of :math:`O(n^2)`.

    Parameters
    ----------
    times : array_like, shape (n,)
        Sorted event times in ``[0, T]``.
    T : float
        End of the observation window.
    mu, alpha, beta : float

    Returns
    -------
    float

    Examples
    --------
    With ``alpha = 0`` it is the Poisson log-likelihood :math:`n \\log\\mu - \\mu T`.

    >>> round(hawkes_loglik([0.5, 1.5, 2.0], 4.0, mu=0.75, alpha=0.0, beta=1.0), 6)
    -3.863046
    """
    _hawkes_params(mu, alpha, beta)
    t = _event_times(times)
    A = _excitation(t, beta)
    return float(np.sum(np.log(mu + alpha * A)) - mu * T - alpha / beta * np.sum(1.0 - np.exp(-beta * (T - t))))


def hawkes_compensator(times: ArrayLike, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Compensator :math:`\\Lambda(t_i) = \\int_0^{t_i} \\lambda(s)\\,ds` at each event.

    By the random time change theorem, the increments of
    :math:`\\Lambda(t_i)` are independent standard exponentials when the
    model is right, which is Ogata's (1988) residual test for point processes.

    Parameters
    ----------
    times : array_like, shape (n,)
        Sorted event times.
    mu, alpha, beta : float

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> hawkes_compensator([1.0, 3.0], mu=0.5, alpha=0.0, beta=1.0).tolist()
    [0.5, 1.5]
    """
    _hawkes_params(mu, alpha, beta)
    t = _event_times(times)
    return mu * t + alpha / beta * (np.arange(t.size) - _excitation(t, beta))


def fit_hawkes(times: ArrayLike, T: float, x0: tuple[float, float, float] | None = None) -> HawkesResult:
    """Fit an exponential-kernel Hawkes process by maximum likelihood.

    Maximises :func:`hawkes_loglik` over the logs of
    :math:`(\\mu, \\alpha, \\beta)` with :func:`scipy.optimize.minimize`
    (L-BFGS-B).

    Parameters
    ----------
    times : array_like, shape (n,)
        Sorted event times in ``[0, T]``.
    T : float
        End of the observation window.
    x0 : tuple of float, optional
        Starting :math:`(\\mu, \\alpha, \\beta)`; by default half the events
        are immigrants and the kernel decays over ten mean inter-event times.

    Returns
    -------
    HawkesResult

    Examples
    --------
    >>> fit = fit_hawkes(simulate_hawkes(1.0, 0.6, 1.2, 2000.0, rng=1), 2000.0)
    >>> bool(abs(fit.branching_ratio - 0.5) < 0.1)
    True
    """
    t = _event_times(times)
    if t.size < 3:
        raise ValueError("need at least three events")
    rate = t.size / T
    start = (0.5 * rate, 0.05 * rate, 0.1 * rate) if x0 is None else x0
    res = minimize(lambda z: -hawkes_loglik(t, T, *np.exp(z)), np.log(start), method="L-BFGS-B")
    mu, alpha, beta = np.exp(res.x)
    return HawkesResult(mu=float(mu), alpha=float(alpha), beta=float(beta), loglik=float(-res.fun))


def simulate_cst(
    lam: ArrayLike,
    mu: float,
    theta: ArrayLike,
    n_events: int,
    n_ticks: int = 200,
    initial_depth: ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> OrderBookResult:
    """Simulate the Cont-Stoikov-Talreja limit order book event by event.

    On a grid of `n_ticks` prices, unit-size orders arrive independently:

    - limit buys at :math:`i` ticks below the best ask, and limit sells at
      :math:`i` ticks above the best bid, each at rate :math:`\\lambda(i)`;
    - market buys and market sells, each at rate :math:`\\mu`, which take one
      order from the best ask or the best bid;
    - cancellations, at rate :math:`\\theta(i)` for each order resting
      :math:`i` ticks from the opposite best quote.

    The book is a continuous-time Markov chain, simulated exactly by
    drawing the time to the next event from the total rate and the event in
    proportion to its rate. Orders more than ``n_levels = len(lam)`` ticks
    from the opposite best quote receive no arrivals and are cancelled at
    rate ``theta[-1]``.

    Parameters
    ----------
    lam : array_like, shape (n_levels,)
        Limit order rates :math:`\\lambda(1), \\dots, \\lambda(n_{\\text{levels}})`.
    mu : float
        Market order rate on each side.
    theta : array_like, shape (n_levels,)
        Cancellation rates per resting order.
    n_events : int
        Number of events to simulate.
    n_ticks : int
        Size of the price grid; the book starts with a one-tick spread at its centre.
    initial_depth : array_like, shape (n_levels,), optional
        Orders at distance ``1..n_levels`` from the opposite best quote at
        the start, on both sides; :math:`\\lambda(i)/\\theta(i)` rounded, at
        least 1, by default.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    OrderBookResult

    Examples
    --------
    >>> book = simulate_cst([1.0, 0.8, 0.6], 0.9, [0.5, 0.5, 0.5], 500, n_ticks=60, rng=0)
    >>> book.events.shape, bool(np.all(book.spread >= 1))
    ((500,), True)
    """
    lam = np.asarray(lam, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    L = lam.size
    if theta.shape != (L,) or L == 0 or mu < 0 or np.any(lam < 0) or np.any(theta < 0):
        raise ValueError("lam and theta must be non-negative arrays of the same length, and mu non-negative")
    if n_ticks < 2 * L + 2:
        raise ValueError("n_ticks must be at least 2 * len(lam) + 2")
    rng = np.random.default_rng(rng)
    depth0 = np.maximum(1, np.rint(np.divide(lam, theta, out=np.ones(L), where=theta > 0))) if initial_depth is None else np.asarray(initial_depth)
    X = np.zeros(n_ticks, dtype=np.int64)  # signed book: bids < 0, asks > 0
    c = n_ticks // 2
    X[c - L : c] = -depth0[::-1].astype(np.int64)
    X[c : c + L] = depth0.astype(np.int64)

    n = int(n_events)
    E, U = rng.standard_exponential(n), rng.random(n)
    times, events = np.zeros(n + 1), np.zeros(n, dtype=np.int64)
    best_bid, best_ask = np.zeros(n + 1, dtype=np.int64), np.zeros(n + 1, dtype=np.int64)
    bid_size, ask_size = np.zeros(n + 1, dtype=np.int64), np.zeros(n + 1, dtype=np.int64)
    depth = np.zeros(L)
    ticks, d = np.arange(n_ticks), np.arange(1, L + 1)
    theta_far = np.append(theta, theta[-1])

    def quotes() -> tuple[int, int]:
        asks, bids = np.flatnonzero(X > 0), np.flatnonzero(X < 0)
        return (int(bids[-1]) if bids.size else -1), (int(asks[0]) if asks.size else n_ticks)

    pB, pA = quotes()
    for k in range(n + 1):
        best_bid[k], best_ask[k] = pB, pA
        bid_size[k] = -X[pB] if pB >= 0 else 0
        ask_size[k] = X[pA] if pA < n_ticks else 0
        if k == n:
            break
        buy_at, sell_at = pA - d, pB + d
        r_lb = np.where(buy_at >= 0, lam, 0.0)
        r_ls = np.where(sell_at < n_ticks, lam, 0.0)
        r_cb = theta_far[np.clip(pA - ticks, 1, L + 1) - 1] * np.maximum(-X, 0)
        r_ca = theta_far[np.clip(ticks - pB, 1, L + 1) - 1] * np.maximum(X, 0)
        rates = np.concatenate([r_lb, r_ls, [mu if pA < n_ticks else 0.0, mu if pB >= 0 else 0.0], r_cb, r_ca])
        cum = np.cumsum(rates)
        dt = E[k] / cum[-1]
        times[k + 1] = times[k] + dt
        ok_b, ok_a = buy_at >= 0, sell_at < n_ticks
        depth[ok_b] += dt * np.maximum(-X[buy_at[ok_b]], 0)
        depth[ok_a] += dt * np.maximum(X[sell_at[ok_a]], 0)
        j = min(int(np.searchsorted(cum, U[k] * cum[-1], side="right")), rates.size - 1)
        if j < L:
            X[pA - 1 - j] -= 1
            events[k] = 0
        elif j < 2 * L:
            X[pB + 1 + j - L] += 1
            events[k] = 1
        elif j == 2 * L:
            X[pA] -= 1
            events[k] = 2
        elif j == 2 * L + 1:
            X[pB] += 1
            events[k] = 3
        elif j < 2 * L + 2 + n_ticks:
            X[j - 2 * L - 2] += 1
            events[k] = 4
        else:
            X[j - 2 * L - 2 - n_ticks] -= 1
            events[k] = 5
        pB, pA = quotes()
    return OrderBookResult(
        times=times,
        events=events,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_size=bid_size,
        ask_size=ask_size,
        mean_depth=depth / (2.0 * times[-1]),
    )


def cst_mid_up_probability(lam: float, mu: float, theta: float, n_max: int = 50) -> np.ndarray:
    """Probability that the mid price moves up next, from the queues at the best quotes.

    With a one-tick spread, Cont, Stoikov and Talreja's best bid and best
    ask queues are independent birth-death processes: orders join at rate
    :math:`\\lambda = \\lambda(1)` and leave at rate :math:`\\mu + \\theta q`
    (a market order or one of the :math:`q` resting orders cancelling). The
    mid moves up when the ask queue empties first. The probability
    :math:`p(a, b)` of that, from bid queue :math:`a` and ask queue
    :math:`b`, solves the backward equation

    .. math::

       (2\\lambda + 2\\mu + (a + b)\\theta)\\,p(a, b) = \\lambda p(a+1, b) + \\lambda p(a, b+1)
       + (\\mu + a\\theta)\\,p(a-1, b) + (\\mu + b\\theta)\\,p(a, b-1),

    with :math:`p(0, b) = 0` and :math:`p(a, 0) = 1`. Queues are capped at
    `n_max` (no arrivals there), which matters only for queues near the cap.
    The paper computes the same probability by Laplace transforms; the
    linear system is exact up to the cap.

    Parameters
    ----------
    lam : float
        Limit order rate at the best quote, :math:`\\lambda(1)`.
    mu : float
        Market order rate.
    theta : float
        Cancellation rate per order at the best quote, :math:`\\theta(1)`.
    n_max : int
        Largest queue.

    Returns
    -------
    ndarray, shape (n_max + 1, n_max + 1)
        ``p[a, b]`` for bid queue ``a`` and ask queue ``b``; ``p[0, 0]`` is ``nan``.

    Examples
    --------
    >>> p = cst_mid_up_probability(1.0, 0.9, 0.5, n_max=30)
    >>> round(float(p[3, 3]), 12), bool(p[5, 1] > 0.5 > p[1, 5])
    (0.5, True)
    """
    if lam < 0 or mu < 0 or theta < 0 or mu + theta <= 0:
        raise ValueError("need non-negative rates with mu + theta > 0")
    N = int(n_max)
    a, b = (g.ravel() for g in np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij"))
    idx = (a - 1) * N + (b - 1)
    up_a, up_b = np.where(a < N, lam, 0.0), np.where(b < N, lam, 0.0)
    down_a, down_b = mu + a * theta, mu + b * theta
    rows, cols, vals = [idx], [idx], [up_a + up_b + down_a + down_b]
    for mask, shift, rate in [(a < N, N, up_a), (b < N, 1, up_b), (a > 1, -N, down_a), (b > 1, -1, down_b)]:
        rows.append(idx[mask])
        cols.append(idx[mask] + shift)
        vals.append(-rate[mask])
    M = scipy.sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N))
    rhs = np.where(b == 1, down_b, 0.0)  # the ask empties: p(a, 0) = 1
    p = np.full((N + 1, N + 1), np.nan)
    p[1:, 1:] = scipy.sparse.linalg.spsolve(M.tocsc(), rhs).reshape(N, N)
    p[0, 1:], p[1:, 0] = 0.0, 1.0
    return p


def order_flow_imbalance(bid: ArrayLike, bid_size: ArrayLike, ask: ArrayLike, ask_size: ArrayLike) -> np.ndarray:
    """Cont, Kukanov and Stoikov's order-flow imbalance of each quote update.

    .. math::

       e_n = q^B_n \\mathbf 1_{P^B_n \\ge P^B_{n-1}} - q^B_{n-1} \\mathbf 1_{P^B_n \\le P^B_{n-1}}
       - q^A_n \\mathbf 1_{P^A_n \\le P^A_{n-1}} + q^A_{n-1} \\mathbf 1_{P^A_n \\ge P^A_{n-1}}

    adds the demand that arrives at the best bid (a larger queue, or a
    higher bid) and subtracts the supply that arrives at the best ask, so
    limit orders, cancellations and market orders all count. Summed over an
    interval it is that interval's OFI.

    Parameters
    ----------
    bid, bid_size, ask, ask_size : array_like, shape (n,)
        Best quotes and the sizes queued there after each update.

    Returns
    -------
    ndarray, shape (n - 1,)

    Examples
    --------
    Two bids join the queue, a market buy takes one ask, then the bid rises.

    >>> order_flow_imbalance([10, 10, 10, 11], [5, 7, 7, 1], [12, 12, 12, 12], [4, 4, 3, 3]).tolist()
    [2.0, 1.0, 1.0]
    """
    pb, qb = np.asarray(bid, dtype=np.float64), np.asarray(bid_size, dtype=np.float64)
    pa, qa = np.asarray(ask, dtype=np.float64), np.asarray(ask_size, dtype=np.float64)
    if not pb.shape == qb.shape == pa.shape == qa.shape or pb.ndim != 1:
        raise ValueError("bid, bid_size, ask and ask_size must be 1-D arrays of the same length")
    dpb, dpa = np.diff(pb), np.diff(pa)
    return qb[1:] * (dpb >= 0) - qb[:-1] * (dpb <= 0) - qa[1:] * (dpa <= 0) + qa[:-1] * (dpa >= 0)


def fit_ofi_impact(price_change: ArrayLike, ofi: ArrayLike) -> OLSResult:
    """Regress price changes on order-flow imbalance, :math:`\\Delta P_k = \\alpha + \\beta\\,\\mathrm{OFI}_k + \\varepsilon_k`.

    Cont, Kukanov and Stoikov found :math:`\\beta` close to inversely
    proportional to the depth at the best quotes and :math:`R^2` around 65%
    over ten-second intervals.

    Parameters
    ----------
    price_change : array_like, shape (n,)
        Mid-price change over each interval.
    ofi : array_like, shape (n,)
        Order-flow imbalance over the same intervals.

    Returns
    -------
    OLSResult
        ``coef = [alpha, beta]``.

    Examples
    --------
    >>> fit_ofi_impact([0.5, -1.0, 1.5], [1.0, -2.0, 3.0]).coef.round(12).tolist()
    [0.0, 0.5]
    """
    dp = np.asarray(price_change, dtype=np.float64)
    e = np.asarray(ofi, dtype=np.float64)
    if dp.shape != e.shape or dp.ndim != 1:
        raise ValueError("price_change and ofi must be 1-D arrays of the same length")
    return ols(dp, np.column_stack([np.ones(e.size), e]))

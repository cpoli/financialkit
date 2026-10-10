"""The stochastic discount factor: Harrison and Kreps's state prices and
the Hansen-Jagannathan bound.

Harrison and Kreps (1979) proved that a market has no arbitrage exactly
when there is a strictly positive linear pricing rule: in a finite-state
market, a positive price for each state's Arrow-Debreu security, or
equivalently a positive stochastic discount factor :math:`m` with
:math:`p = E[m x]` for every payoff :math:`x`. Normalised, the state
prices are the risk-neutral probabilities. When the market is complete
they are unique; otherwise every positive solution prices the traded
assets, and a new claim has an interval of arbitrage-free prices.

Hansen and Jagannathan (1991) asked what the returns alone say about
:math:`m`. For any :math:`m` that prices the gross returns
:math:`E[m R] = 1`, the volatility :math:`\\sigma(m)` is at least the
bound below, which for a given mean is attained by a combination of
the returns themselves. Since :math:`\\sigma(m)/E(m)` must exceed the
largest Sharpe ratio on offer, a US equity Sharpe ratio near 0.4
requires an extremely volatile discount factor.

References: J. M. Harrison and D. M. Kreps, "Martingales and Arbitrage
in Multiperiod Securities Markets," *J. Econ. Theory* 20 (1979)
381-408; L. P. Hansen and R. Jagannathan, "Implications of Security
Market Data for Models of Dynamic Economies," *J. Polit. Econ.* 99
(1991) 225-262; J. H. Cochrane, *Asset Pricing*, rev. ed., Princeton
(2005), ch. 4-5.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from financialkit.asset_pricing.core.base import HJBound

__all__ = ["FiniteStateMarket", "hansen_jagannathan_bound", "max_sharpe_ratio"]


class FiniteStateMarket:
    """A one-period market with finitely many states, traded payoffs and their prices.

    Parameters
    ----------
    payoffs : array_like, shape (S, N)
        Payoff of each of ``N`` securities in each of ``S`` states.
    prices : array_like, shape (N,)
        Today's prices.
    probabilities : array_like, shape (S,), optional
        Physical state probabilities, needed only for the SDF; equal by default.

    Examples
    --------
    A bond paying 1 and a stock paying 2 or 0.5, priced 0.95 and 1:

    >>> mkt = FiniteStateMarket([[1.0, 2.0], [1.0, 0.5]], [0.95, 1.0])
    >>> mkt.state_prices().round(4).tolist(), mkt.is_complete, mkt.is_arbitrage_free()
    ([0.35, 0.6], True, True)
    """

    def __init__(self, payoffs: ArrayLike, prices: ArrayLike, probabilities: Optional[ArrayLike] = None):
        self.payoffs = np.atleast_2d(np.asarray(payoffs, dtype=np.float64))
        self.prices = np.asarray(prices, dtype=np.float64)
        S, N = self.payoffs.shape
        if self.prices.shape != (N,):
            raise ValueError("prices must have one entry per payoff column")
        self.probabilities = np.full(S, 1.0 / S) if probabilities is None else np.asarray(probabilities, dtype=np.float64)

    @property
    def is_complete(self) -> bool:
        """bool: Whether the payoffs span every state (rank ``S``)."""
        return bool(np.linalg.matrix_rank(self.payoffs) == self.payoffs.shape[0])

    def _max_min_state_price(self) -> tuple[float, np.ndarray]:
        """Maximise ``t`` subject to ``X'q = p``, ``q >= t``, ``t <= 1``."""
        S = self.payoffs.shape[0]
        c = np.zeros(S + 1)
        c[-1] = -1.0
        A_eq = np.hstack([self.payoffs.T, np.zeros((self.payoffs.shape[1], 1))])
        A_ub = np.hstack([-np.eye(S), np.ones((S, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(S), A_eq=A_eq, b_eq=self.prices, bounds=[(None, None)] * S + [(None, 1.0)])
        if res.status != 0:
            return -np.inf, np.full(S, np.nan)
        return float(res.x[-1]), res.x[:S]

    def is_arbitrage_free(self, tol: float = 1e-10) -> bool:
        """Whether a strictly positive state-price vector exists (Harrison-Kreps)."""
        return self._max_min_state_price()[0] > tol

    def state_prices(self) -> np.ndarray:
        """A strictly positive state-price vector, unique when the market is complete.

        Returns
        -------
        ndarray, shape (S,)

        Raises
        ------
        ValueError
            If the market admits an arbitrage.
        """
        if not self.is_arbitrage_free():
            raise ValueError("the market admits an arbitrage, so no positive state prices exist")
        if self.is_complete:
            q, *_ = np.linalg.lstsq(self.payoffs.T, self.prices, rcond=None)
            return q
        return self._max_min_state_price()[1]

    def risk_free_return(self) -> float:
        """Gross return of the riskless payoff, ``1 / sum(q)``."""
        return float(1.0 / self.state_prices().sum())

    def risk_neutral_probabilities(self) -> np.ndarray:
        """State prices normalised to sum to one."""
        q = self.state_prices()
        return q / q.sum()

    def sdf(self) -> np.ndarray:
        """Stochastic discount factor in each state, :math:`m_s = q_s / \\pi_s`."""
        return self.state_prices() / self.probabilities

    def arbitrage_portfolio(self) -> np.ndarray:
        """Holdings that cost nothing or less and pay something and never lose, if any exist.

        Solves a linear program over holdings ``theta`` with payoffs
        ``X theta >= 0`` and cost ``p @ theta <= 0``, maximising the total
        payoff minus the cost within a unit box. Returns zeros when the
        market is arbitrage free.

        Returns
        -------
        ndarray, shape (N,)
        """
        X, p = self.payoffs, self.prices
        N = X.shape[1]
        A_ub = np.vstack([-X, p[None, :]])
        res = linprog(-(X.sum(axis=0) - p), A_ub=A_ub, b_ub=np.zeros(X.shape[0] + 1), bounds=[(-1.0, 1.0)] * N)
        theta = res.x if res.status == 0 else np.zeros(N)
        return np.where(np.abs(theta) < 1e-12, 0.0, theta)

    def price_bounds(self, claim: ArrayLike) -> tuple[float, float]:
        """Lowest and highest arbitrage-free price of a new payoff.

        The extremes of :math:`q' x` over nonnegative state prices that
        price the traded assets. The bounds coincide in a complete market.

        Parameters
        ----------
        claim : array_like, shape (S,)

        Returns
        -------
        tuple of float
            ``(low, high)``.
        """
        x = np.asarray(claim, dtype=np.float64)
        S = x.size
        kw = dict(A_eq=self.payoffs.T, b_eq=self.prices, bounds=[(0.0, None)] * S)
        lo = linprog(x, **kw)
        hi = linprog(-x, **kw)
        return float(lo.fun), float(-hi.fun)


def hansen_jagannathan_bound(mean_returns: ArrayLike, cov: ArrayLike, sdf_means: ArrayLike) -> HJBound:
    """The smallest volatility of a discount factor that prices a set of gross returns.

    For :math:`E[m R] = 1` with :math:`E[m] = v`,
    :math:`\\sigma^2(m) \\ge (\\mathbf 1 - v \\mu)' \\Sigma^{-1} (\\mathbf 1 - v \\mu)`,
    attained by :math:`m^* = v + (\\mathbf 1 - v\\mu)'\\Sigma^{-1}(R - \\mu)`.

    Parameters
    ----------
    mean_returns : array_like, shape (N,)
        Expected gross returns of risky assets.
    cov : array_like, shape (N, N)
        Their covariance, nonsingular.
    sdf_means : array_like, shape (M,)
        Values of :math:`E[m]`, the inverse of a candidate riskless return.

    Returns
    -------
    HJBound

    Examples
    --------
    One asset: the bound is :math:`|1 - v\\mu| / \\sigma`.

    >>> hansen_jagannathan_bound([1.07], [[0.0289]], [0.99]).sdf_stds.round(4).tolist()
    [0.3488]
    """
    mu = np.asarray(mean_returns, dtype=np.float64)
    C = np.asarray(cov, dtype=np.float64)
    v = np.atleast_1d(np.asarray(sdf_means, dtype=np.float64))
    a = 1.0 - v[:, None] * mu[None, :]
    var = np.einsum("ij,ij->i", a, np.linalg.solve(C, a.T).T)
    return HJBound(sdf_means=v, sdf_stds=np.sqrt(var))


def max_sharpe_ratio(expected_excess_returns: ArrayLike, cov: ArrayLike) -> float:
    """The largest Sharpe ratio of any portfolio, :math:`\\sqrt{\\mu' \\Sigma^{-1} \\mu}`.

    Hansen and Jagannathan's bound in its simplest form:
    :math:`\\sigma(m)/E(m)` is at least this number.

    Parameters
    ----------
    expected_excess_returns : array_like, shape (N,)
    cov : array_like, shape (N, N)

    Returns
    -------
    float

    Examples
    --------
    >>> round(max_sharpe_ratio([0.06], [[0.04]]), 12)
    0.3
    """
    mu = np.asarray(expected_excess_returns, dtype=np.float64)
    return float(np.sqrt(mu @ np.linalg.solve(np.atleast_2d(np.asarray(cov, dtype=np.float64)), mu)))

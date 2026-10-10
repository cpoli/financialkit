"""Result containers for financialkit.exotic_options.

The closed-form pricers (barriers, binaries, lookbacks, Asians,
compounds, choosers, forward starts, exchange, rainbow and quanto
options) return plain floats or arrays and broadcast over their inputs.
Monte Carlo pricers return the
:class:`~financialkit.monte_carlo.core.base.MonteCarloResult` family from
:mod:`financialkit.monte_carlo`, so their standard errors and confidence
intervals compare on the same footing as every other estimator in the
package. Two families carry more structure: a Longstaff-Schwartz
valuation keeps its exercise boundary, and a variance-swap replication
keeps its strip of option weights. These are dataclasses, never bare
tuples, so the visualizers have a stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from financialkit.monte_carlo.core.base import MonteCarloResult

__all__ = ["LongstaffSchwartzResult", "VarianceSwapResult"]


@dataclass
class LongstaffSchwartzResult(MonteCarloResult):
    """An American (Bermudan) option value by least-squares Monte Carlo (Longstaff-Schwartz 2001).

    :attr:`estimate` is the mean discounted cash flow of the regression
    exercise policy and :attr:`std_error` its standard error; the policy is
    fitted on the same paths, so the estimate carries a small in-sample
    bias.
    """

    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (n_exercise + 1,): The exercise dates, starting at 0."""

    exercise_boundary: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (n_exercise + 1,): For a put, the highest price in
    the simulated in-the-money range at which the fitted policy exercises
    on each date (for a call, the lowest); ``nan`` where it never does,
    and the strike at expiry."""

    exercise_probability: float = 0.0
    """float: Fraction of paths exercised before maturity."""


@dataclass
class VarianceSwapResult:
    """The fair strike of a variance swap from a strip of out-of-the-money options (Demeterfi et al. 1999).

    The fair variance is :math:`\\sum_i w_i Q(K_i) - (F/K_0 - 1)^2 / T`,
    with :math:`Q` the out-of-the-money option price at strike
    :math:`K_i` and :math:`w_i = 2 e^{rT} \\Delta K_i / (T K_i^2)`.
    """

    fair_variance: float
    """float: Annualized fair variance strike :math:`K_{\\mathrm{var}}`."""

    strikes: np.ndarray
    """ndarray, shape (n,): Strikes of the replicating strip."""

    weights: np.ndarray
    """ndarray, shape (n,): Number of options held per unit of variance notional, :math:`w_i`."""

    option_prices: np.ndarray
    """ndarray, shape (n,): Out-of-the-money option prices :math:`Q(K_i)` (the average of put and call at :math:`K_0`)."""

    forward: float
    """float: Forward price :math:`F = S e^{(r - q) T}`."""

    K0: float
    """float: Largest strike at or below the forward, separating puts from calls."""

    T: float
    """float: Maturity in years."""

    @property
    def contributions(self) -> np.ndarray:
        """ndarray: Each option's share :math:`w_i Q(K_i)` of the fair variance."""
        return self.weights * self.option_prices

    @property
    def vol_strike(self) -> float:
        """float: The volatility :math:`\\sqrt{K_{\\mathrm{var}}}` quoted for the swap."""
        return float(np.sqrt(self.fair_variance))

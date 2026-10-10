"""Monte Carlo Greeks under Black-Scholes, without finite-difference bumps.

* :func:`likelihood_ratio_greeks` -- Glynn (1987) and Reiman-Weiss
  (1989): differentiate the density of :math:`S_T`, not the payoff, so
  any payoff (including digitals) works, at the price of a noisier weight.
* :func:`pathwise_greeks` -- Broadie and Glasserman (1996): differentiate
  the payoff along each path; low variance, but only for payoffs that are
  continuous in the parameter.

With :math:`S_T = S \\exp((r - q - \\sigma^2/2)T + \\sigma\\sqrt{T} Z)` the
likelihood-ratio weights are

.. math::

    \\Delta: \\frac{Z}{S\\sigma\\sqrt T}, \\qquad
    \\mathcal V: \\frac{Z^2 - 1}{\\sigma} - Z\\sqrt T, \\qquad
    \\Gamma: \\frac{Z^2 - 1}{S^2\\sigma^2 T} - \\frac{Z}{S^2\\sigma\\sqrt T},

and the pathwise derivatives of a call are
:math:`e^{-rT}\\mathbf 1\\{S_T > K\\} S_T / S` and
:math:`e^{-rT}\\mathbf 1\\{S_T > K\\} S_T (\\sqrt T Z - \\sigma T)`.

References: P. W. Glynn, "Likelihood Ratio Gradient Estimation: An
Overview," *Proc. 1987 Winter Simulation Conference*, 366-375; M. Broadie
and P. Glasserman, "Estimating Security Price Derivatives Using
Simulation," *Management Science* 42 (1996) 269-285; Glasserman (2004),
Chapter 7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.integrators import RNGLike
from financialkit.monte_carlo.core.base import MonteCarloGreeks
from financialkit.monte_carlo.utils.numerics import terminal_prices, vanilla_payoff

__all__ = ["likelihood_ratio_greeks", "pathwise_greeks"]


def _mean_se(y: np.ndarray) -> tuple[float, float]:
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size))


def _draw(n_paths: int, rng: RNGLike, z: ArrayLike | None) -> np.ndarray:
    return np.random.default_rng(rng).standard_normal(int(n_paths)) if z is None else np.asarray(z, dtype=np.float64).ravel()


def likelihood_ratio_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    digital: bool = False,
    n_paths: int = 100_000,
    rng: RNGLike = None,
    z: ArrayLike | None = None,
) -> MonteCarloGreeks:
    """Price, delta, vega and gamma by the likelihood-ratio (score-function) method (Glynn 1987).

    Each Greek is :math:`E[e^{-rT}\\,\\text{payoff}(S_T)\\,w(Z)]` with the
    score weight :math:`w` from the module docstring, estimated on the
    same paths as the price.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, risk-free rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    digital : bool
        Price a cash-or-nothing digital paying 1 instead of a vanilla.
    n_paths : int
        Number of terminal prices; ignored if `z` is given.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    z : array_like, optional
        Standard normals to use, e.g. common random numbers with
        :func:`pathwise_greeks`.

    Returns
    -------
    MonteCarloGreeks

    Examples
    --------
    Black-Scholes delta of the Hull Example 15.6 call is 0.7791:

    >>> g = likelihood_ratio_greeks(42.0, 40.0, 0.5, 0.10, 0.20, n_paths=400_000, rng=0)
    >>> abs(g.delta - 0.7791) < 4 * g.delta_se
    True
    """
    z = _draw(n_paths, rng, z)
    rt = np.sqrt(T)
    V = np.exp(-r * T) * vanilla_payoff(terminal_prices(S, T, r, sigma, z, q), K, option_type, digital)
    price, price_se = _mean_se(V)
    delta, delta_se = _mean_se(V * z / (S * sigma * rt))
    vega, vega_se = _mean_se(V * ((z**2 - 1.0) / sigma - z * rt))
    gamma, gamma_se = _mean_se(V * ((z**2 - 1.0) / (S**2 * sigma**2 * T) - z / (S**2 * sigma * rt)))
    return MonteCarloGreeks(price, delta, vega, gamma, price_se, delta_se, vega_se, gamma_se, z.size, "likelihood_ratio")


def pathwise_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    rng: RNGLike = None,
    z: ArrayLike | None = None,
) -> MonteCarloGreeks:
    """Price, delta and vega of a vanilla option by pathwise differentiation (Broadie-Glasserman 1996).

    The payoff is Lipschitz in :math:`S` and :math:`\\sigma`, so the
    derivative and the expectation commute; its derivative jumps at the
    strike, so there is no pathwise gamma (returned as ``nan``). Digital
    payoffs are discontinuous and are refused.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, risk-free rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    n_paths : int
        Number of terminal prices; ignored if `z` is given.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    z : array_like, optional
        Standard normals to use.

    Returns
    -------
    MonteCarloGreeks

    Examples
    --------
    >>> import numpy as np
    >>> g = pathwise_greeks(42.0, 40.0, 0.5, 0.10, 0.20, n_paths=100_000, rng=0)
    >>> abs(g.delta - 0.7791) < 4 * g.delta_se, bool(np.isnan(g.gamma))
    (True, True)
    """
    z = _draw(n_paths, rng, z)
    S_T = terminal_prices(S, T, r, sigma, z, q)
    disc = np.exp(-r * T)
    V = disc * vanilla_payoff(S_T, K, option_type)
    slope = disc * (vanilla_payoff(S_T, K, option_type, digital=True) * (1.0 if option_type == "call" else -1.0))
    price, price_se = _mean_se(V)
    delta, delta_se = _mean_se(slope * S_T / S)
    vega, vega_se = _mean_se(slope * S_T * (np.sqrt(T) * z - sigma * T))
    nan = float("nan")
    return MonteCarloGreeks(price, delta, vega, nan, price_se, delta_se, vega_se, nan, z.size, "pathwise")

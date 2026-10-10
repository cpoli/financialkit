r"""The explicit scheme in log-spot as a trinomial tree.

In :math:`x = \ln S` the Black-Scholes equation has constant coefficients,
:math:`V_\tau = \tfrac12\sigma^2 V_{xx} + \nu V_x - rV` with
:math:`\nu = r - q - \tfrac12\sigma^2`. Its explicit discretisation, with
the discount taken implicitly as in Hull, reads

.. math:: V_j^{n+1} = \frac{1}{1 + r\Delta t}\bigl(p_d V_{j-1}^n + p_m V_j^n + p_u V_{j+1}^n\bigr),
          \quad p_{u,d} = \Delta t\Bigl(\frac{\sigma^2}{2\Delta x^2} \pm \frac{\nu}{2\Delta x}\Bigr),
          \quad p_m = 1 - \frac{\sigma^2\Delta t}{\Delta x^2}.

Brennan and Schwartz (1978) pointed out that the weights sum to one and
match the mean :math:`\nu\Delta t` and variance :math:`\sigma^2\Delta t` of
the log-price increment: the explicit scheme *is* a trinomial tree, and
it is stable exactly when the weights are probabilities. With
:math:`\Delta x = \sigma\sqrt{3\Delta t}` the middle weight is 2/3, Hull's
standard trinomial tree.

References: M. J. Brennan and E. S. Schwartz, *J. Financial and
Quantitative Analysis* 13 (1978) 461-474; Hull, *Options, Futures, and
Other Derivatives*, 10th ed., Ch. 21.
"""

from __future__ import annotations

import numpy as np

from financialkit.options.utils.common import option_sign, payoff
from financialkit.pde_methods.core.base import FDResult, TrinomialProbabilities
from financialkit.pde_methods.systems.stability import diffusion_number
from financialkit.pde_methods.utils.operators import discounted_intrinsic, interpolate_at

__all__ = ["explicit_trinomial_probabilities", "log_explicit_price"]


def explicit_trinomial_probabilities(sigma: float, r: float, dt: float, dx: float, q: float = 0.0) -> TrinomialProbabilities:
    """Weights of the explicit log-spot scheme.

    Parameters
    ----------
    sigma, r : float
        Volatility and rate.
    dt, dx : float
        Time step and log-spot step.
    q : float
        Dividend yield.

    Returns
    -------
    TrinomialProbabilities

    Examples
    --------
    With :math:`\\Delta x = \\sigma\\sqrt{3\\Delta t}` the middle weight is 2/3:

    >>> p = explicit_trinomial_probabilities(0.2, 0.05, 0.01, 0.2 * np.sqrt(0.03))
    >>> round(p.p_mid, 12), round(p.total, 12), p.is_probability
    (0.666666666667, 1.0, True)
    """
    nu = r - q - 0.5 * sigma**2
    a, b = sigma**2 * dt / (2.0 * dx**2), nu * dt / (2.0 * dx)
    return TrinomialProbabilities(p_down=float(a - b), p_mid=float(1.0 - 2.0 * a), p_up=float(a + b), discount=float(1.0 / (1.0 + r * dt)))


def log_explicit_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    american: bool = False,
    n_time: int = 200,
    dx: float | None = None,
) -> FDResult:
    """Value a vanilla option by the explicit log-spot scheme, a trinomial tree on a fixed grid.

    The grid has :math:`2N + 1` nodes centred on the spot, so the value at
    the spot never feels the boundaries: it equals the root of the
    trinomial tree with the same weights.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    american : bool
        Exercise early whenever the payoff exceeds the continuation value.
    n_time : int
        Number of time steps :math:`N`.
    dx : float, optional
        Log-spot step (default :math:`\\sigma\\sqrt{3\\Delta t}`).

    Returns
    -------
    FDResult
        ``stability`` holds the diffusion number :math:`\\sigma^2\\Delta t/\\Delta x^2`.

    Examples
    --------
    >>> res = log_explicit_price(100.0, 100.0, 1.0, 0.05, 0.2, n_time=500)
    >>> round(res.price, 2)  # Black-Scholes: 10.45
    10.45
    """
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        raise ValueError("S, K, T and sigma must all be positive")
    option_sign(option_type)
    n = int(n_time)
    dt = T / n
    dx = sigma * np.sqrt(3.0 * dt) if dx is None else float(dx)
    p = explicit_trinomial_probabilities(sigma, r, dt, dx, q)
    grid = S * np.exp(dx * np.arange(-n, n + 1))
    exercise = np.asarray(payoff(grid, K, option_type))
    surface = np.empty((n + 1, grid.size))
    surface[0] = V = exercise.copy()
    for k in range(1, n + 1):
        new = np.empty_like(V)
        new[1:-1] = p.discount * (p.p_down * V[:-2] + p.p_mid * V[1:-1] + p.p_up * V[2:])
        new[[0, -1]] = discounted_intrinsic(grid[[0, -1]], K, k * dt, r, q, option_type)
        V = np.maximum(new, exercise) if american else new
        surface[k] = V
    _, delta, gamma = interpolate_at(grid, V, S)
    return FDResult(
        price=float(V[n]),
        delta=delta,
        gamma=gamma,
        S=grid,
        V=V,
        method="explicit_log",
        times=np.linspace(0.0, T, n + 1),
        surface=surface[::-1].copy(),
        stability=diffusion_number(sigma, dt, dx),
    )

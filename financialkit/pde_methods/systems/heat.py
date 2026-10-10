r"""The Black-Scholes equation as the heat equation.

Black and Scholes (1973) solved their equation by changing variables
until it became the heat equation, whose solution was known. With
:math:`x = \ln(S/K)`, :math:`\tau = \tfrac12 \sigma^2 (T - t)`,
:math:`k = 2r/\sigma^2`, :math:`k_q = 2(r - q)/\sigma^2` and

.. math:: V(S, t) = K e^{\alpha x + \beta\tau} u(x, \tau), \qquad
          \alpha = -\tfrac12 (k_q - 1), \qquad \beta = -\tfrac14 (k_q - 1)^2 - k,

the price :math:`V` solves Black-Scholes exactly when :math:`u` solves
:math:`u_\tau = u_{xx}` with :math:`u(x, 0) = e^{-\alpha x}\,\text{payoff}(K e^x)/K`.
Numerically the transform trades a variable-coefficient equation for a
constant-coefficient one on a uniform grid in :math:`\ln S`, where the
stability and accuracy theory of the heat equation applies verbatim.

References: F. Black and M. Scholes, *J. Political Economy* 81 (1973)
637-654; P. Wilmott, S. Howison and J. Dewynne, *The Mathematics of
Financial Derivatives* (Cambridge, 1995), Ch. 5.
"""

from __future__ import annotations

import numpy as np

from financialkit.options.utils.common import option_sign, payoff
from financialkit.pde_methods.core.base import HeatEquationResult
from financialkit.pde_methods.utils.operators import discounted_intrinsic, interpolate_at
from financialkit.pde_methods.utils.stepping import theta_march

__all__ = ["heat_equation_exponents", "heat_equation_price"]


def heat_equation_exponents(r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    r"""The exponents :math:`(\alpha, \beta)` that turn Black-Scholes into :math:`u_\tau = u_{xx}`.

    Parameters
    ----------
    r, sigma, q : float

    Returns
    -------
    alpha, beta : float

    Examples
    --------
    With :math:`r = \sigma^2/2` the drift term vanishes, :math:`k = 1`:

    >>> [round(v, 12) for v in heat_equation_exponents(0.02, 0.2)]
    [0.0, -1.0]
    """
    k, kq = 2.0 * r / sigma**2, 2.0 * (r - q) / sigma**2
    alpha = -0.5 * (kq - 1.0)
    return float(alpha), float(-0.25 * (kq - 1.0) ** 2 - k)


def heat_equation_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_space: int = 400,
    n_time: int = 100,
    theta: float = 0.5,
    rannacher_steps: int = 2,
    width: float = 6.0,
) -> HeatEquationResult:
    """Price a European option by solving the heat equation and transforming back.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    n_space : int
        Number of intervals of the uniform :math:`x` grid (rounded up to
        even, so the spot is a node).
    n_time : int
        Number of steps in :math:`\\tau`.
    theta : float
        Implicitness of the heat-equation scheme.
    rannacher_steps : int
        Implicit start steps (see :func:`~financialkit.pde_methods.utils.stepping.theta_march`).
    width : float
        Half-width of the grid in standard deviations :math:`\\sigma\\sqrt{T}`,
        added to the distance between spot and strike.

    Returns
    -------
    HeatEquationResult

    Examples
    --------
    >>> res = heat_equation_price(100.0, 100.0, 1.0, 0.05, 0.2)
    >>> round(res.price, 2)  # Black-Scholes: 10.45
    10.45
    """
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        raise ValueError("S, K, T and sigma must all be positive")
    option_sign(option_type)
    n = int(n_space) + int(n_space) % 2
    alpha, beta = heat_equation_exponents(r, sigma, q)
    x0 = np.log(S / K)
    half = abs(x0) + width * sigma * np.sqrt(T)
    x = x0 + np.linspace(-half, half, n + 1)
    dx = x[1] - x[0]
    u0 = np.exp(-alpha * x) * np.asarray(payoff(K * np.exp(x), K, option_type)) / K
    edges = x[[0, -1]]

    def boundary(tau: float) -> tuple[float, float]:
        v = discounted_intrinsic(K * np.exp(edges), K, 2.0 * tau / sigma**2, r, q, option_type) / K
        u = np.exp(-alpha * edges - beta * tau) * v
        return float(u[0]), float(u[1])

    m = n - 1
    lower, upper, diag = np.full(m, 1.0 / dx**2), np.full(m, 1.0 / dx**2), np.full(m, -2.0 / dx**2)
    tau_end = 0.5 * sigma**2 * T
    u, _ = theta_march(u0, lower, diag, upper, tau_end, n_time, boundary, theta=theta, rannacher_steps=rannacher_steps)
    tau = np.linspace(0.0, tau_end, n_time + 1)
    values = K * np.exp(alpha * x[None, :] + beta * tau[:, None]) * u
    grid = K * np.exp(x)
    _, delta, gamma = interpolate_at(grid, values[-1], S)
    return HeatEquationResult(
        price=float(values[-1, n // 2]),
        delta=delta,
        gamma=gamma,
        S=grid,
        V=values[-1],
        method="heat_equation",
        times=np.linspace(0.0, T, n_time + 1),
        surface=values[::-1].copy(),
        x=x,
        tau=tau,
        u=u,
        alpha=alpha,
        beta=beta,
    )

"""European options on a stock paying discrete cash dividends.

The stock follows geometric Brownian motion between dividend dates and
drops by the cash amount :math:`D_i` at each date :math:`t_i`, but never below
zero. The textbook fixes (Hull's escrowed model, which subtracts the
present value of the dividends from the spot, or adding it to the strike)
change the volatility of what is being modelled and misprice options,
most visibly for dividends close to expiry. Haug, Haug and Lewis (2003)
went back to the definition: just after the first dividend, the option is
worth a European option on the reduced stock, so its value today is the
risk-neutral expectation

.. math:: V_0 = e^{-r t_1}\\, \\mathbb{E}\\left[V\\big(\\max(S_{t_1} - D_1, 0), t_1\\big)\\right],

one integral per dividend, the last one over the Black-Scholes price.
Here each integral is Gauss-Legendre quadrature over the standard normal
driving the log-normal step, split where the integrand has a kink (the
liquidation point :math:`S_{t_i^-} = D_i` and the strike), and nested for
several dividends.

References: E. G. Haug, J. Haug and A. Lewis, "Back to Basics: a New
Approach to the Discrete Dividend Problem," *Wilmott Magazine*
(September 2003); Haug, *The Complete Guide to Option Pricing Formulas*,
2nd ed., Section 9.1; Hull, *Options, Futures, and Other Derivatives*,
10th ed., Section 15.12.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit._validation import nonnegative, positive
from financialkit._validation import option_type as check_option_type
from financialkit.options.systems.closed_form import black_scholes_price
from financialkit.options.utils.common import as_output

__all__ = ["discrete_dividend_price"]

_Z_MAX = 9.0  # the normal mass beyond |z| = 9 is 2e-19
_Z_FIXED = np.array([-_Z_MAX, -4.5, 0.0, 4.5, _Z_MAX])


def discrete_dividend_price(
    S: ArrayLike,
    K: ArrayLike,
    T: float,
    r: float,
    sigma: float,
    dividend_times: ArrayLike,
    dividends: ArrayLike,
    option_type: str = "call",
    n_nodes: int = 16,
) -> np.ndarray | float:
    r"""Haug-Haug-Lewis (2003) price of a European option with discrete cash dividends.

    Integrates the price just after each dividend over the log-normal
    distribution of the spot just before it, by Gauss-Legendre quadrature
    on six segments of :math:`z \in [-9, 9]`, split at the kinks and at
    :math:`z = 0, \pm 4.5`, with ``n_nodes`` nodes each (the cost is
    ``(6 n_nodes) ** m`` for ``m`` dividends). The spot after a dividend is
    :math:`\max(S_{t_i^-} - D_i, 0)`, so a stock that cannot pay is
    liquidated.

    Parameters
    ----------
    S, K : array_like
        Spot and strike, broadcast together.
    T, r, sigma : float
        Maturity (years), risk-free rate and volatility of the stock between
        dividends.
    dividend_times : array_like, shape (m,)
        Ex-dividend times in ``(0, T]``, increasing. Times after ``T`` are
        ignored.
    dividends : array_like, shape (m,)
        Non-negative cash amounts.
    option_type : {"call", "put"}
    n_nodes : int
        Gauss-Legendre nodes per segment.

    Returns
    -------
    float or ndarray

    Examples
    --------
    A dividend paid at expiry turns a call struck at ``K`` into a call
    struck at ``K + D``:

    >>> import numpy as np
    >>> c = discrete_dividend_price(100.0, 100.0, 1.0, 0.05, 0.3, [1.0], [4.0])
    >>> bool(np.isclose(c, black_scholes_price(100.0, 104.0, 1.0, 0.05, 0.3)))
    True
    """
    check_option_type(option_type)
    positive(T=T, sigma=sigma)
    times, amounts = np.atleast_1d(np.asarray(dividend_times, dtype=np.float64)), np.atleast_1d(np.asarray(dividends, dtype=np.float64))
    if times.shape != amounts.shape:
        raise ValueError(f"dividend_times and dividends must have the same shape, got {times.shape} and {amounts.shape}")
    positive(dividend_times=times)
    nonnegative(dividends=amounts)
    if np.any(np.diff(times) < 0):
        raise ValueError("dividend_times must be increasing")
    keep = times <= T
    times, amounts = times[keep], amounts[keep]
    S, K = np.broadcast_arrays(np.asarray(S, dtype=np.float64), np.asarray(K, dtype=np.float64))
    nonnegative(S=S, K=K)
    shape = S.shape
    u, wu = np.polynomial.legendre.leggauss(n_nodes)
    n_total = 6 * n_nodes

    def value(spot: np.ndarray, strike: np.ndarray, i: int, t_prev: float) -> np.ndarray:
        if i == times.size:
            return np.asarray(black_scholes_price(spot, strike, T - t_prev, r, sigma, 0.0, option_type))
        dt = times[i] - t_prev
        mu, sd = (r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
        with np.errstate(divide="ignore"):
            # z at which the stock just covers the dividend, and at which what is left reaches the strike
            kinks = np.sort(np.stack([np.log(amounts[i] / spot), np.log((amounts[i] + strike) / spot)], axis=1) - mu, axis=1) / sd
        fixed = np.broadcast_to(_Z_FIXED, (spot.size, _Z_FIXED.size))
        edges = np.sort(np.column_stack([fixed, np.clip(kinks, -_Z_MAX, _Z_MAX)]), axis=1)
        half = 0.5 * np.diff(edges, axis=1)  # (n, 6) half-widths
        z = ((edges[:, :-1] + half)[:, :, None] + half[:, :, None] * u).reshape(-1, n_total)
        w = (half[:, :, None] * wu).reshape(-1, n_total) * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        after = np.maximum(spot[:, None] * np.exp(mu + sd * z) - amounts[i], 0.0)
        later = value(after.ravel(), np.repeat(strike, n_total), i + 1, times[i]).reshape(-1, n_total)
        return np.exp(-r * dt) * np.sum(later * w, axis=1)

    return as_output(value(S.ravel(), K.ravel(), 0, 0.0).reshape(shape))

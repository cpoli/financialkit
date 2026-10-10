"""Discrete delta hedging: the Black-Scholes replication argument, simulated.

Black and Scholes (1973) priced an option by holding :math:`\\Delta` shares
against it and rebalancing continuously, which removes all risk. Here a
dealer sells the option at its Black-Scholes price, holds the
Black-Scholes delta in the stock, finances it in a bank account at
``r``, and rebalances only ``n_rebalance`` times. The terminal P&L is no
longer zero, but its standard deviation shrinks like
:math:`1/\\sqrt{n}` (Boyle and Emanuel 1980), and its mean is zero under
the risk-neutral drift whatever ``n`` is, because the discounted hedge
portfolio is a martingale. Spot paths come from
:func:`financialkit.integrators.gbm_paths`.

References: F. Black and M. Scholes, *J. Political Economy* 81 (1973)
637-654; P. P. Boyle and D. Emanuel, *J. Financial Economics* 8 (1980)
259-282; Hull, *Options, Futures, and Other Derivatives*, 10th ed.,
Section 19.4.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.integrators import RNGLike, gbm_paths
from financialkit.options.core.base import HedgeResult
from financialkit.options.systems.closed_form import black_scholes_price
from financialkit.options.systems.greeks import black_scholes_greeks
from financialkit.options.utils.common import payoff

__all__ = ["delta_hedge"]


def delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_rebalance: int,
    n_paths: int,
    mu: Optional[float] = None,
    q: float = 0.0,
    option_type: str = "call",
    hedge_sigma: Optional[float] = None,
    rng: RNGLike = None,
) -> HedgeResult:
    """Simulate a short option hedged with the Black-Scholes delta.

    Parameters
    ----------
    S0, K, T, r : float
        Initial spot, strike, maturity (years) and risk-free rate.
    sigma : float
        Realized volatility of the simulated paths.
    n_rebalance : int
        Number of equally spaced hedge rebalances (the hedge is set at
        time 0 and adjusted at each later date before ``T``).
    n_paths : int
        Number of simulated paths.
    mu : float, optional
        Real-world drift of the stock; defaults to the risk-neutral
        ``r - q``.
    q : float
        Continuous dividend yield, paid to the stock holder.
    option_type : {"call", "put"}
    hedge_sigma : float, optional
        Volatility used to price and delta-hedge the option; defaults to
        ``sigma``. A mismatch shows the cost of hedging at the wrong vol.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    HedgeResult

    Examples
    --------
    >>> res = delta_hedge(100.0, 100.0, 1.0, 0.05, 0.2, n_rebalance=52, n_paths=2000, rng=0)
    >>> bool(abs(res.pnl.mean()) < 4 * res.pnl.std() / 2000**0.5)
    True
    """
    if hedge_sigma is None:
        hedge_sigma = sigma
    if mu is None:
        mu = r - q
    times, S = gbm_paths(S0, mu, sigma, T, n_rebalance, n_paths, rng=np.random.default_rng(rng))
    dt = T / n_rebalance
    premium = float(black_scholes_price(S0, K, T, r, hedge_sigma, q, option_type))
    delta = np.asarray(black_scholes_greeks(S[:, 0], K, T, r, hedge_sigma, q, option_type).delta)
    cash = premium - delta * S[:, 0]
    for k in range(1, n_rebalance + 1):
        # interest on the account, plus the dividend on the shares held over the step
        cash = cash * np.exp(r * dt) + delta * S[:, k] * np.expm1(q * dt)
        if k < n_rebalance:
            new_delta = np.asarray(black_scholes_greeks(S[:, k], K, T - times[k], r, hedge_sigma, q, option_type).delta)
            cash -= (new_delta - delta) * S[:, k]
            delta = new_delta
    pnl = cash + delta * S[:, -1] - np.asarray(payoff(S[:, -1], K, option_type))
    return HedgeResult(times=times, spot=S, pnl=pnl, option_price=premium)

"""Drawdowns and Young's Calmar ratio.

A drawdown is the fall of wealth from its running peak. The maximum
drawdown is the worst such fall an investor would have lived through,
which variance-based measures miss because they ignore the order of
returns. Terry Young's Calmar ratio (1991) divides the annualised
return by the maximum drawdown, the trade-off commodity-trading advisors
are judged on.

References: T. W. Young, "Calmar Ratio: A Smoother Tool," *Futures* 20
(1991) 40; M. Magdon-Ismail, A. F. Atiya, A. Pratap and Y. S.
Abu-Mostafa, "On the Maximum Drawdown of a Brownian Motion," *J. Applied
Probability* 41 (2004) 147-161.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.constants import TRADING_DAYS_PER_YEAR
from financialkit.risk.core.base import DrawdownResult

__all__ = ["drawdowns", "calmar_ratio"]


def drawdowns(returns: ArrayLike, log: bool = False) -> DrawdownResult:
    """Drawdown path and maximum drawdown of a compounded return series.

    Parameters
    ----------
    returns : array_like, shape (n,)
        Periodic returns in time order.
    log : bool
        Treat the returns as log returns rather than simple returns.

    Returns
    -------
    DrawdownResult

    Examples
    --------
    Up 50%, down 50%, up 20%: the peak of 1.5 falls to 0.75, a 50% drawdown.

    >>> res = drawdowns([0.5, -0.5, 0.2])
    >>> res.max_drawdown, res.peak_index, res.trough_index, res.recovery_index
    (0.5, 1, 2, -1)
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    growth = np.exp(r) if log else 1.0 + r
    wealth = np.concatenate(([1.0], np.cumprod(growth)))
    peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / peak
    trough = int(np.argmax(dd))
    peak_idx = int(np.argmax(wealth[: trough + 1]))
    after = np.nonzero(wealth[trough:] >= peak[trough])[0]
    recovery = int(trough + after[0]) if dd[trough] > 0 and after.size else -1
    return DrawdownResult(
        wealth=wealth,
        peak=peak,
        drawdown=dd,
        max_drawdown=float(dd[trough]),
        peak_index=peak_idx,
        trough_index=trough,
        recovery_index=recovery,
    )


def calmar_ratio(returns: ArrayLike, periods_per_year: float = TRADING_DAYS_PER_YEAR, log: bool = False) -> float:
    """Young's Calmar ratio: compound annual growth rate over maximum drawdown.

    Young used the trailing 36 months; pass that window's returns to
    reproduce his convention.

    Parameters
    ----------
    returns : array_like, shape (n,)
        Periodic returns.
    periods_per_year : float
        12 for monthly, 252 for daily returns.
    log : bool
        Treat the returns as log returns.

    Returns
    -------
    float
        ``inf`` if there was no drawdown.

    Examples
    --------
    >>> round(calmar_ratio([0.10, -0.05, 0.08, 0.02], periods_per_year=4), 4)
    3.0234
    """
    res = drawdowns(returns, log=log)
    years = (res.wealth.size - 1) / periods_per_year
    cagr = res.wealth[-1] ** (1.0 / years) - 1.0
    return float(cagr / res.max_drawdown) if res.max_drawdown > 0 else float("inf")

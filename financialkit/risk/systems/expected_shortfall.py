"""Expected shortfall: the Acerbi-Tasche sample estimator and the normal
and Student-t closed forms.

Expected shortfall is the average loss in the worst :math:`1 - \\alpha`
of outcomes. For a distribution with atoms, "the average of the losses
beyond VaR" is not coherent; Acerbi and Tasche (2002) showed that the
right definition charges only the part of the atom at the quantile
needed to fill a tail of probability exactly :math:`1 - \\alpha`, which
for a sample of size :math:`n` and :math:`p = 1 - \\alpha` reads

.. math::

   \\mathrm{ES} = -\\frac{1}{np}\\Big(\\sum_{i=1}^{\\lfloor np \\rfloor} X_{(i)}
   + (np - \\lfloor np \\rfloor) X_{(\\lfloor np \\rfloor + 1)}\\Big).

References: C. Acerbi and D. Tasche, "On the Coherence of Expected
Shortfall," *J. Banking Finance* 26 (2002) 1487-1503; A. J. McNeil, R.
Frey and P. Embrechts, *Quantitative Risk Management*, revised ed.
(2015), Sec. 2.3.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.risk.utils.tail import weighted_tail

__all__ = ["expected_shortfall", "parametric_es"]


def expected_shortfall(pnl: ArrayLike, alpha: float = 0.99) -> float:
    """Acerbi-Tasche expected shortfall of equally likely scenarios.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Scenario profits and losses.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        A positive loss, never below the VaR at the same level.

    Examples
    --------
    With 100 outcomes and a 2.5% tail the estimator takes the two worst
    losses in full and half of the third.

    >>> import numpy as np
    >>> expected_shortfall(-np.arange(1.0, 101.0), 0.975)
    99.2
    """
    return weighted_tail(pnl, alpha)[1]


def parametric_es(
    mu: ArrayLike, sigma: ArrayLike, alpha: ArrayLike = 0.99, horizon: ArrayLike = 1.0, distribution: str = "normal", df: float | None = None
) -> np.ndarray | float:
    """Expected shortfall of a normal or (unit-variance-scaled) Student-t profit and loss.

    Normal: :math:`-\\mu h + \\sigma\\sqrt{h}\\, \\varphi(z_\\alpha) / (1 - \\alpha)`.
    Student-t with :math:`\\nu` degrees of freedom and :math:`t_\\alpha = t_\\nu^{-1}(\\alpha)`:
    :math:`-\\mu h + \\sigma\\sqrt{h}\\sqrt{(\\nu - 2)/\\nu}\\; \\frac{g_\\nu(t_\\alpha)}{1 - \\alpha} \\frac{\\nu + t_\\alpha^2}{\\nu - 1}`
    (McNeil, Frey and Embrechts, Example 2.15).

    Parameters
    ----------
    mu, sigma, alpha, horizon, distribution, df
        As in :func:`~financialkit.risk.systems.var.parametric_var`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    The Basel 97.5% ES of a standard normal is close to its 99% VaR.

    >>> round(parametric_es(0.0, 1.0, 0.975), 4)
    2.3378
    """
    mu, sigma, alpha, h = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, alpha, horizon))
    if np.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("alpha must lie in (0, 1)")
    if distribution == "normal":
        k = stats.norm.pdf(stats.norm.ppf(alpha)) / (1.0 - alpha)
    elif distribution == "t":
        if df is None or df <= 2:
            raise ValueError("distribution='t' needs df > 2 so the variance is finite")
        t_a = stats.t.ppf(alpha, df)
        k = np.sqrt((df - 2.0) / df) * stats.t.pdf(t_a, df) / (1.0 - alpha) * (df + t_a**2) / (df - 1.0)
    else:
        raise ValueError(f"distribution must be 'normal' or 't', got {distribution!r}")
    out = -mu * h + sigma * np.sqrt(h) * k
    return float(out) if out.ndim == 0 else out

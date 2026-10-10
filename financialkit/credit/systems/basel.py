"""The Basel II internal-ratings-based (IRB) capital formula.

Gordy (2003) showed that capital is portfolio-invariant (each loan's
charge depends only on its own parameters) exactly in Vasicek's
asymptotic single-risk-factor model. The Basel Committee (2004) built
the IRB formula on it: capital per unit exposure is the 99.9% Vasicek
quantile of the loss rate minus the expected loss, times a maturity
adjustment,

.. math::

   K = \\left[LGD \\cdot N\\left(\\frac{N^{-1}(PD) + \\sqrt{R}\\, N^{-1}(0.999)}{\\sqrt{1 - R}}\\right)
       - PD \\cdot LGD\\right] \\frac{1 + (M - 2.5)\\, b}{1 - 1.5\\, b},

with :math:`b = (0.11852 - 0.05478 \\ln PD)^2`, a supervisory asset
correlation :math:`R` that falls from 24% to 12% as PD rises, and risk
weighted assets :math:`12.5 \\cdot K \\cdot EAD`.

References: Basel Committee on Banking Supervision, *International
Convergence of Capital Measurement and Capital Standards: A Revised
Framework* (2004, comprehensive version 2006), paragraphs 272-330;
BCBS, *An Explanatory Note on the Basel II IRB Risk Weight Functions*
(2005); M. B. Gordy, "A Risk-Factor Model Foundation for Ratings-Based
Bank Capital Rules," *J. Financial Intermediation* 12 (2003) 199-232.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.credit.systems.vasicek import vasicek_quantile
from financialkit.options.utils.common import as_output

__all__ = ["basel_correlation", "basel_maturity_adjustment", "basel_irb_capital", "basel_risk_weight"]

_ASSET_CLASSES = ("corporate", "mortgage", "revolving", "retail")


def _weight(pd: np.ndarray, k: float) -> np.ndarray:
    return -np.expm1(-k * pd) / -np.expm1(-k)


def basel_correlation(pd: ArrayLike, asset_class: str = "corporate", sales: ArrayLike | None = None) -> np.ndarray | float:
    """Supervisory asset correlation :math:`R` for an asset class.

    - ``"corporate"`` (also sovereign and bank): 12% to 24%, interpolated
      with weight :math:`(1 - e^{-50 PD})/(1 - e^{-50})`, less the SME
      adjustment :math:`0.04 (1 - (S - 5)/45)` when annual ``sales``
      :math:`S` (EUR millions, clipped to 5-50) are given;
    - ``"mortgage"``: 15%; ``"revolving"``: 4%;
    - ``"retail"`` (other retail): 3% to 16% with weight factor 35.

    Parameters
    ----------
    pd : array_like
    asset_class : {"corporate", "mortgage", "revolving", "retail"}
    sales : array_like, optional

    Returns
    -------
    float or ndarray
    """
    pd = np.asarray(pd, dtype=np.float64)
    if asset_class == "corporate":
        w = _weight(pd, 50.0)
        R = 0.12 * w + 0.24 * (1.0 - w)
        if sales is not None:
            S = np.clip(np.asarray(sales, dtype=np.float64), 5.0, 50.0)
            R = R - 0.04 * (1.0 - (S - 5.0) / 45.0)
    elif asset_class == "mortgage":
        R = np.full(pd.shape, 0.15)
    elif asset_class == "revolving":
        R = np.full(pd.shape, 0.04)
    elif asset_class == "retail":
        w = _weight(pd, 35.0)
        R = 0.03 * w + 0.16 * (1.0 - w)
    else:
        raise ValueError(f"asset_class must be one of {_ASSET_CLASSES}, got {asset_class!r}")
    return as_output(R)


def basel_maturity_adjustment(pd: ArrayLike, maturity: ArrayLike = 2.5) -> np.ndarray | float:
    """Maturity adjustment :math:`(1 + (M - 2.5) b) / (1 - 1.5 b)`, with :math:`b = (0.11852 - 0.05478 \\ln PD)^2`.

    Parameters
    ----------
    pd : array_like
    maturity : array_like
        Effective maturity :math:`M` in years (1 to 5 in the framework).

    Returns
    -------
    float or ndarray
    """
    pd, M = np.asarray(pd, dtype=np.float64), np.asarray(maturity, dtype=np.float64)
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    return as_output((1.0 + (M - 2.5) * b) / (1.0 - 1.5 * b))


def basel_irb_capital(
    pd: ArrayLike,
    lgd: ArrayLike = 0.45,
    maturity: ArrayLike = 2.5,
    asset_class: str = "corporate",
    sales: ArrayLike | None = None,
    alpha: float = 0.999,
    pd_floor: float = 0.0003,
) -> np.ndarray | float:
    """IRB capital requirement :math:`K` per unit of exposure at default.

    Retail classes have no maturity adjustment. A defaulted exposure
    (``pd = 1``) gets zero unexpected-loss capital.

    Parameters
    ----------
    pd : array_like
        One-year default probability (floored at 0.03%).
    lgd : array_like
        Loss given default (45% is the foundation-IRB senior unsecured value).
    maturity : array_like
        Effective maturity in years.
    asset_class : {"corporate", "mortgage", "revolving", "retail"}
    sales : array_like, optional
        Annual sales in EUR millions for the SME correlation adjustment.
    alpha : float
        Confidence level, 99.9% in the framework.
    pd_floor : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    A corporate loan with PD 1%, LGD 45% and M = 2.5:

    >>> round(12.5 * basel_irb_capital(0.01), 4)
    0.9232
    """
    pd = np.maximum(np.asarray(pd, dtype=np.float64), pd_floor)
    lgd = np.asarray(lgd, dtype=np.float64)
    R = np.asarray(basel_correlation(np.minimum(pd, 1.0 - 1e-12), asset_class, sales))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.asarray(vasicek_quantile(alpha, np.minimum(pd, 1.0 - 1e-12), R))
        K = lgd * q - pd * lgd
        if asset_class == "corporate":
            K = K * np.asarray(basel_maturity_adjustment(np.minimum(pd, 1.0 - 1e-12), maturity))
    return as_output(np.where(pd >= 1.0, 0.0, np.maximum(K, 0.0)))


def basel_risk_weight(
    pd: ArrayLike, lgd: ArrayLike = 0.45, maturity: ArrayLike = 2.5, asset_class: str = "corporate", sales: ArrayLike | None = None, scaling: float = 1.0
) -> np.ndarray | float:
    """Risk weight :math:`12.5 \\cdot K` (times the 1.06 scaling factor if passed).

    Parameters
    ----------
    pd, lgd, maturity, asset_class, sales
        As in :func:`basel_irb_capital`.
    scaling : float
        The framework multiplies IRB risk-weighted assets by 1.06; the
        Committee's illustrative tables use 1.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Basel II framework, illustrative IRB risk weights (corporate, LGD 45%, M = 2.5):

    >>> [round(100 * basel_risk_weight(p), 2) for p in (0.0003, 0.01)]
    [14.44, 92.32]
    """
    return as_output(12.5 * scaling * np.asarray(basel_irb_capital(pd, lgd, maturity, asset_class, sales)))

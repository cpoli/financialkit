"""Covariance estimators for portfolio construction: Sharpe (1963) and Ledoit-Wolf (2004).

The sample covariance of :math:`d` assets has :math:`d(d+1)/2` free
parameters, so with a few years of data it is noisy, ill-conditioned and
singular once :math:`d` exceeds the number of observations; optimizers
then exploit its errors. Sharpe's single-index model imposes structure,
:math:`\\Sigma = \\sigma_m^2 \\beta\\beta' + \\mathrm{diag}(\\sigma_\\varepsilon^2)`,
with only :math:`3d + 1` parameters. Ledoit and Wolf instead shrink the
sample matrix towards a structured target with the intensity that
minimizes the expected Frobenius loss, estimated from the data:
towards a scaled identity (Ledoit and Wolf 2004a) or towards the
constant-correlation matrix (Ledoit and Wolf 2004b). The shrinkage
estimators are hand-written, since no allowed dependency provides them.

References: W. F. Sharpe, "A Simplified Model for Portfolio Analysis,"
*Management Sci.* 9 (1963) 277-293; O. Ledoit and M. Wolf, "A
Well-Conditioned Estimator for Large-Dimensional Covariance Matrices,"
*J. Multivariate Anal.* 88 (2004) 365-411; O. Ledoit and M. Wolf,
"Honey, I Shrunk the Sample Covariance Matrix," *J. Portfolio Mgmt.* 30
(2004) 110-119.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.portfolio.core.base import CovarianceResult

__all__ = ["sample_covariance", "single_index_covariance", "ledoit_wolf"]


def _returns(returns: ArrayLike) -> np.ndarray:
    X = np.asarray(returns, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("returns must have shape (n_periods, n_assets) with at least two periods")
    return X


def sample_covariance(returns: ArrayLike) -> CovarianceResult:
    """The unbiased sample covariance matrix.

    Parameters
    ----------
    returns : array_like, shape (n, d)

    Returns
    -------
    CovarianceResult

    Examples
    --------
    >>> sample_covariance([[0.01, 0.02], [0.03, 0.0], [-0.01, 0.01]]).cov
    array([[ 4.e-04, -1.e-04],
           [-1.e-04,  1.e-04]])
    """
    X = _returns(returns)
    d = X.shape[1]
    return CovarianceResult(cov=np.atleast_2d(np.cov(X, rowvar=False)), method="sample", n_params=d * (d + 1) // 2)


def single_index_covariance(returns: ArrayLike, market: ArrayLike) -> CovarianceResult:
    """Sharpe's single-index covariance, :math:`\\sigma_m^2\\beta\\beta' + \\mathrm{diag}(\\sigma_\\varepsilon^2)`.

    Each asset is regressed on the market by least squares,
    :math:`R_i = \\alpha_i + \\beta_i R_m + \\varepsilon_i`, and the
    residuals are taken to be uncorrelated across assets.

    Parameters
    ----------
    returns : array_like, shape (n, d)
    market : array_like, shape (n,)
        Returns of the index.

    Returns
    -------
    CovarianceResult
        With ``betas`` set and ``target`` holding the market part
        :math:`\\sigma_m^2\\beta\\beta'`.

    Examples
    --------
    Assets that are exact multiples of the market have no residual risk.

    >>> import numpy as np
    >>> m = np.array([0.01, -0.02, 0.03, 0.0])
    >>> res = single_index_covariance(np.column_stack([m, 2 * m]), m)
    >>> np.round(res.betas, 12), bool(np.allclose(res.cov, np.var(m, ddof=1) * np.outer([1, 2], [1, 2])))
    (array([1., 2.]), True)
    """
    X = _returns(returns)
    m = np.asarray(market, dtype=np.float64)
    n, d = X.shape
    if m.shape != (n,):
        raise ValueError(f"market must have shape ({n},), got {m.shape}")
    design = np.column_stack([np.ones(n), m])
    coef, *_ = np.linalg.lstsq(design, X, rcond=None)
    resid = X - design @ coef
    beta = coef[1]
    systematic = np.var(m, ddof=1) * np.outer(beta, beta)
    resid_var = (resid**2).sum(axis=0) / (n - 2)
    return CovarianceResult(cov=systematic + np.diag(resid_var), method="single_index", n_params=3 * d + 1, target=systematic, betas=beta)


def ledoit_wolf(returns: ArrayLike, target: str = "identity") -> CovarianceResult:
    """Ledoit-Wolf shrinkage of the sample covariance towards a structured target.

    The estimate is :math:`\\delta F + (1 - \\delta) S`, where :math:`S` is
    the sample covariance (divided by :math:`n`) and the intensity
    :math:`\\delta \\in [0, 1]` is the consistent estimate of the
    loss-minimizing value.

    Parameters
    ----------
    returns : array_like, shape (n, d)
    target : {"identity", "constant_correlation"}
        ``"identity"``: :math:`F = (\\mathrm{tr}\\,S / d) I` (2004a);
        ``"constant_correlation"``: sample variances with every
        correlation set to the average one (2004b).

    Returns
    -------
    CovarianceResult

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((30, 20))
    >>> res = ledoit_wolf(X)
    >>> bool(0 < res.shrinkage < 1), bool(res.condition_number < sample_covariance(X).condition_number)
    (True, True)
    """
    X = _returns(returns)
    n, d = X.shape
    Y = X - X.mean(axis=0)
    S = Y.T @ Y / n
    if target == "identity":
        mu = np.trace(S) / d
        F = mu * np.eye(d)
        # Norms are tr(A A') / d, as in Ledoit-Wolf (2004a), Sec. 3.
        d2 = np.sum((S - F) ** 2) / d
        # sum_k ||y_k y_k' - S||^2 = sum_k ||y_k||^4 - n ||S||^2, since sum_k y_k y_k' = n S.
        b_bar2 = (np.sum(np.sum(Y**2, axis=1) ** 2) - n * np.sum(S**2)) / d / n**2
        delta = min(b_bar2, d2) / d2 if d2 > 0 else 1.0
    elif target == "constant_correlation":
        s = np.sqrt(np.diag(S))
        R = S / np.outer(s, s)
        r_bar = (R.sum() - d) / (d * (d - 1))
        F = r_bar * np.outer(s, s)
        np.fill_diagonal(F, np.diag(S))
        Y2 = Y**2
        pi_mat = Y2.T @ Y2 / n - S**2
        pi_hat = pi_mat.sum()
        # theta_{ii,ij} = E[(y_i^2 - s_ii)(y_i y_j - s_ij)], Ledoit-Wolf (2004b), Appendix B.
        theta = (Y**3).T @ Y / n - np.diag(S)[:, None] * S
        off = np.ones((d, d)) - np.eye(d)
        rho_hat = np.trace(pi_mat) + r_bar * np.sum(off * (np.outer(1 / s, s) * theta + np.outer(s, 1 / s) * theta.T)) / 2.0
        gamma_hat = np.sum((F - S) ** 2)
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa / n, 0.0, 1.0))
    else:
        raise ValueError(f"target must be 'identity' or 'constant_correlation', got {target!r}")
    return CovarianceResult(cov=delta * F + (1.0 - delta) * S, method="ledoit_wolf", n_params=d * (d + 1) // 2, shrinkage=float(delta), target=F)

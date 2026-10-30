"""Unit roots, stationarity and cointegration: the ADF, KPSS, Engle-Granger and Johansen tests.

Dickey and Fuller (1979) showed that under a unit root the t-ratio of
:math:`\\gamma` in :math:`\\Delta y_t = \\gamma y_{t-1} + e_t` does not
have a t distribution, and tabulated its distribution by simulation; Said
and Dickey (1984) added lagged differences to absorb serial correlation.
Engle and Granger (1987) tested for cointegration by running the same test
on the residuals of a levels regression, with critical values that depend
on the number of variables. Critical values come from MacKinnon's (2010)
response surfaces :math:`c(T) = \\beta_\\infty + \\beta_1/T + \\beta_2/T^2 + \\beta_3/T^3`.
Johansen (1988) estimated the whole cointegrating space by maximum
likelihood, as the reduced-rank regression of :math:`\\Delta y_t` on
:math:`y_{t-1}`, and tested its rank with likelihood-ratio statistics
(:func:`scipy.linalg.eigh` solves the generalized eigenproblem).
Kwiatkowski, Phillips, Schmidt and Shin (1992) reversed the null:
stationarity, rejected by a large partial-sum statistic.
References: D. A. Dickey and W. A. Fuller, Journal of the American
Statistical Association 74(366) (1979), 427-431; S. E. Said and D. A.
Dickey, Biometrika 71(3) (1984), 599-607; R. F. Engle and C. W. J.
Granger, "Co-Integration and Error Correction," Econometrica 55(2) (1987),
251-276; J. G. MacKinnon, "Critical Values for Cointegration Tests,"
Queen's Economics Department Working Paper 1227 (2010), Table 2; S.
Johansen, "Statistical Analysis of Cointegration Vectors," Journal of
Economic Dynamics and Control 12(2-3) (1988), 231-254; J. G. MacKinnon, A.
A. Haug and L. Michelis, "Numerical Distribution Functions of Likelihood
Ratio Tests for Cointegration," Journal of Applied Econometrics 14(5)
(1999), 563-577; D. Kwiatkowski, P. C. B. Phillips, P. Schmidt and Y. Shin,
Journal of Econometrics 54(1-3) (1992), 159-178.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import eigh

from financialkit._deprecation import renamed_parameter
from financialkit.time_series.core.base import CointegrationResult, JohansenResult, StationarityResult, UnitRootResult
from financialkit.time_series.systems.efficiency import autocovariance
from financialkit.time_series.utils.numerics import lag_matrix, ols

__all__ = ["mackinnon_critical_values", "adf_test", "engle_granger", "kpss_test", "johansen_test"]

# MacKinnon (2010), Table 2: (beta_inf, beta_1, beta_2, beta_3) for the 1%, 5%
# and 10% levels, by deterministic terms and number of variables N.
_MACKINNON_2010: dict[str, dict[int, tuple[tuple[float, float, float, float], ...]]] = {
    "n": {
        1: ((-2.56574, -2.2358, -3.627, 0.0), (-1.94100, -0.2686, -3.365, 31.223), (-1.61682, 0.2656, -2.714, 25.364)),
    },
    "c": {
        1: ((-3.43035, -6.5393, -16.786, -79.433), (-2.86154, -2.8903, -4.234, -40.040), (-2.56677, -1.5384, -2.809, 0.0)),
        2: ((-3.89644, -10.9519, -33.527, 0.0), (-3.33613, -6.1101, -6.823, 0.0), (-3.04445, -4.2412, -2.720, 0.0)),
        3: ((-4.29374, -14.4354, -33.195, 47.433), (-3.74066, -8.5632, -10.852, 27.982), (-3.45218, -6.2143, -3.718, 0.0)),
        4: ((-4.64332, -18.1031, -37.972, 0.0), (-4.09600, -11.2349, -11.175, 0.0), (-3.81020, -8.3931, -4.137, 0.0)),
    },
    "ct": {
        1: ((-3.95877, -9.0531, -28.428, -134.155), (-3.41049, -4.3904, -9.036, -45.374), (-3.12705, -2.5856, -3.925, -22.380)),
        2: ((-4.32762, -15.4387, -35.679, 0.0), (-3.78057, -9.5106, -12.074, 0.0), (-3.49631, -7.0815, -7.538, 21.892)),
        3: ((-4.66305, -18.7688, -49.793, 104.244), (-4.11890, -11.8922, -19.031, 77.332), (-3.83511, -9.0723, -8.504, 35.403)),
        4: ((-4.96940, -22.4694, -52.599, 51.314), (-4.42871, -14.5876, -18.228, 39.647), (-4.14633, -11.2500, -9.873, 54.109)),
    },
}


@renamed_parameter("nobs", "n_obs", "financialkit.time_series.mackinnon_critical_values")
def mackinnon_critical_values(n_obs: int, regression: str = "c", n_vars: int = 1) -> dict:
    """MacKinnon (2010) finite-sample critical values of the Dickey-Fuller and Engle-Granger tests.

    Parameters
    ----------
    n_obs : int
        Observations in the test regression.
    regression : {"n", "c", "ct"}
        No deterministic terms, a constant, or a constant and a linear trend.
    n_vars : int
        1 for a unit-root test; the number of variables in the
        cointegrating regression (2 to 4) for Engle-Granger.

    Returns
    -------
    dict
        ``{"1%": ..., "5%": ..., "10%": ...}``.

    Examples
    --------
    >>> {k: round(v, 3) for k, v in mackinnon_critical_values(10**9).items()}
    {'1%': -3.43, '5%': -2.862, '10%': -2.567}
    """
    try:
        rows = _MACKINNON_2010[regression][n_vars]
    except KeyError:
        raise ValueError("supported: regression 'n' with n_vars=1, or 'c'/'ct' with n_vars in 1..4") from None
    T = float(n_obs)
    return {lvl: b0 + b1 / T + b2 / T**2 + b3 / T**3 for lvl, (b0, b1, b2, b3) in zip(("1%", "5%", "10%"), rows, strict=True)}


def _deterministic(n: int, regression: str) -> np.ndarray:
    if regression == "n":
        return np.empty((n, 0))
    if regression == "c":
        return np.ones((n, 1))
    if regression == "ct":
        return np.column_stack([np.ones(n), np.arange(1.0, n + 1)])
    raise ValueError("regression must be 'n', 'c' or 'ct'")


def _adf_regression(y: np.ndarray, lags: int, regression: str, drop: int):
    """ADF regression on ``t >= drop + 1`` with `lags` lagged differences."""
    dy = np.diff(y)
    X_lags = lag_matrix(dy, drop)[:, :lags]
    target = dy[drop:]
    ylag = y[drop:-1]
    X = np.column_stack([ylag, X_lags, _deterministic(target.size, regression)])
    return ols(target, X)


def adf_test(x, lags: Optional[int] = None, regression: str = "c", max_lags: Optional[int] = None) -> UnitRootResult:
    """Augmented Dickey-Fuller test of a unit root.

    Regresses

    .. math:: \\Delta y_t = \\text{det}_t + \\gamma y_{t-1} + \\sum_{i=1}^{k} \\delta_i \\Delta y_{t-i} + e_t

    and compares the t-ratio of :math:`\\gamma` with MacKinnon's critical
    values; the null :math:`\\gamma = 0` (a unit root) is rejected for
    large negative values.

    Parameters
    ----------
    x : array_like
    lags : int, optional
        Lagged differences :math:`k`. If omitted, chosen by AIC from
        ``0..max_lags`` on a common sample, then the test is rerun on all
        available data.
    regression : {"n", "c", "ct"}
    max_lags : int, optional
        Defaults to Schwert's rule :math:`\\lfloor 12 (n/100)^{1/4} \\rfloor`.

    Returns
    -------
    UnitRootResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> adf_test(np.cumsum(rng.standard_normal(500)), lags=0).rejects()
    False
    >>> adf_test(rng.standard_normal(500), lags=0).rejects()
    True
    """
    y = np.asarray(x, dtype=np.float64)
    n = y.size
    if lags is None:
        max_lags = int(12 * (n / 100) ** 0.25) if max_lags is None else max_lags
        max_lags = min(max_lags, n // 2 - 3)
        best_aic, lags = np.inf, 0
        for k in range(max_lags + 1):
            res = _adf_regression(y, k, regression, max_lags)
            m = res.residuals.size
            aic = m * np.log(res.residuals @ res.residuals / m) + 2 * res.coef.size
            if aic < best_aic:
                best_aic, lags = aic, k
    res = _adf_regression(y, lags, regression, lags)
    n_obs = res.residuals.size
    return UnitRootResult(
        statistic=float(res.t_values[0]),
        critical_values=mackinnon_critical_values(n_obs, regression),
        lags=int(lags),
        n_obs=n_obs,
        regression=regression,
        gamma=float(res.coef[0]),
    )


def engle_granger(y, x, lags: Optional[int] = 0, trend: str = "c") -> CointegrationResult:
    """Engle-Granger two-step test of cointegration between `y` and the columns of `x`.

    Step 1 regresses :math:`y_t = a + b' x_t + u_t` by least squares, which
    is super-consistent when the series are cointegrated. Step 2 runs a
    Dickey-Fuller test without deterministic terms on :math:`\\hat u_t`.
    Because :math:`b` was estimated to make :math:`\\hat u_t` look
    stationary, the critical values are more negative than Dickey and
    Fuller's and depend on the number of variables (MacKinnon 2010).

    Parameters
    ----------
    y : array_like, shape (n,)
    x : array_like, shape (n,) or (n, k)
        One to three regressors.
    lags : int or None
        Lagged differences in step 2 (``None`` chooses by AIC).
    trend : {"c", "ct"}
        Deterministic terms of the step-1 regression.

    Returns
    -------
    CointegrationResult

    Examples
    --------
    >>> from financialkit.time_series.utils.synthetic import simulate_cointegrated_pair
    >>> y, x = simulate_cointegrated_pair(1000, beta=2.0, phi=0.8, rng=0)
    >>> res = engle_granger(y, x)
    >>> round(float(res.beta[0]), 1), res.rejects()
    (2.0, True)
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(x, dtype=np.float64)
    X = X[:, None] if X.ndim == 1 else X
    det = _deterministic(y.size, trend)
    first = ols(y, np.column_stack([det, X]))
    u = first.residuals
    adf = adf_test(u, lags=lags, regression="n")
    cv = mackinnon_critical_values(adf.n_obs, trend, n_vars=1 + X.shape[1])
    return CointegrationResult(
        statistic=adf.statistic,
        critical_values=cv,
        intercept=float(first.coef[0]),
        beta=first.coef[det.shape[1] :],
        residuals=u,
        adf=adf,
        r_squared=first.r_squared,
    )


# Kwiatkowski, Phillips, Schmidt and Shin (1992), Table 1: upper-tail
# asymptotic critical values at 10%, 5%, 2.5% and 1%.
_KPSS_1992 = {
    "c": {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739},
    "ct": {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216},
}


def kpss_test(x, regression: str = "c", lags: Optional[int] = None) -> StationarityResult:
    """Kwiatkowski-Phillips-Schmidt-Shin test of stationarity against a unit root.

    The residuals :math:`e_t` of a regression of :math:`y_t` on a constant
    (``"c"``) or a constant and trend (``"ct"``) are cumulated into
    :math:`S_t = \\sum_{s \\le t} e_s`, and

    .. math:: \\eta = \\frac{T^{-2}\\sum_{t=1}^{T} S_t^2}{\\hat\\sigma^2_\\infty},
       \\qquad \\hat\\sigma^2_\\infty = \\hat\\gamma_0 + 2\\sum_{j=1}^{l}\\Bigl(1 - \\frac{j}{l+1}\\Bigr)\\hat\\gamma_j

    is the LM statistic of a zero-variance random-walk component, with
    Newey and West's Bartlett-window long-run variance. Stationarity is the
    *null*, rejected for large :math:`\\eta`, so KPSS and Dickey-Fuller
    answer the same question from opposite sides.

    Parameters
    ----------
    x : array_like
    regression : {"c", "ct"}
    lags : int, optional
        Bartlett-window lags :math:`l`; defaults to KPSS's
        :math:`l_{12} = \\lfloor 12 (T/100)^{1/4} \\rfloor`.

    Returns
    -------
    StationarityResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> kpss_test(rng.standard_normal(500)).rejects()
    False
    >>> kpss_test(np.cumsum(rng.standard_normal(500))).rejects()
    True
    """
    y = np.asarray(x, dtype=np.float64)
    T = y.size
    if regression not in _KPSS_1992:
        raise ValueError("regression must be 'c' or 'ct'")
    e = ols(y, _deterministic(T, regression)).residuals
    l = int(12 * (T / 100) ** 0.25) if lags is None else int(lags)
    l = min(l, T - 1)
    g = autocovariance(e, l)
    w = 1.0 - np.arange(1, l + 1) / (l + 1.0)
    lrv = float(g[0] + 2.0 * w @ g[1:])
    S = np.cumsum(e)
    eta = float(S @ S) / (T * T * lrv)
    return StationarityResult(statistic=eta, critical_values=dict(_KPSS_1992[regression]), lags=l, n_obs=T, regression=regression)


# MacKinnon, Haug and Michelis (1999): asymptotic critical values of
# Johansen's trace and maximum-eigenvalue statistics at 10%, 5% and 1%, for
# k - r = 1..5 nonstationary directions; "n" has no deterministic terms and
# "c" an unrestricted constant (a linear trend in the levels).
_JOHANSEN_CV: dict[str, dict[str, tuple[tuple[float, float, float], ...]]] = {
    "n": {
        "trace": ((2.9762, 4.1296, 6.9406), (10.4741, 12.3212, 16.3640), (21.7781, 24.2761, 29.5147), (37.0339, 40.1749, 46.5716), (56.2839, 60.0627, 67.6367)),
        "max_eigen": (
            (2.9762, 4.1296, 6.9406),
            (9.4748, 11.2246, 15.0923),
            (15.7175, 17.7961, 22.2519),
            (21.8370, 24.1592, 29.0609),
            (27.9160, 30.4428, 35.7359),
        ),
    },
    "c": {
        "trace": ((2.7055, 3.8415, 6.6349), (13.4294, 15.4943, 19.9349), (27.0669, 29.7961, 35.4628), (44.4929, 47.8545, 54.6815), (65.8202, 69.8189, 77.8202)),
        "max_eigen": (
            (2.7055, 3.8415, 6.6349),
            (12.2971, 14.2639, 18.5200),
            (18.8928, 21.1314, 25.8650),
            (25.1236, 27.5858, 32.7172),
            (31.2379, 33.8777, 39.3693),
        ),
    },
}


def _johansen_cv(k: int, regression: str, test: str) -> dict:
    rows = _JOHANSEN_CV[regression][test]
    # Entry r tests k - r nonstationary directions.
    table = np.array([rows[k - r - 1] for r in range(k)])
    return {lvl: table[:, j] for j, lvl in enumerate(("10%", "5%", "1%"))}


def johansen_test(y, lags: int = 1, regression: str = "c") -> JohansenResult:
    """Johansen's maximum-likelihood test of the cointegration rank of a VAR.

    The VAR is written in error-correction form

    .. math:: \\Delta y_t = \\mu + \\Pi y_{t-1} + \\sum_{i=1}^{l}\\Gamma_i\\Delta y_{t-i} + u_t,
       \\qquad \\Pi = \\alpha\\beta',

    and the rank of :math:`\\Pi` is the number of cointegrating relations.
    After partialling the constant and lagged differences out of
    :math:`\\Delta y_t` and :math:`y_{t-1}` (residuals :math:`R_0, R_1`,
    moment matrices :math:`S_{ij} = R_i'R_j/T`), the likelihood is
    maximized by the eigenvectors of
    :math:`S_{10}S_{00}^{-1}S_{01}\\,v = \\lambda S_{11} v`, and the
    likelihood-ratio statistics are functions of the eigenvalues. Their
    distributions are Dickey-Fuller-like functionals of Brownian motion and
    depend only on :math:`k - r` and the deterministic terms.

    Parameters
    ----------
    y : array_like, shape (n, k)
        Levels of two to five series.
    lags : int
        Lagged differences :math:`l` (a VAR of order ``lags + 1`` in levels).
    regression : {"n", "c"}
        No deterministic terms, or an unrestricted constant :math:`\\mu`,
        for series with drift (the critical values assume
        :math:`\\mu` generates a linear trend in the levels).

    Returns
    -------
    JohansenResult

    Examples
    --------
    Two random walks with drift and a third series cointegrated with the first:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> walks = np.cumsum(0.1 + rng.standard_normal((1000, 2)), axis=0)
    >>> y = np.column_stack([walks, 2.0 * walks[:, 0] + rng.standard_normal(1000)])
    >>> res = johansen_test(y)
    >>> res.rank(), round(float(res.cointegrating_vectors(1)[2, 0]), 2)
    (1, -0.5)
    """
    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim != 2 or not 2 <= Y.shape[1] <= 5:
        raise ValueError("y must have shape (n, k) with 2 <= k <= 5")
    if regression not in _JOHANSEN_CV:
        raise ValueError("regression must be 'n' or 'c'")
    n, k = Y.shape
    dY = np.diff(Y, axis=0)
    T = dY.shape[0] - lags
    if T <= k * (lags + 1) + 1:
        raise ValueError("series too short for this number of lags")
    Z = [dY[lags - i : dY.shape[0] - i] for i in range(1, lags + 1)]
    if regression == "c":
        Z.append(np.ones((T, 1)))
    R0, R1 = dY[lags:], Y[lags:-1]
    if Z:
        Zm = np.hstack(Z)
        R0 = R0 - Zm @ np.linalg.lstsq(Zm, R0, rcond=None)[0]
        R1 = R1 - Zm @ np.linalg.lstsq(Zm, R1, rcond=None)[0]
    S00, S01, S11 = R0.T @ R0 / T, R0.T @ R1 / T, R1.T @ R1 / T
    lam, V = eigh(S01.T @ np.linalg.solve(S00, S01), S11)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0.0, 1.0 - 1e-15), V[:, order]
    log1m = np.log1p(-lam)
    trace = -T * np.cumsum(log1m[::-1])[::-1]
    max_eigen = -T * log1m
    return JohansenResult(
        eigenvalues=lam,
        trace=trace,
        max_eigen=max_eigen,
        trace_critical_values=_johansen_cv(k, regression, "trace"),
        max_eigen_critical_values=_johansen_cv(k, regression, "max_eigen"),
        beta=V,
        alpha=S01 @ V,
        lags=int(lags),
        n_obs=T,
        regression=regression,
    )

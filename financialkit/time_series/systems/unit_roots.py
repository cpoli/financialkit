"""Unit roots and cointegration: the augmented Dickey-Fuller and Engle-Granger tests.

Dickey and Fuller (1979) showed that under a unit root the t-ratio of
:math:`\\gamma` in :math:`\\Delta y_t = \\gamma y_{t-1} + e_t` does not
have a t distribution, and tabulated its distribution by simulation; Said
and Dickey (1984) added lagged differences to absorb serial correlation.
Engle and Granger (1987) tested for cointegration by running the same test
on the residuals of a levels regression, with critical values that depend
on the number of variables. Critical values come from MacKinnon's (2010)
response surfaces :math:`c(T) = \\beta_\\infty + \\beta_1/T + \\beta_2/T^2 + \\beta_3/T^3`.
References: D. A. Dickey and W. A. Fuller, Journal of the American
Statistical Association 74(366) (1979), 427-431; S. E. Said and D. A.
Dickey, Biometrika 71(3) (1984), 599-607; R. F. Engle and C. W. J.
Granger, "Co-Integration and Error Correction," Econometrica 55(2) (1987),
251-276; J. G. MacKinnon, "Critical Values for Cointegration Tests,"
Queen's Economics Department Working Paper 1227 (2010), Table 2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.time_series.core.base import CointegrationResult, UnitRootResult
from financialkit.time_series.utils.numerics import lag_matrix, ols

__all__ = ["mackinnon_critical_values", "adf_test", "engle_granger"]

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


def mackinnon_critical_values(nobs: int, regression: str = "c", n_vars: int = 1) -> dict:
    """MacKinnon (2010) finite-sample critical values of the Dickey-Fuller and Engle-Granger tests.

    Parameters
    ----------
    nobs : int
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
    T = float(nobs)
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
    nobs = res.residuals.size
    return UnitRootResult(
        statistic=float(res.t_values[0]),
        critical_values=mackinnon_critical_values(nobs, regression),
        lags=int(lags),
        nobs=nobs,
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
    cv = mackinnon_critical_values(adf.nobs, trend, n_vars=1 + X.shape[1])
    return CointegrationResult(
        statistic=adf.statistic,
        critical_values=cv,
        intercept=float(first.coef[0]),
        beta=first.coef[det.shape[1] :],
        residuals=u,
        adf=adf,
        r_squared=first.r_squared,
    )

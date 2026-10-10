"""Building yield curves: forward rates, bootstrapping, splines, parametric fits and PCA.

* :func:`forward_rates`: Lutz's (1940) forward rates implied by a zero curve.
* :func:`bootstrap_zero_curve` / :func:`bootstrap_par_curve`: zero rates
  stripped one bond at a time (hand-written; no library equivalent), with
  each new node found by :func:`scipy.optimize.brentq` so that coupons
  falling between nodes are discounted on the curve being built.
* :func:`carleton_cooper_discount_factors`: Carleton and Cooper's (1976)
  regression of bond prices on their cash flows for the discount factors
  (:func:`numpy.linalg.lstsq`).
* :func:`fit_mcculloch_spline`: McCulloch's (1971, 1975) spline discount
  function, linear in its coefficients and so fitted by least squares.
* :class:`NelsonSiegelCurve`, :class:`SvenssonCurve` and their fits
  (:func:`scipy.optimize.least_squares`, started from a grid over the decay
  parameters on which the betas are linear).
* :func:`yield_curve_pca`: Litterman and Scheinkman's (1991) level, slope
  and curvature as principal components (:func:`numpy.linalg.eigh`).

References: F. A. Lutz, Quarterly Journal of Economics 55(1) (1940), 36-63;
J. H. McCulloch, Journal of Business 44(1) (1971), 19-31, and Journal of
Finance 30(3) (1975), 811-830; W. T. Carleton and I. A. Cooper, Journal of
Finance 31(4) (1976), 1067-1083; C. R. Nelson and A. F. Siegel, Journal of
Business 60(4) (1987), 473-489; R. Litterman and J. Scheinkman, Journal of
Fixed Income 1(1) (1991), 54-61; L. E. O. Svensson, NBER Working Paper
4871 (1994); J. C. Hull, *Options, Futures, and Other Derivatives*, 9th ed.
(Pearson, 2015), Sec. 4.5-4.7.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq, least_squares

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CurveFitResult, CurvePCAResult, YieldCurve, ZeroCurve
from financialkit.fixed_income.systems.bonds import bond_cash_flows

__all__ = [
    "forward_rates",
    "bootstrap_zero_curve",
    "bootstrap_par_curve",
    "carleton_cooper_discount_factors",
    "SplineDiscountCurve",
    "fit_mcculloch_spline",
    "NelsonSiegelCurve",
    "SvenssonCurve",
    "fit_nelson_siegel",
    "fit_svensson",
    "yield_curve_pca",
]


def forward_rates(tenors, zero_rates) -> np.ndarray:
    r"""Lutz's forward rates for each period between consecutive tenors.

    .. math:: f_{k} = \frac{r_k t_k - r_{k-1} t_{k-1}}{t_k - t_{k-1}}, \qquad t_0 = 0

    The first period's forward is the first zero rate. Under the
    expectations hypothesis these are the market's forecasts of future
    short rates, and the zero rate is their time average.

    Parameters
    ----------
    tenors : array_like, shape (n,)
        Strictly increasing positive maturities.
    zero_rates : array_like, shape (n,)
        Continuously compounded zero rates.

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    Hull (9th ed., Table 4.5):

    >>> forward_rates([1, 2, 3, 4, 5], [0.03, 0.04, 0.046, 0.05, 0.053]).round(4).tolist()
    [0.03, 0.05, 0.058, 0.062, 0.065]
    """
    t = np.asarray(tenors, dtype=np.float64)
    r = np.asarray(zero_rates, dtype=np.float64)
    if t.shape != r.shape or np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("tenors must be positive, strictly increasing and match zero_rates")
    rt = np.concatenate([[0.0], r * t])
    return np.diff(rt) / np.diff(np.concatenate([[0.0], t]))


def bootstrap_zero_curve(maturities, coupon_rates, prices, freq: int = 2, face: float = 100.0) -> ZeroCurve:
    """Strip a zero curve from coupon bond prices, shortest maturity first.

    Each bond adds one node at its maturity. Its earlier cash flows are
    discounted on the curve built so far, linearly interpolated in the zero
    rate; when a coupon falls between the last known node and the new one,
    its rate depends on the unknown node too, so the node is solved for with
    :func:`scipy.optimize.brentq` rather than by back-substitution.

    Parameters
    ----------
    maturities : array_like, shape (n,)
        Distinct bond maturities (years).
    coupon_rates : array_like, shape (n,)
        Annual coupon rates (0 for zero-coupon bonds).
    prices : array_like, shape (n,)
        Dirty prices per `face`.
    freq : int
        Coupon frequency.
    face : float

    Returns
    -------
    ZeroCurve
        Nodes at the bond maturities.

    Examples
    --------
    Hull (9th ed., Tables 4.3-4.4):

    >>> curve = bootstrap_zero_curve([0.25, 0.5, 1.0, 1.5, 2.0], [0, 0, 0, 0.08, 0.12], [97.5, 94.9, 90.0, 96.0, 101.6])
    >>> (100 * curve.rates).round(3).tolist()
    [10.127, 10.469, 10.536, 10.681, 10.808]
    """
    T = np.asarray(maturities, dtype=np.float64)
    c = np.broadcast_to(np.asarray(coupon_rates, dtype=np.float64), T.shape)
    P = np.broadcast_to(np.asarray(prices, dtype=np.float64), T.shape)
    order = np.argsort(T)
    if np.any(np.diff(T[order]) <= 0):
        raise ValueError("maturities must be distinct")
    node_t: list[float] = []
    node_r: list[float] = []
    for k in order:
        cf = bond_cash_flows(face, float(c[k]), float(T[k]), freq)

        def mispricing(r, cf=cf, k=k):
            return cf.present_value(ZeroCurve(node_t + [float(T[k])], node_r + [r])) - P[k]

        node_r.append(float(brentq(mispricing, -0.5, 2.0, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)))
        node_t.append(float(T[k]))
    return ZeroCurve(node_t, node_r)


def bootstrap_par_curve(maturities, par_rates, freq: int = 2) -> ZeroCurve:
    """Zero curve from par yields (or swap rates): every bond prices at par.

    Examples
    --------
    A flat 5% semiannual par curve is a flat zero curve at
    :math:`2 \\ln 1.025`:

    >>> curve = bootstrap_par_curve([1, 2, 5, 10], [0.05] * 4)
    >>> bool(abs(curve.rates - 2 * np.log(1.025)).max() < 1e-9)
    True
    """
    return bootstrap_zero_curve(maturities, par_rates, 100.0, freq=freq, face=100.0)


def carleton_cooper_discount_factors(times, cash_flow_matrix, prices) -> CurveFitResult:
    r"""Discount factors by regressing bond prices on their cash flows.

    Bond :math:`i` pays :math:`C_{ik}` at the common dates :math:`t_k`,
    so :math:`P_i = \sum_k C_{ik} d_k`. With as many bonds as dates this is
    a triangular system (bootstrapping); with more bonds it is solved in
    the least-squares sense, which averages out pricing noise.

    Parameters
    ----------
    times : array_like, shape (m,)
        Payment dates shared by the bonds.
    cash_flow_matrix : array_like, shape (n, m)
        Row ``i`` holds bond ``i``'s payment on each date (0 where none).
    prices : array_like, shape (n,)

    Returns
    -------
    CurveFitResult
        ``params`` are the discount factors; ``curve`` the implied zero curve.

    Examples
    --------
    >>> res = carleton_cooper_discount_factors([1.0, 2.0], [[105.0, 0.0], [6.0, 106.0]], [100.0, 99.0])
    >>> res.params.round(6).tolist()
    [0.952381, 0.880054]
    """
    t = np.asarray(times, dtype=np.float64)
    C = np.asarray(cash_flow_matrix, dtype=np.float64)
    p = np.asarray(prices, dtype=np.float64)
    if C.shape != (p.size, t.size):
        raise ValueError("cash_flow_matrix must have shape (len(prices), len(times))")
    d, *_ = np.linalg.lstsq(C, p, rcond=None)
    return CurveFitResult(curve=ZeroCurve(t, -np.log(d) / t), params=d, residuals=C @ d - p, model="carleton_cooper")


class SplineDiscountCurve(YieldCurve):
    r"""McCulloch's spline discount function :math:`d(t) = 1 + \sum_j a_j g_j(t)`.

    The basis is the truncated power basis of the given degree,
    :math:`t, t^2, \dots, t^p` and :math:`(t - \kappa)_+^p` at each interior
    knot :math:`\kappa`, so :math:`d(0) = 1` holds by construction.

    Parameters
    ----------
    coefficients : array_like
        The :math:`a_j`, polynomial terms first.
    knots : array_like
        Interior knots (years).
    degree : int
        2 (McCulloch, 1971) or 3 (McCulloch, 1975).
    """

    def __init__(self, coefficients, knots, degree: int = 3):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.knots = np.asarray(knots, dtype=np.float64)
        self.degree = int(degree)
        if self.coefficients.size != self.degree + self.knots.size:
            raise ValueError("need degree + len(knots) coefficients")

    @staticmethod
    def basis(t, knots, degree: int) -> np.ndarray:
        """Truncated power basis at `t`, shape ``t.shape + (degree + len(knots),)``."""
        t = np.asarray(t, dtype=np.float64)[..., None]
        powers = t ** np.arange(1, degree + 1)
        trunc = np.maximum(t - np.asarray(knots, dtype=np.float64), 0.0) ** degree
        return np.concatenate([powers, trunc], axis=-1)

    def discount(self, t):
        return 1.0 + self.basis(t, self.knots, self.degree) @ self.coefficients

    def zero_rate(self, t):
        t = np.maximum(np.asarray(t, dtype=np.float64), 1e-8)
        return -np.log(self.discount(t)) / t


def fit_mcculloch_spline(bonds: list, prices, knots=None, degree: int = 3) -> CurveFitResult:
    """Fit McCulloch's spline discount function to bond prices by linear least squares.

    Because the discount function is linear in its coefficients, so is
    every bond price: :math:`P_i - \\sum_k c_{ik} = \\sum_j a_j \\sum_k c_{ik} g_j(t_{ik})`.

    Parameters
    ----------
    bonds : list of CashFlows
    prices : array_like, shape (n,)
        Dirty prices.
    knots : array_like, optional
        Interior knots. By default about :math:`\\sqrt{n} - 1` of them at
        quantiles of the maturities, McCulloch's rule of thumb.
    degree : int
        Spline degree (2 or 3).

    Returns
    -------
    CurveFitResult
        ``curve`` is a :class:`SplineDiscountCurve`; residuals are in price units.

    Examples
    --------
    >>> from financialkit.fixed_income.core.base import FlatCurve
    >>> bonds = [bond_cash_flows(100.0, 0.05, T) for T in np.arange(1, 11)]
    >>> prices = [b.present_value(FlatCurve(0.04)) for b in bonds]
    >>> res = fit_mcculloch_spline(bonds, prices)
    >>> bool(abs(res.curve.zero_rate(5.0) - 0.04) < 1e-4)
    True
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(bonds) != p.size:
        raise ValueError("need one price per bond")
    if knots is None:
        n_knots = max(int(round(np.sqrt(p.size))) - 1, 0)
        maturities = np.array([b.times[-1] for b in bonds])
        knots = np.quantile(maturities, np.linspace(0, 1, n_knots + 2)[1:-1]) if n_knots else np.empty(0)
    knots = np.asarray(knots, dtype=np.float64)
    X = np.array([b.amounts @ SplineDiscountCurve.basis(b.times, knots, degree) for b in bonds])
    y = p - np.array([b.amounts.sum() for b in bonds])
    a, *_ = np.linalg.lstsq(X, y, rcond=None)
    return CurveFitResult(curve=SplineDiscountCurve(a, knots, degree), params=a, residuals=X @ a - y, model="mcculloch_spline")


def _ns_slope(x):
    """(1 - e^{-x}) / x, equal to 1 at x = 0."""
    x = np.asarray(x, dtype=np.float64)
    safe = np.where(x == 0.0, 1.0, x)
    return np.where(x == 0.0, 1.0, -np.expm1(-safe) / safe)


def _ns_loadings(t, tau):
    """Nelson-Siegel level, slope and curvature loadings at maturities `t`."""
    x = np.asarray(t, dtype=np.float64) / tau
    slope = _ns_slope(x)
    return np.stack([np.ones_like(x), slope, slope - np.exp(-x)], axis=-1)


class NelsonSiegelCurve(YieldCurve):
    r"""Nelson and Siegel's (1987) parsimonious yield curve.

    .. math::

       r(t) = \beta_0 + \beta_1 \frac{1 - e^{-t/\tau}}{t/\tau}
              + \beta_2 \left(\frac{1 - e^{-t/\tau}}{t/\tau} - e^{-t/\tau}\right)

    :math:`\beta_0` is the long rate, :math:`\beta_0 + \beta_1` the short
    rate, :math:`\beta_2` a hump whose position :math:`\tau` sets.

    Examples
    --------
    >>> curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
    >>> round(float(curve.zero_rate(0.0)), 6), round(float(curve.zero_rate(1e6)), 6)
    (0.03, 0.05)
    """

    def __init__(self, beta0: float, beta1: float, beta2: float, tau: float):
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.beta0, self.beta1, self.beta2, self.tau = float(beta0), float(beta1), float(beta2), float(tau)

    @property
    def params(self) -> np.ndarray:
        """ndarray: ``[beta0, beta1, beta2, tau]``."""
        return np.array([self.beta0, self.beta1, self.beta2, self.tau])

    def zero_rate(self, t):
        return _ns_loadings(t, self.tau) @ np.array([self.beta0, self.beta1, self.beta2])

    def instantaneous_forward(self, t, h: float = 0.0):
        r""":math:`f(t) = \beta_0 + \beta_1 e^{-t/\tau} + \beta_2 (t/\tau) e^{-t/\tau}`, in closed form."""
        x = np.asarray(t, dtype=np.float64) / self.tau
        return self.beta0 + (self.beta1 + self.beta2 * x) * np.exp(-x)


class SvenssonCurve(YieldCurve):
    r"""Svensson's (1994) extension of Nelson-Siegel with a second hump.

    .. math::

       r(t) = \text{NS}(t; \beta_0, \beta_1, \beta_2, \tau_1)
              + \beta_3 \left(\frac{1 - e^{-t/\tau_2}}{t/\tau_2} - e^{-t/\tau_2}\right)
    """

    def __init__(self, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float, tau2: float):
        if tau1 <= 0 or tau2 <= 0:
            raise ValueError("tau1 and tau2 must be positive")
        self.beta0, self.beta1, self.beta2, self.beta3 = float(beta0), float(beta1), float(beta2), float(beta3)
        self.tau1, self.tau2 = float(tau1), float(tau2)

    @property
    def params(self) -> np.ndarray:
        """ndarray: ``[beta0, beta1, beta2, beta3, tau1, tau2]``."""
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2])

    def zero_rate(self, t):
        return _sv_loadings(t, self.tau1, self.tau2) @ np.array([self.beta0, self.beta1, self.beta2, self.beta3])

    def instantaneous_forward(self, t, h: float = 0.0):
        t = np.asarray(t, dtype=np.float64)
        x1, x2 = t / self.tau1, t / self.tau2
        return self.beta0 + (self.beta1 + self.beta2 * x1) * np.exp(-x1) + self.beta3 * x2 * np.exp(-x2)


def _sv_loadings(t, tau1, tau2):
    ns = _ns_loadings(t, tau1)
    second = _ns_loadings(t, tau2)[..., 2:]
    return np.concatenate([ns, second], axis=-1)


def _linear_betas(L, y):
    beta, *_ = np.linalg.lstsq(L, y, rcond=None)
    return beta, float(np.sum((L @ beta - y) ** 2))


def fit_nelson_siegel(tenors, yields, tau_grid=None) -> CurveFitResult:
    """Fit a :class:`NelsonSiegelCurve` to zero yields.

    For fixed :math:`\\tau` the betas are a linear least-squares problem,
    so the best :math:`\\tau` on a grid gives the starting point and
    :func:`scipy.optimize.least_squares` then refines all four parameters.

    Parameters
    ----------
    tenors, yields : array_like, shape (n,)
        Maturities and continuously compounded zero yields.
    tau_grid : array_like, optional
        Candidate decay parameters; default 60 points from 0.1 to 15 years.

    Returns
    -------
    CurveFitResult

    Examples
    --------
    >>> t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    >>> res = fit_nelson_siegel(t, NelsonSiegelCurve(0.05, -0.02, 0.015, 1.8).zero_rate(t))
    >>> res.params.round(6).tolist()
    [0.05, -0.02, 0.015, 1.8]
    """
    t = np.asarray(tenors, dtype=np.float64)
    y = np.asarray(yields, dtype=np.float64)
    grid = np.geomspace(0.1, 15.0, 60) if tau_grid is None else np.asarray(tau_grid, dtype=np.float64)
    best = min(grid, key=lambda tau: _linear_betas(_ns_loadings(t, tau), y)[1])
    beta, _ = _linear_betas(_ns_loadings(t, best), y)
    sol = least_squares(
        lambda p: _ns_loadings(t, p[3]) @ p[:3] - y,
        np.append(beta, best),
        bounds=([-np.inf] * 3 + [1e-3], [np.inf] * 3 + [50.0]),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    curve = NelsonSiegelCurve(*sol.x)
    return CurveFitResult(curve=curve, params=curve.params, residuals=curve.zero_rate(t) - y, model="nelson_siegel")


def fit_svensson(tenors, yields, tau_grid=None) -> CurveFitResult:
    """Fit a :class:`SvenssonCurve` to zero yields.

    The same two-stage scheme as :func:`fit_nelson_siegel`, with a grid
    over pairs :math:`\\tau_1 < \\tau_2` (which also keeps the two hump
    terms identifiable).

    Examples
    --------
    >>> t = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30])
    >>> true = SvenssonCurve(0.045, -0.015, -0.02, 0.03, 1.0, 6.0)
    >>> bool(fit_svensson(t, true.zero_rate(t)).rmse < 1e-7)
    True
    """
    t = np.asarray(tenors, dtype=np.float64)
    y = np.asarray(yields, dtype=np.float64)
    grid = np.geomspace(0.2, 15.0, 15) if tau_grid is None else np.asarray(tau_grid, dtype=np.float64)
    pairs = [(a, b) for a in grid for b in grid if b > 1.2 * a]
    tau1, tau2 = min(pairs, key=lambda p: _linear_betas(_sv_loadings(t, *p), y)[1])
    beta, _ = _linear_betas(_sv_loadings(t, tau1, tau2), y)
    sol = least_squares(
        lambda p: _sv_loadings(t, p[4], p[5]) @ p[:4] - y,
        np.concatenate([beta, [tau1, tau2]]),
        bounds=([-np.inf] * 4 + [1e-3, 1e-3], [np.inf] * 4 + [50.0, 50.0]),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    curve = SvenssonCurve(*sol.x)
    return CurveFitResult(curve=curve, params=curve.params, residuals=curve.zero_rate(t) - y, model="svensson")


def yield_curve_pca(changes, tenors: Optional[np.ndarray] = None) -> CurvePCAResult:
    """Principal components of yield-curve changes: Litterman and Scheinkman's level, slope and curvature.

    Parameters
    ----------
    changes : array_like, shape (n_obs, n_tenors)
        Yield changes (one row per date, one column per maturity).
    tenors : array_like, shape (n_tenors,), optional
        Maturities of the columns (for plotting); default ``0, 1, ...``.

    Returns
    -------
    CurvePCAResult

    Examples
    --------
    >>> from financialkit.fixed_income.utils.synthetic import simulate_yield_changes
    >>> tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    >>> pca = yield_curve_pca(simulate_yield_changes(tenors, 2000, rng=0), tenors)
    >>> bool(pca.explained_variance_ratio[:3].sum() > 0.95)
    True
    """
    X = np.asarray(changes, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("changes must be 2-D (n_obs, n_tenors)")
    Xc = X - X.mean(axis=0)
    w, V = np.linalg.eigh(np.cov(Xc, rowvar=False))
    order = np.argsort(w)[::-1]
    w, V = np.maximum(w[order], 0.0), V[:, order]
    for k in range(V.shape[1]):
        if k == 0:
            flip = V[:, 0].sum() < 0
        elif k == 1:
            flip = V[-1, 1] < V[0, 1]
        else:
            flip = V[np.argmax(np.abs(V[:, k])), k] < 0
        if flip:
            V[:, k] = -V[:, k]
    t = np.arange(X.shape[1], dtype=np.float64) if tenors is None else np.asarray(tenors, dtype=np.float64)
    return CurvePCAResult(tenors=t, eigenvalues=w, loadings=V, scores=Xc @ V)

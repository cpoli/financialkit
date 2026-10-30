"""Supporting numerics for financialkit.exotic_options: the bivariate normal
distribution, barrier-type parsing, and the Brownian-bridge corrections
that let Monte Carlo price continuously monitored barriers and lookbacks
without bias from a discrete grid.

Between two grid points a log-price path is a Brownian bridge, whose
probability of crossing a level and whose extremum have closed forms
(Glasserman 2004, Sec. 6.4). Applied step by step they turn a coarse
simulation into an exact one for continuous monitoring.

References: D. B. Owen, "Tables for Computing Bivariate Normal
Probabilities," *Ann. Math. Statist.* 27 (1956) 1075-1090; P. Glasserman,
*Monte Carlo Methods in Financial Engineering* (Springer, 2004),
Sections 6.4 and 3.1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import owens_t
from scipy.stats import norm

from financialkit._validation import in_interval, one_of
from financialkit.integrators import RNGLike, gbm_paths

__all__ = [
    "bivariate_normal_cdf",
    "parse_barrier_type",
    "barrier_crossing_probability",
    "bridge_extremum",
    "deterministic_hit_time",
    "spot_paths",
]

_BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def bivariate_normal_cdf(h: ArrayLike, k: ArrayLike, rho: ArrayLike) -> np.ndarray | float:
    """Standard bivariate normal distribution :math:`M(h, k; \\rho) = P(X \\le h, Y \\le k)`.

    Computed exactly from Owen's T function (:func:`scipy.special.owens_t`):

    .. math:: M(h, k; \\rho) = \\tfrac12 N(h) + \\tfrac12 N(k)
              - T\\!\\left(h, \\frac{k - \\rho h}{h \\sqrt{1 - \\rho^2}}\\right)
              - T\\!\\left(k, \\frac{h - \\rho k}{k \\sqrt{1 - \\rho^2}}\\right) - \\beta,

    with :math:`\\beta = 0` when :math:`h` and :math:`k` have the same sign
    and :math:`\\tfrac12` otherwise (Owen 1956). It broadcasts over its
    inputs, which the compound and rainbow option formulas rely on.
    Infinite limits are exact: :math:`M(\\infty, k; \\rho) = N(k)` and
    :math:`M(-\\infty, k; \\rho) = 0`.

    Parameters
    ----------
    h, k : array_like
        Upper integration limits.
    rho : array_like
        Correlation, :math:`|\\rho| < 1`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import numpy as np
    >>> bool(abs(bivariate_normal_cdf(0.0, 0.0, 0.5) - (0.25 + np.arcsin(0.5) / (2 * np.pi))) < 1e-12)
    True
    """
    h, k, rho = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (h, k, rho)))
    in_interval(-1.0, 1.0, rho=rho)
    h_inf, k_inf = np.isinf(h), np.isinf(k)
    limit = np.where((h == -np.inf) | (k == -np.inf), 0.0, np.where(h_inf, norm.cdf(k), norm.cdf(h)))
    h, k = np.where(h_inf, 1.0, h), np.where(k_inf, 1.0, k)
    # Owen's formula is singular at h = 0 or k = 0; the distribution is
    # continuous there, so evaluate an ulp-sized step to the right.
    h = np.where(h == 0.0, 1e-15, h)
    k = np.where(k == 0.0, 1e-15, k)
    s = np.sqrt(1.0 - rho**2)
    beta = np.where((h > 0) == (k > 0), 0.0, 0.5)
    out = 0.5 * norm.cdf(h) + 0.5 * norm.cdf(k) - owens_t(h, (k - rho * h) / (h * s)) - owens_t(k, (h - rho * k) / (k * s)) - beta
    out = np.where(h_inf | k_inf, limit, np.clip(out, 0.0, 1.0))
    return float(out) if out.ndim == 0 else out


def parse_barrier_type(barrier_type: str) -> tuple[bool, bool]:
    """Split ``"down-and-out"`` and friends into ``(is_down, is_out)``.

    Parameters
    ----------
    barrier_type : {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}

    Returns
    -------
    is_down, is_out : bool

    Examples
    --------
    >>> parse_barrier_type("up-and-in")
    (False, False)
    """
    one_of("barrier_type", barrier_type, _BARRIER_TYPES)
    return barrier_type.startswith("down"), barrier_type.endswith("out")


def barrier_crossing_probability(x0: ArrayLike, x1: ArrayLike, b: float, sigma: float, dt: float) -> np.ndarray:
    """Probability that a Brownian bridge from `x0` to `x1` crosses the level `b`.

    For a log-price with volatility :math:`\\sigma` over a step :math:`\\Delta t`,
    with both endpoints on the same side of :math:`b`,

    .. math:: P = \\exp\\!\\left(-\\frac{2 (x_0 - b)(x_1 - b)}{\\sigma^2 \\Delta t}\\right);

    it is 1 if either endpoint is on the far side. The drift does not
    matter once both endpoints are fixed (Glasserman 2004, Sec. 6.4).

    Parameters
    ----------
    x0, x1 : array_like
        Log-prices at the start and end of the step.
    b : float
        Log of the barrier.
    sigma : float
        Volatility.
    dt : float
        Step length in years.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np
    >>> round(float(barrier_crossing_probability(0.1, 0.1, 0.0, 0.2, 0.25)), 6)
    0.135335
    """
    d0 = np.asarray(x0, dtype=np.float64) - b
    d1 = np.asarray(x1, dtype=np.float64) - b
    same_side = d0 * d1 > 0
    with np.errstate(over="ignore"):
        p = np.exp(-2.0 * np.where(same_side, d0 * d1, 0.0) / (sigma**2 * dt))
    return np.where(same_side, p, 1.0)


def bridge_extremum(log_paths: ArrayLike, sigma: float, dt: float, kind: str = "max", rng=None) -> np.ndarray:
    """Exact samples of each path's continuous maximum or minimum given its grid values.

    Within a step from :math:`x_0` to :math:`x_1` the maximum of the
    Brownian bridge is

    .. math:: M = \\tfrac12\\left[x_0 + x_1 + \\sqrt{(x_1 - x_0)^2 - 2\\sigma^2 \\Delta t \\ln U}\\right],
              \\quad U \\sim U(0, 1],

    and the minimum is the mirror image (Glasserman 2004, Sec. 6.4). The
    path's extremum is the extremum over its steps.

    Parameters
    ----------
    log_paths : array_like, shape (n_paths, n_steps + 1)
        Log-prices on an equally spaced grid.
    sigma : float
        Volatility.
    dt : float
        Step length in years.
    kind : {"max", "min"}
    rng : numpy.random.Generator or int, optional
        Generator or seed.

    Returns
    -------
    ndarray, shape (n_paths,)
        The log of each path's running maximum (or minimum) over the whole interval.
    """
    x = np.asarray(log_paths, dtype=np.float64)
    one_of("kind", kind, ("max", "min"))
    sign = 1.0 if kind == "max" else -1.0
    x0, x1 = sign * x[:, :-1], sign * x[:, 1:]
    u = 1.0 - np.random.default_rng(rng).random(x0.shape)
    m = 0.5 * (x0 + x1 + np.sqrt((x1 - x0) ** 2 - 2.0 * sigma**2 * dt * np.log(u)))
    return sign * m.max(axis=1)


def deterministic_hit_time(S: ArrayLike, H: ArrayLike, T: ArrayLike, b: ArrayLike, is_down: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Whether, and when, the riskless path :math:`S e^{bt}` reaches the barrier :math:`H` by :math:`T`.

    With zero volatility, or at :math:`T = 0`, a path-dependent payoff is
    the payoff of this one path. It is monotone, so it reaches a barrier
    below the spot (``is_down``) only if :math:`S e^{bT} \\le H`, and then at
    :math:`t^* = \\ln(H/S)/b`; a spot of zero stays at zero.

    Parameters
    ----------
    S, H, T : array_like
        Spot (already on the live side of the barrier), barrier and maturity.
    b : array_like
        Cost of carry :math:`r - q`.
    is_down : array_like of bool
        Whether the barrier is below the spot.

    Returns
    -------
    hit : ndarray of bool
    t_hit : ndarray
        The hitting time where ``hit``, else `T`.

    Examples
    --------
    >>> hit, t = deterministic_hit_time(100.0, 90.0, 2.0, -0.1, True)
    >>> bool(hit), round(float(t), 4)
    (True, 1.0536)
    """
    S, H, T, b, is_down = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, H, T, b)), np.asarray(is_down, dtype=bool))
    fwd = S * np.exp(b * T)
    hit = np.where(is_down, fwd <= H, fwd >= H) & (S > 0)
    t_hit = np.where(hit, np.log(np.where(hit, H / np.where(hit, S, 1.0), 2.0)) / np.where(hit & (b != 0), b, 1.0), T)
    return hit, t_hit


def spot_paths(
    S: float,
    T: float,
    mu: float,
    sigma: float | None,
    n_steps: int,
    n_paths: int,
    paths: ArrayLike | None = None,
    discount: ArrayLike | None = None,
    rng: RNGLike = None,
    dW: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The spot paths a Monte Carlo pricer runs on, and their discount factors.

    Without `paths` this is :func:`financialkit.integrators.gbm_paths` with
    drift `mu`, and the pricer discounts at its constant rate. With `paths`
    the simulation is skipped, so a pricer can run on any model's paths
    (Heston, local volatility, jumps); `paths` must start at `S` and sit on
    the equally spaced grid :math:`t_i = iT/n`.

    Parameters
    ----------
    S, T, mu : float
        Spot, maturity and risk-neutral drift :math:`r - q`.
    sigma : float or None
        GBM volatility; needed only without `paths`.
    n_steps, n_paths : int
        GBM grid and path count; ignored with `paths`.
    paths : array_like, shape (n_paths, n_steps + 1), optional
        Spot paths, including the spot at :math:`t = 0`.
    discount : array_like, optional
        Discount factors :math:`D(0, t_i)` broadcastable to the shape of
        `paths`: shape ``(n_steps + 1,)`` for a term structure, or one row
        per path for stochastic rates. Needs `paths`.
    rng : numpy.random.Generator or int, optional
        Generator or seed for the GBM paths.
    dW : array_like, optional
        Brownian increments for the GBM paths.

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1)
    discount : ndarray or None
        `discount` broadcast to the shape of `paths`, or ``None`` when not given.

    Examples
    --------
    >>> paths, D = spot_paths(100.0, 1.0, 0.05, None, 2, 1, paths=[[100.0, 104.0, 98.0]], discount=[1.0, 0.98, 0.95])
    >>> paths.shape, D.shape
    ((1, 3), (1, 3))
    """
    if paths is None:
        if discount is not None:
            raise ValueError("discount needs paths; without them the pricer discounts at r")
        if sigma is None:
            raise ValueError("sigma is required unless paths are given")
        return gbm_paths(S, mu, sigma, T, int(n_steps), int(n_paths), rng=rng, dW=dW)[1], None
    P = np.asarray(paths, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError(f"paths must have shape (n_paths, n_steps + 1) with n_steps >= 1, got {P.shape}")
    if not np.allclose(P[:, 0], S):
        raise ValueError("paths must start at the spot S")
    if discount is None:
        return P, None
    try:
        D = np.broadcast_to(np.asarray(discount, dtype=np.float64), P.shape)
    except ValueError:
        raise ValueError(f"discount of shape {np.shape(discount)} does not broadcast to paths of shape {P.shape}") from None
    if np.any(D <= 0) or not np.allclose(D[:, 0], 1.0):
        raise ValueError("discount factors must be positive and start at D(0, 0) = 1")
    return P, D

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

__all__ = [
    "bivariate_normal_cdf",
    "parse_barrier_type",
    "barrier_crossing_probability",
    "bridge_extremum",
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
    if np.any(np.abs(rho) >= 1.0):
        raise ValueError("bivariate_normal_cdf needs |rho| < 1")
    # Owen's formula is singular at h = 0 or k = 0; the distribution is
    # continuous there, so evaluate an ulp-sized step to the right.
    h = np.where(h == 0.0, 1e-15, h)
    k = np.where(k == 0.0, 1e-15, k)
    s = np.sqrt(1.0 - rho**2)
    beta = np.where((h > 0) == (k > 0), 0.0, 0.5)
    out = 0.5 * norm.cdf(h) + 0.5 * norm.cdf(k) - owens_t(h, (k - rho * h) / (h * s)) - owens_t(k, (h - rho * k) / (k * s)) - beta
    out = np.clip(out, 0.0, 1.0)
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
    if barrier_type not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}, got {barrier_type!r}")
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
    if kind not in ("max", "min"):
        raise ValueError(f"kind must be 'max' or 'min', got {kind!r}")
    sign = 1.0 if kind == "max" else -1.0
    x0, x1 = sign * x[:, :-1], sign * x[:, 1:]
    u = 1.0 - np.random.default_rng(rng).random(x0.shape)
    m = 0.5 * (x0 + x1 + np.sqrt((x1 - x0) ** 2 - 2.0 * sigma**2 * dt * np.log(u)))
    return sign * m.max(axis=1)

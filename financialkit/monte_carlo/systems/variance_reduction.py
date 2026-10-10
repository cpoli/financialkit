"""Variance reduction: the same expectation from fewer samples.

* :func:`neyman_allocation` and :func:`stratified_estimate` -- Neyman's
  (1934) stratified sampling, with proportional or optimal allocation.
* :func:`importance_sampling_normal` -- Kahn and Marshall's (1953)
  importance sampling, here as a mean shift of the driving normals with
  the likelihood-ratio weight.
* :func:`antithetic_estimate` -- Hammersley and Morton's (1956) antithetic
  variates: average :math:`f(Z)` and :math:`f(-Z)`.
* :func:`latin_hypercube_estimate` -- McKay, Beckman and Conover's (1979)
  Latin hypercube sampling (:class:`scipy.stats.qmc.LatinHypercube`).
* :func:`control_variate_estimate` -- Lavenberg and Welch's (1981)
  regression control variates, with any number of controls.
* :func:`moment_match` and :func:`moment_matching_estimate` --
  Barraquand's (1995) quadratic resampling: rescale the normals to the
  exact first two moments.

Estimators whose samples are dependent (Latin hypercube, moment matching)
take their standard error from independent replications, so every
:class:`~financialkit.monte_carlo.core.base.MonteCarloResult` here is
comparable through
:meth:`~financialkit.monte_carlo.core.base.MonteCarloResult.variance_ratio`.

References: J. Neyman, "On the Two Different Aspects of the
Representative Method," *J. Royal Statistical Society* 97 (1934) 558-625;
H. Kahn and A. W. Marshall, "Methods of Reducing Sample Size in Monte
Carlo Computations," *J. Operations Research Society of America* 1 (1953)
263-278; J. M. Hammersley and K. W. Morton, "A New Monte Carlo
Technique: Antithetic Variates," *Proc. Cambridge Phil. Soc.* 52 (1956)
449-475; M. D. McKay, R. J. Beckman and W. J. Conover, "A Comparison of
Three Methods for Selecting Values of Input Variables," *Technometrics*
21 (1979) 239-245; S. S. Lavenberg and P. D. Welch, "A Perspective on
the Use of Control Variables to Increase the Efficiency of Monte Carlo
Simulations," *Management Science* 27 (1981) 322-335; J. Barraquand,
"Numerical Valuation of High Dimensional Multivariate European
Securities," *Management Science* 41 (1995) 1882-1891; Glasserman
(2004), Chapter 4.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from financialkit.integrators import RNGLike
from financialkit.monte_carlo.core.base import ControlVariateResult, MonteCarloResult, StratifiedResult
from financialkit.monte_carlo.utils.numerics import qmc_engine, sample_mean

__all__ = [
    "neyman_allocation",
    "stratified_estimate",
    "importance_sampling_normal",
    "antithetic_estimate",
    "latin_hypercube_estimate",
    "control_variate_estimate",
    "moment_match",
    "moment_matching_estimate",
]

NormalIntegrand = Callable[[np.ndarray], ArrayLike]


def _normals(gen: np.random.Generator, n: int, dim: int | None) -> np.ndarray:
    return gen.standard_normal(int(n) if dim is None else (int(n), int(dim)))


def neyman_allocation(weights: ArrayLike, stds: ArrayLike, n: int, min_per_stratum: int = 2) -> np.ndarray:
    """Neyman's optimal allocation :math:`n_h \\propto p_h \\sigma_h` of `n` samples to strata.

    It minimises :math:`\\sum_h p_h^2 \\sigma_h^2 / n_h` subject to
    :math:`\\sum_h n_h = n`. Counts are rounded by largest remainder and
    floored at `min_per_stratum` so every stratum keeps a variance
    estimate.

    Parameters
    ----------
    weights : array_like, shape (H,)
        Stratum probabilities :math:`p_h`.
    stds : array_like, shape (H,)
        Within-stratum standard deviations :math:`\\sigma_h`.
    n : int
        Total sample size.
    min_per_stratum : int
        Smallest count assigned to a stratum.

    Returns
    -------
    ndarray of int, shape (H,)
        Sums to `n` (if ``n >= H * min_per_stratum``).

    Examples
    --------
    >>> neyman_allocation([0.5, 0.5], [1.0, 3.0], 100)
    array([25, 75])
    """
    p, s = np.asarray(weights, dtype=np.float64), np.asarray(stds, dtype=np.float64)
    H, n = p.size, int(n)
    if n < H * min_per_stratum:
        raise ValueError(f"n = {n} is too small for {H} strata with {min_per_stratum} samples each")
    w = p * s
    w = p if w.sum() <= 0 else w
    exact = n * w / w.sum()
    alloc = np.maximum(np.floor(exact).astype(np.int64), min_per_stratum)
    while alloc.sum() < n:
        alloc[np.argmax(exact - alloc)] += 1
    while alloc.sum() > n:
        alloc[np.argmax(np.where(alloc > min_per_stratum, alloc - exact, -np.inf))] -= 1
    return alloc


def stratified_estimate(
    f: Callable[[np.ndarray], ArrayLike],
    n: int,
    n_strata: int = 10,
    allocation: str = "neyman",
    n_pilot: int = 200,
    rng: RNGLike = None,
    confidence: float = 0.95,
) -> StratifiedResult:
    """Stratified estimate of :math:`E[f(U)]`, :math:`U \\sim U[0, 1)`, over equal-probability strata.

    Stratum :math:`h` is :math:`[h/H, (h+1)/H)`. The estimator and its
    variance are

    .. math:: \\hat\\mu = \\sum_h p_h \\bar f_h, \\qquad \\operatorname{Var}\\hat\\mu = \\sum_h p_h^2 \\frac{s_h^2}{n_h},

    which never exceeds the crude variance under proportional allocation
    and is smallest under Neyman's allocation (Neyman 1934). A
    one-dimensional uniform drives any Black-Scholes payoff through
    ``norm.ppf``, so this stratifies the terminal price.

    Parameters
    ----------
    f : callable
        Vectorised integrand of uniforms of shape ``(n,)``.
    n : int
        Samples after the pilot.
    n_strata : int
        Number of strata :math:`H`.
    allocation : {"neyman", "proportional"}
        Neyman's allocation uses the pilot's within-stratum standard
        deviations; proportional gives :math:`n_h = n / H`.
    n_pilot : int
        Pilot samples for Neyman's allocation, spread evenly over the
        strata; their cost is included in ``n_samples``.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    StratifiedResult
    """
    gen = np.random.default_rng(rng)
    H = int(n_strata)
    p = np.full(H, 1.0 / H)

    def draw(h: int, m: int) -> np.ndarray:
        return np.asarray(f((h + gen.random(m)) / H), dtype=np.float64)

    if allocation == "proportional":
        alloc = neyman_allocation(p, np.ones(H), n)
        cost = int(n)
    elif allocation == "neyman":
        per = max(2, int(n_pilot) // H)
        pilot_std = np.array([draw(h, per).std(ddof=1) for h in range(H)])
        alloc = neyman_allocation(p, pilot_std, n)
        cost = int(n) + per * H
    else:
        raise ValueError(f"allocation must be 'neyman' or 'proportional', got {allocation!r}")
    samples = [draw(h, int(m)) for h, m in enumerate(alloc)]
    means = np.array([y.mean() for y in samples])
    stds = np.array([y.std(ddof=1) for y in samples])
    se = float(np.sqrt(np.sum(p**2 * stds**2 / alloc)))
    return StratifiedResult(float(p @ means), se, cost, confidence, f"stratified ({allocation})", alloc, means, stds)


def importance_sampling_normal(f: NormalIntegrand, n: int, shift: ArrayLike, rng: RNGLike = None, confidence: float = 0.95) -> MonteCarloResult:
    """Importance-sampling estimate of :math:`E[f(Z)]`, :math:`Z \\sim N(0, I)`, by a mean shift.

    Sampling :math:`Z \\sim N(\\theta, I)` instead and weighting by the
    likelihood ratio

    .. math:: E[f(Z)] = E_\\theta\\left[f(Z)\\,e^{-\\theta^\\top Z + |\\theta|^2/2}\\right]

    concentrates samples where :math:`f` matters (Kahn-Marshall 1953;
    Glasserman 2004, Sec. 4.6). A shift towards the strike turns a deep
    out-of-the-money option or a rare loss from a few lucky hits into a
    routine estimate.

    Parameters
    ----------
    f : callable
        Vectorised integrand of normals: shape ``(n,)`` for a scalar
        `shift`, ``(n, d)`` for a vector one.
    n : int
        Number of samples.
    shift : float or array_like, shape (d,)
        Mean :math:`\\theta` of the sampling distribution.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    :math:`P(Z > 4) \\approx 3.17 \\times 10^{-5}`, from 20,000 samples:

    >>> res = importance_sampling_normal(lambda z: (z > 4.0).astype(float), 20_000, shift=4.0, rng=0)
    >>> abs(res.estimate - 3.1671e-5) < 4 * res.std_error, res.std_error < 1e-6
    (True, True)
    """
    theta = np.asarray(shift, dtype=np.float64)
    gen = np.random.default_rng(rng)
    if theta.ndim == 0:
        z = gen.standard_normal(int(n)) + theta
        log_w = -theta * z + 0.5 * theta**2
    else:
        z = gen.standard_normal((int(n), theta.size)) + theta
        log_w = -z @ theta + 0.5 * theta @ theta
    return sample_mean(np.asarray(f(z), dtype=np.float64) * np.exp(log_w), confidence, "importance sampling")


def antithetic_estimate(f: NormalIntegrand, n_pairs: int, dim: int | None = None, rng: RNGLike = None, confidence: float = 0.95) -> MonteCarloResult:
    """Antithetic-variates estimate of :math:`E[f(Z)]`, :math:`Z \\sim N(0, I)` (Hammersley-Morton 1956).

    Each normal draw is paired with its mirror image and the estimator
    averages the pair means :math:`\\tfrac12[f(Z) + f(-Z)]`, which are
    i.i.d. Per evaluation the variance falls by the factor
    :math:`1 + \\rho`, with :math:`\\rho = \\operatorname{corr}(f(Z), f(-Z))`,
    so it helps exactly when :math:`f` is monotone.

    Parameters
    ----------
    f : callable
        Vectorised integrand of normals of shape ``(n,)`` (``dim=None``)
        or ``(n, dim)``.
    n_pairs : int
        Number of antithetic pairs; the cost is ``2 * n_pairs``.
    dim : int, optional
        Dimension of :math:`Z`.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    For a linear integrand the pairs cancel exactly:

    >>> antithetic_estimate(lambda z: 3.0 + z, 1000, rng=0).std_error < 1e-12
    True
    """
    z = _normals(np.random.default_rng(rng), n_pairs, dim)
    pairs = 0.5 * (np.asarray(f(z), dtype=np.float64) + np.asarray(f(-z), dtype=np.float64))
    return sample_mean(pairs, confidence, "antithetic", n_evaluations=2 * int(n_pairs))


def _replicate(estimates: list[float], n_per: int, confidence: float, method: str) -> MonteCarloResult:
    return sample_mean(estimates, confidence, method, n_evaluations=len(estimates) * int(n_per))


def latin_hypercube_estimate(
    f: Callable[[np.ndarray], ArrayLike], n: int, dim: int, n_replications: int = 10, rng: RNGLike = None, confidence: float = 0.95
) -> MonteCarloResult:
    """Latin hypercube estimate of :math:`\\int_{[0,1)^d} f(u)\\,du` (McKay-Beckman-Conover 1979).

    Each coordinate of the ``n`` points takes exactly one value in each of
    the ``n`` intervals :math:`[k/n, (k+1)/n)`, in an independent random
    order per coordinate, so every one-dimensional margin is stratified at
    once. The variance falls most for nearly additive integrands. The
    points of one design are dependent, so the standard error comes from
    `n_replications` independent designs.

    Parameters
    ----------
    f : callable
        Vectorised integrand of points of shape ``(n, dim)``.
    n : int
        Points per design.
    dim : int
        Dimension.
    n_replications : int
        Independent designs.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    MonteCarloResult
    """
    gen = np.random.default_rng(rng)
    est = [float(np.mean(np.asarray(f(qmc_engine("lhs", dim, rng=gen).random(int(n)))))) for _ in range(int(n_replications))]
    return _replicate(est, n, confidence, "Latin hypercube")


def control_variate_estimate(y: ArrayLike, controls: ArrayLike, control_means: ArrayLike, confidence: float = 0.95) -> ControlVariateResult:
    """Control-variate estimate of :math:`E[Y]` from controls with known means (Lavenberg-Welch 1981).

    The coefficients :math:`\\beta` are the least-squares slopes of
    :math:`Y` on the controls :math:`X`, and

    .. math:: \\hat\\mu = \\bar Y - \\beta^\\top (\\bar X - \\mu_X),
              \\qquad \\operatorname{Var}\\hat\\mu \\approx (1 - R^2) \\frac{s_Y^2}{n}.

    The standard error uses the regression residuals with ``n - k - 1``
    degrees of freedom. Discounted :math:`S_T` (mean :math:`S e^{-qT}`) is
    the classic control for an option payoff.

    Parameters
    ----------
    y : array_like, shape (n,)
        Samples of the target.
    controls : array_like, shape (n,) or (n, k)
        Samples of the controls, drawn jointly with `y`.
    control_means : float or array_like, shape (k,)
        Exact expectations of the controls.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    ControlVariateResult

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(0).standard_normal(10_000)
    >>> res = control_variate_estimate(np.exp(x), x, 0.0)
    >>> res.covers(np.exp(0.5)), res.std_error < res.crude_std_error
    (True, True)
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(controls, dtype=np.float64).reshape(y.size, -1)
    mu = np.broadcast_to(np.asarray(control_means, dtype=np.float64), (X.shape[1],))
    n, k = X.shape
    Xc = X - X.mean(axis=0)
    beta = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)[0]
    resid = (y - y.mean()) - Xc @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0
    est = float(y.mean() - beta @ (X.mean(axis=0) - mu))
    se = float(np.sqrt(resid @ resid / (n - k - 1) / n))
    crude = sample_mean(y)
    return ControlVariateResult(est, se, n, confidence, "control variate", beta, r2, crude.estimate, crude.std_error)


def moment_match(z: ArrayLike) -> np.ndarray:
    """Rescale normal draws to sample mean exactly 0 and sample variance exactly 1, per column (Barraquand 1995).

    Parameters
    ----------
    z : array_like, shape (n,) or (n, d)

    Returns
    -------
    ndarray, same shape

    Examples
    --------
    >>> import numpy as np
    >>> w = moment_match(np.random.default_rng(0).standard_normal((500, 3)))
    >>> bool(np.allclose(w.mean(axis=0), 0.0) and np.allclose(w.std(axis=0), 1.0))
    True
    """
    z = np.asarray(z, dtype=np.float64)
    return (z - z.mean(axis=0)) / z.std(axis=0)


def moment_matching_estimate(
    f: NormalIntegrand, n: int, dim: int | None = None, n_replications: int = 10, rng: RNGLike = None, confidence: float = 0.95
) -> MonteCarloResult:
    """Estimate :math:`E[f(Z)]` from moment-matched normals (Barraquand 1995).

    Each replication draws ``n`` normals, applies :func:`moment_match` and
    averages :math:`f`. Matching makes the samples dependent and the
    estimator slightly biased (:math:`O(1/n)`), so the standard error
    comes from `n_replications` independent replications.

    Parameters
    ----------
    f : callable
        Vectorised integrand of normals of shape ``(n,)`` or ``(n, dim)``.
    n : int
        Draws per replication.
    dim : int, optional
        Dimension of :math:`Z`.
    n_replications : int
        Independent replications.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    MonteCarloResult
    """
    gen = np.random.default_rng(rng)
    est = [float(np.mean(np.asarray(f(moment_match(_normals(gen, n, dim)))))) for _ in range(int(n_replications))]
    return _replicate(est, n, confidence, "moment matching")

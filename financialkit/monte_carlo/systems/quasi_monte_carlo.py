"""Quasi-Monte Carlo: low-discrepancy points, their error bound and randomization.

* :func:`van_der_corput` and :func:`halton` -- Halton's (1960) radical
  inverses in coprime bases.
* :func:`star_discrepancy`, :func:`l2_star_discrepancy` and
  :func:`koksma_hlawka_bound` -- the Koksma-Hlawka inequality (Koksma
  1942, Hlawka 1961): integration error at most variation times star
  discrepancy.
* :func:`sobol` -- Sobol's (1967) digital :math:`(t, s)`-sequences
  (:class:`scipy.stats.qmc.Sobol`).
* :func:`randomized_qmc_estimate` -- Owen's (1995) scrambling, which
  keeps the low discrepancy, makes every point uniform and gives error
  bars from independent scramblings.
* :func:`gbm_path_estimate` -- a discounted path-dependent payoff under
  Black-Scholes by pseudo-random, plain Sobol or Brownian-bridge Sobol
  increments: Paskov and Traub's (1995) high-dimensional finance
  integrals and Caflisch, Morokoff and Owen's (1997) bridge ordering.

References: J. H. Halton, "On the Efficiency of Certain Quasi-Random
Sequences of Points in Evaluating Multi-Dimensional Integrals," *Numer.
Math.* 2 (1960) 84-90; E. Hlawka, "Funktionen von beschränkter Variation
in der Theorie der Gleichverteilung," *Ann. Mat. Pura Appl.* 54 (1961)
325-333; I. M. Sobol, "On the Distribution of Points in a Cube and the
Approximate Evaluation of Integrals," *USSR Comput. Math. Math. Phys.* 7
(1967) 86-112; A. B. Owen, "Randomly Permuted (t, m, s)-Nets and (t,
s)-Sequences," in *Monte Carlo and Quasi-Monte Carlo Methods in
Scientific Computing* (Springer, 1995) 299-317; S. H. Paskov and J. F.
Traub, "Faster Valuation of Financial Derivatives," *J. Portfolio
Management* 22 (1995) 113-120; R. E. Caflisch, W. Morokoff and A. Owen,
"Valuation of Mortgage-Backed Securities Using Brownian Bridges to
Reduce Effective Dimension," *J. Computational Finance* 1 (1997) 27-46;
Glasserman (2004), Chapter 5.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import qmc

from financialkit.integrators import RNGLike, brownian_increments, gbm_paths, sobol_increments
from financialkit.monte_carlo.core.base import MonteCarloResult
from financialkit.monte_carlo.utils.numerics import qmc_engine, sample_mean

__all__ = [
    "van_der_corput",
    "halton",
    "sobol",
    "star_discrepancy",
    "l2_star_discrepancy",
    "koksma_hlawka_bound",
    "randomized_qmc_estimate",
    "gbm_path_estimate",
]


def van_der_corput(n: int, base: int = 2, start: int = 1) -> np.ndarray:
    """The radical-inverse (van der Corput) sequence in `base`.

    The :math:`k`-th point mirrors the base-`b` digits of :math:`k` about
    the radix point, :math:`k = \\sum_j a_j b^j \\mapsto \\sum_j a_j
    b^{-j-1}`, so each new point falls in the largest gap left so far.

    Parameters
    ----------
    n : int
        Number of points.
    base : int
        Integer base, at least 2.
    start : int
        Index of the first point; 1 skips the origin.

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> van_der_corput(7).tolist()
    [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    """
    if base < 2:
        raise ValueError(f"base must be at least 2, got {base}")
    k = np.arange(start, start + int(n), dtype=np.int64)
    x = np.zeros(k.size)
    scale = 1.0 / base
    while np.any(k > 0):
        k, digit = np.divmod(k, base)
        x += digit * scale
        scale /= base
    return x


def halton(n: int, dim: int, scramble: bool = False, rng: RNGLike = None) -> np.ndarray:
    """``n`` points of the ``dim``-dimensional Halton sequence (Halton 1960).

    Coordinate :math:`i` is the van der Corput sequence in the
    :math:`i`-th prime. Unscrambled points start at index 0 (the origin),
    as :class:`scipy.stats.qmc.Halton` does; neighbouring large primes
    give visibly correlated projections, which scrambling removes.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Dimension.
    scramble : bool
        Apply scipy's random digit permutation.
    rng : numpy.random.Generator or int, optional
        Seeds the scrambling.

    Returns
    -------
    ndarray, shape (n, dim)
    """
    return qmc_engine("halton", dim, scramble, rng).random(int(n))


def sobol(n: int, dim: int, scramble: bool = True, rng: RNGLike = None) -> np.ndarray:
    """``n`` points of the ``dim``-dimensional Sobol sequence (Sobol 1967), via :class:`scipy.stats.qmc.Sobol`.

    The first :math:`2^m` points form a :math:`(t, m, s)`-net: every
    elementary box of volume :math:`2^{t-m}` holds exactly :math:`2^t` of
    them. Use a power of two for `n` to keep that balance (scipy warns
    otherwise).

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Dimension, at most 21201.
    scramble : bool
        Owen-scramble the digits (Owen 1995); unscrambled points start at
        the origin.
    rng : numpy.random.Generator or int, optional
        Seeds the scrambling.

    Returns
    -------
    ndarray, shape (n, dim)

    Examples
    --------
    The first eight unscrambled points stratify each axis into eighths:

    >>> import numpy as np
    >>> x = sobol(8, 2, scramble=False)
    >>> np.sort(np.floor(8 * x[:, 1])).tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    """
    return qmc_engine("sobol", dim, scramble, rng).random(int(n))


def star_discrepancy(points: ArrayLike) -> float:
    """Exact star discrepancy :math:`D_N^*` of a point set in one or two dimensions.

    .. math:: D_N^* = \\sup_{a \\in [0,1]^d} \\left| \\frac{\\#\\{x_i \\in [0, a)\\}}{N} - \\prod_j a_j \\right|.

    In one dimension :math:`D_N^* = \\frac{1}{2N} + \\max_i |x_{(i)} -
    \\frac{2i - 1}{2N}|` (Niederreiter 1992, Thm. 2.6). In two the
    supremum is attained on the grid of point coordinates (and 1), which
    is searched with closed and open boxes in :math:`O(N^2)`; coordinates
    are assumed distinct.

    Parameters
    ----------
    points : array_like, shape (N,) or (N, d), d <= 2
        Points in :math:`[0, 1)^d`.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> N = 10
    >>> round(star_discrepancy((2 * np.arange(1, N + 1) - 1) / (2 * N)), 12)
    0.05
    """
    x = np.asarray(points, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    N, d = x.shape
    if d == 1:
        xs = np.sort(x[:, 0])
        return float(0.5 / N + np.max(np.abs(xs - (2 * np.arange(1, N + 1) - 1) / (2 * N))))
    if d != 2:
        raise ValueError(f"exact star discrepancy is implemented for d <= 2, got d = {d}; use l2_star_discrepancy")
    ox, oy = np.argsort(x[:, 0]), np.argsort(x[:, 1])
    rx, ry = np.empty(N, dtype=np.int64), np.empty(N, dtype=np.int64)
    rx[ox], ry[oy] = np.arange(N), np.arange(N)
    grid = np.zeros((N + 1, N + 1))
    grid[rx + 1, ry + 1] = 1.0
    # counts[i, j] = #{rank_x < i, rank_y < j}: the open box [0, a_i) x [0, b_j).
    counts = grid.cumsum(axis=0).cumsum(axis=1)
    a = np.concatenate([x[ox, 0], [1.0]])
    b = np.concatenate([x[oy, 1], [1.0]])
    vol = a[:, None] * b[None, :]
    nxt = np.minimum(np.arange(N + 1) + 1, N)
    closed_counts = counts[nxt][:, nxt]  # the closed box [0, a_i] x [0, b_j]
    open_counts = counts
    return float(max(np.max(vol - open_counts / N), np.max(closed_counts / N - vol)))


def l2_star_discrepancy(points: ArrayLike) -> float:
    """The :math:`L^2`-star discrepancy, in any dimension (Warnock's formula, :func:`scipy.stats.qmc.discrepancy`).

    The root-mean-square of the local discrepancy over anchored boxes; it
    never exceeds :math:`D_N^*` and is cheap in high dimension.

    Parameters
    ----------
    points : array_like, shape (N, d)

    Returns
    -------
    float
    """
    x = np.asarray(points, dtype=np.float64)
    return float(qmc.discrepancy(x[:, None] if x.ndim == 1 else x, method="L2-star"))


def koksma_hlawka_bound(points: ArrayLike, variation: float) -> float:
    """Koksma-Hlawka bound :math:`V_{HK}(f)\\,D_N^*` on the error of the equal-weight rule at `points`.

    .. math:: \\left| \\frac1N \\sum_i f(x_i) - \\int_{[0,1]^d} f \\right| \\le V_{HK}(f)\\,D_N^*(x_1, \\dots, x_N).

    Parameters
    ----------
    points : array_like, shape (N,) or (N, d), d <= 2
        The integration nodes.
    variation : float
        Hardy-Krause variation of the integrand (for example 1 for a
        monotone function on :math:`[0, 1]` with range of length 1, 3 for
        :math:`f(x, y) = xy`).

    Returns
    -------
    float
    """
    return float(variation) * star_discrepancy(points)


def randomized_qmc_estimate(
    f: Callable[[np.ndarray], ArrayLike],
    n: int,
    dim: int,
    n_replications: int = 16,
    sequence: str = "sobol",
    rng: RNGLike = None,
    confidence: float = 0.95,
) -> MonteCarloResult:
    """Randomized quasi-Monte Carlo estimate of :math:`\\int_{[0,1)^d} f(u)\\,du` (Owen 1995).

    Each replication integrates over an independently scrambled Sobol or
    Halton point set. Every scrambled point is uniform, so each
    replication is unbiased and their spread is an honest standard error;
    for smooth integrands Owen-scrambled nets reach
    :math:`O(n^{-3/2})` root-mean-square error.

    Parameters
    ----------
    f : callable
        Vectorised integrand of points of shape ``(n, dim)``.
    n : int
        Points per replication (a power of two for Sobol).
    dim : int
        Dimension.
    n_replications : int
        Independent scramblings.
    sequence : {"sobol", "halton"}
    rng : numpy.random.Generator or int, optional
        Seeds the scramblings.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> import numpy as np
    >>> res = randomized_qmc_estimate(lambda u: np.prod(2 * u, axis=1), 1024, dim=3, rng=0)
    >>> res.covers(1.0), res.std_error < 1e-3
    (True, True)
    """
    gen = np.random.default_rng(rng)
    est = [float(np.mean(np.asarray(f(qmc_engine(sequence, dim, True, gen).random(int(n)))))) for _ in range(int(n_replications))]
    return sample_mean(est, confidence, f"randomized {sequence}", n_evaluations=int(n) * int(n_replications))


def gbm_path_estimate(
    payoff: Callable[[np.ndarray], ArrayLike],
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    sampler: str = "pseudo",
    q: float = 0.0,
    n_replications: int = 8,
    rng: RNGLike = None,
    confidence: float = 0.95,
) -> MonteCarloResult:
    """Discounted path-dependent payoff under Black-Scholes, by pseudo- or quasi-random paths.

    Paths are exact GBM on ``n_steps`` dates from
    :func:`financialkit.integrators.gbm_paths`, so the integral has
    dimension ``n_steps``. The increments come from
    :func:`~financialkit.integrators.brownian_increments` (``"pseudo"``),
    from scrambled Sobol points with coordinate ``i`` driving step ``i``
    (``"sobol"``, Paskov-Traub 1995), or from the same points assembled by
    Brownian bridge so the leading coordinates set the coarse shape
    (``"bridge"``, Caflisch-Morokoff-Owen 1997). Every sampler is
    replicated `n_replications` times with fresh seeds and the standard
    error is taken across replications, so the three are compared on one
    footing.

    Parameters
    ----------
    payoff : callable
        Undiscounted payoff of the price paths of shape
        ``(n_paths, n_steps + 1)`` (column 0 is ``S``).
    S, T, r, sigma : float
        Spot, maturity, risk-free rate and volatility.
    n_steps : int
        Monitoring dates, the dimension of the integral.
    n_paths : int
        Paths per replication (a power of two for the Sobol samplers).
    sampler : {"pseudo", "sobol", "bridge"}
    q : float
        Continuous dividend yield.
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
    if sampler not in ("pseudo", "sobol", "bridge"):
        raise ValueError(f"sampler must be 'pseudo', 'sobol' or 'bridge', got {sampler!r}")
    gen = np.random.default_rng(rng)
    est = []
    for _ in range(int(n_replications)):
        if sampler == "pseudo":
            dW = brownian_increments(T, n_steps, n_paths, rng=gen)
        else:
            dW = sobol_increments(T, n_steps, n_paths, bridge=sampler == "bridge", rng=gen)
        _, paths = gbm_paths(S, r - q, sigma, T, n_steps, n_paths, dW=dW)
        est.append(float(np.exp(-r * T) * np.mean(np.asarray(payoff(paths)))))
    return sample_mean(est, confidence, sampler, n_evaluations=int(n_paths) * int(n_replications))

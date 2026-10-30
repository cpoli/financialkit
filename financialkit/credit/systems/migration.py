"""Rating migration: CreditMetrics, generator matrices, and finding a valid generator.

CreditMetrics (J. P. Morgan 1997) values a bond in every rating it could
migrate to over a year, weights the values with the rating agency's
transition matrix, and reads credit VaR off the resulting distribution.
Joint migrations come from correlated asset returns: an obligor moves to
the rating whose band of standard-normal thresholds its asset return
falls in.

Jarrow, Lando and Turnbull (1997) moved migration to continuous time: a
generator :math:`\\Lambda` (rows summing to zero, non-negative off the
diagonal) gives :math:`P(t) = e^{\\Lambda t}` for every horizon, computed
with :func:`scipy.linalg.expm`. Going back from an annual matrix with
:func:`scipy.linalg.logm` often fails: Israel, Rosenthal and Wei (2001)
showed that the logarithm of an empirical matrix usually has small
negative off-diagonal entries, and proposed zeroing them and restoring
the row sums.

References: J. P. Morgan, *CreditMetrics -- Technical Document* (1997);
R. A. Jarrow, D. Lando and S. M. Turnbull, "A Markov Model for the Term
Structure of Credit Risk Spreads," *Rev. Financial Stud.* 10 (1997)
481-523; R. B. Israel, J. S. Rosenthal and J. Z. Wei, "Finding
Generators for Markov Chains via Empirical Transition Matrices, with
Applications to Credit Ratings," *Math. Finance* 11 (2001) 245-265; A.
Kreinin and M. Sidelnikova, "Regularization Algorithms for Transition
Matrices," *Algo Research Quarterly* 4 (2001) 23-40.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg, stats

from financialkit._deprecation import renamed_parameter
from financialkit.credit.core.base import CreditMetricsResult
from financialkit.credit.utils.discounting import discount_factors
from financialkit.credit.utils.matrices import check_transition_matrix
from financialkit.fixed_income import CurveLike
from financialkit.integrators import RNGLike

__all__ = [
    "migration_thresholds",
    "creditmetrics_bond",
    "joint_migration_probabilities",
    "simulate_migrations",
    "creditmetrics_portfolio",
    "generator_from_transition",
    "transition_from_generator",
    "is_valid_generator",
    "regularize_generator",
    "jlt_risky_zero",
]


def migration_thresholds(row: ArrayLike) -> np.ndarray:
    """Asset-return thresholds that reproduce one row of a transition matrix.

    States are ordered best first and default last, as in the matrix.
    An asset return :math:`Z` below ``thresholds[-1]`` means default,
    between ``thresholds[j]`` and ``thresholds[j - 1]`` means rating ``j``.

    Parameters
    ----------
    row : array_like, shape (k,)
        Migration probabilities from the current rating.

    Returns
    -------
    ndarray, shape (k + 1,)
        Decreasing thresholds, from ``+inf`` down to ``-inf``.

    Examples
    --------
    A BBB obligor defaults below about -2.91 standard deviations:

    >>> from financialkit.credit.utils.matrices import creditmetrics_matrix
    >>> round(float(migration_thresholds(creditmetrics_matrix()[3])[-2]), 2)
    -2.91
    """
    p = np.asarray(row, dtype=np.float64)
    p = p / p.sum()
    tail = np.cumsum(p[::-1])[::-1]  # P(rating j or worse)
    return np.r_[np.inf, stats.norm.ppf(np.clip(tail[1:], 0.0, 1.0)), -np.inf]


def creditmetrics_bond(probabilities: ArrayLike, values: ArrayLike) -> CreditMetricsResult:
    """Value distribution of one bond over its possible ratings in a year.

    Parameters
    ----------
    probabilities : array_like, shape (k,)
        Migration probabilities (a row of the transition matrix).
    values : array_like, shape (k,)
        Bond value in each end-of-year rating, default included.

    Returns
    -------
    CreditMetricsResult

    Examples
    --------
    CreditMetrics Technical Document, Chapter 3: the five-year BBB bond
    has mean value 107.09 and standard deviation 2.99.

    >>> from financialkit.credit.utils.matrices import creditmetrics_bbb_values, creditmetrics_matrix
    >>> res = creditmetrics_bond(creditmetrics_matrix(normalize=False)[3], creditmetrics_bbb_values())
    >>> round(res.mean, 2), round(res.std, 2)
    (107.09, 2.99)
    """
    p = np.asarray(probabilities, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if p.shape != v.shape:
        raise ValueError("probabilities and values must have the same shape")
    return CreditMetricsResult(values=v, probabilities=p / p.sum())


def joint_migration_probabilities(row1: ArrayLike, row2: ArrayLike, rho: float) -> np.ndarray:
    """Joint one-year migration probabilities of two obligors with asset correlation ``rho``.

    Each cell is a bivariate-normal rectangle probability between the two
    obligors' thresholds, computed with
    :data:`scipy.stats.multivariate_normal`.

    Parameters
    ----------
    row1, row2 : array_like, shape (k1,) and (k2,)
        Migration probabilities of each obligor.
    rho : float
        Asset-return correlation.

    Returns
    -------
    ndarray, shape (k1, k2)
        Rows and columns sum to the two marginals.
    """
    t1 = np.clip(migration_thresholds(row1), -12.0, 12.0)
    t2 = np.clip(migration_thresholds(row2), -12.0, 12.0)
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    X, Y = np.meshgrid(t1, t2, indexing="ij")
    F = mvn.cdf(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(X.shape)
    # F[i, j] = P(Z1 < t1[i], Z2 < t2[j]); thresholds decrease, so rectangles are a double difference.
    return np.clip(F[:-1, :-1] - F[1:, :-1] - F[:-1, 1:] + F[1:, 1:], 0.0, None)


@renamed_parameter("n_sims", "n_paths", "financialkit.credit.simulate_migrations")
def simulate_migrations(P: ArrayLike, ratings: ArrayLike, rho: float, n_paths: int, rng: RNGLike = None) -> np.ndarray:
    """Simulate end-of-year ratings with one-factor Gaussian asset correlation.

    Asset returns are :math:`Z_i = \\sqrt\\rho\\, M + \\sqrt{1 - \\rho}\\,\\varepsilon_i`.

    Parameters
    ----------
    P : array_like, shape (k, k)
        Transition matrix.
    ratings : array_like of int, shape (m,)
        Current rating index of each obligor.
    rho : float
        Pairwise asset correlation.
    n_paths : int
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray of int, shape (n_paths, m)
    """
    P = check_transition_matrix(P, atol=1e-3)
    ratings = np.asarray(ratings, dtype=np.intp)
    rng = np.random.default_rng(rng)
    M = rng.standard_normal((n_paths, 1))
    Z = np.sqrt(rho) * M + np.sqrt(1.0 - rho) * rng.standard_normal((n_paths, ratings.size))
    out = np.empty(Z.shape, dtype=np.intp)
    for j, g in enumerate(ratings):
        t = migration_thresholds(P[g])
        out[:, j] = np.searchsorted(-t, -Z[:, j], side="right") - 1
    return out


@renamed_parameter("n_sims", "n_paths", "financialkit.credit.creditmetrics_portfolio")
def creditmetrics_portfolio(P: ArrayLike, ratings: ArrayLike, values: ArrayLike, rho: float, n_paths: int, rng: RNGLike = None) -> CreditMetricsResult:
    """Monte Carlo CreditMetrics value distribution of a bond portfolio.

    Parameters
    ----------
    P : array_like, shape (k, k)
    ratings : array_like of int, shape (m,)
        Current rating of each bond.
    values : array_like, shape (k,) or (m, k)
        Value of each bond in each end-of-year rating.
    rho : float
        Asset correlation.
    n_paths : int
    rng : numpy.random.Generator, int or None

    Returns
    -------
    CreditMetricsResult
        Equally weighted portfolio-value scenarios.
    """
    ratings = np.asarray(ratings, dtype=np.intp)
    V = np.broadcast_to(np.asarray(values, dtype=np.float64), (ratings.size, np.shape(P)[0]))
    new = simulate_migrations(P, ratings, rho, n_paths, rng)
    total = V[np.arange(ratings.size), new].sum(axis=1)
    return CreditMetricsResult(values=total, probabilities=np.full(n_paths, 1.0 / n_paths))


def generator_from_transition(P: ArrayLike, t: float = 1.0) -> np.ndarray:
    """Candidate generator :math:`\\log(P) / t` (principal matrix logarithm).

    The result has zero row sums but may have negative off-diagonal
    entries; see :func:`regularize_generator`.

    Parameters
    ----------
    P : array_like, shape (k, k)
    t : float
        Horizon of ``P`` in years.

    Returns
    -------
    ndarray, shape (k, k)
    """
    P = check_transition_matrix(P)
    if np.any(np.diag(P) <= 0) or np.linalg.det(P) <= 0:
        raise ValueError("the principal logarithm needs a positive diagonal and determinant")
    return np.real(linalg.logm(P)) / t


def transition_from_generator(Q: ArrayLike, t: ArrayLike = 1.0) -> np.ndarray:
    """Transition matrix :math:`P(t) = e^{Qt}` by :func:`scipy.linalg.expm`.

    Parameters
    ----------
    Q : array_like, shape (k, k)
    t : float or array_like
        Horizon(s).

    Returns
    -------
    ndarray, shape (k, k) or (n_t, k, k)

    Examples
    --------
    >>> Q = np.array([[-0.1, 0.1], [0.0, 0.0]])
    >>> bool(np.isclose(transition_from_generator(Q, 2.0)[0, 1], 1 - np.exp(-0.2)))
    True
    """
    Q = np.asarray(Q, dtype=np.float64)
    ts = np.asarray(t, dtype=np.float64)
    if ts.ndim == 0:
        return linalg.expm(Q * float(ts))
    return np.stack([linalg.expm(Q * float(s)) for s in ts])


def is_valid_generator(Q: ArrayLike, atol: float = 1e-12) -> bool:
    """Whether ``Q`` has zero row sums and non-negative off-diagonal entries.

    Parameters
    ----------
    Q : array_like, shape (k, k)
    atol : float

    Returns
    -------
    bool
    """
    Q = np.asarray(Q, dtype=np.float64)
    off = Q - np.diag(np.diag(Q))
    return bool(np.all(off >= -atol) and np.allclose(Q.sum(axis=1), 0.0, atol=max(atol, 1e-10)))


def regularize_generator(Q: ArrayLike, method: str = "diagonal") -> np.ndarray:
    """Turn a candidate generator into a valid one.

    - ``"diagonal"`` (Israel, Rosenthal and Wei 2001): set negative
      off-diagonal entries to zero and absorb the change in the diagonal.
    - ``"weighted"`` (Kreinin and Sidelnikova 2001): set them to zero and
      take the excess from every remaining non-zero entry of the row in
      proportion to its absolute value.

    Parameters
    ----------
    Q : array_like, shape (k, k)
    method : {"diagonal", "weighted"}

    Returns
    -------
    ndarray, shape (k, k)
    """
    Q = np.array(Q, dtype=np.float64)
    k = Q.shape[0]
    eye = np.eye(k, dtype=bool)
    if method == "diagonal":
        Q[(Q < 0) & ~eye] = 0.0
        Q[eye] = 0.0
        Q[eye] = -Q.sum(axis=1)
        return Q
    if method != "weighted":
        raise ValueError(f'method must be "diagonal" or "weighted", got {method!r}')
    for i in range(k):
        row = Q[i]
        g = abs(row[i]) + np.sum(np.maximum(row[~eye[i]], 0.0))
        b = np.sum(np.maximum(-row[~eye[i]], 0.0))
        neg = (row < 0) & ~eye[i]
        row[neg] = 0.0
        if g > 0:
            row[(row != 0)] -= b * np.abs(row[(row != 0)]) / g
        row[~eye[i]] = np.maximum(row[~eye[i]], 0.0)
        row[i] = -np.sum(row[~eye[i]])
    return Q


def jlt_risky_zero(T: ArrayLike, r: CurveLike, Q: ArrayLike, rating: int, recovery: float = 0.4, premia: ArrayLike | None = None) -> np.ndarray | float:
    """Jarrow-Lando-Turnbull zero-coupon bond of an issuer with a given rating.

    Under the pricing measure the generator is :math:`\\tilde Q = U Q`
    with :math:`U = \\operatorname{diag}(\\mu)` the risk premia (``premia``,
    ones if omitted). With recovery of treasury :math:`\\delta`,

    .. math:: P^d(0, T) = P(0, T)\\,[\\delta + (1 - \\delta)(1 - \\tilde P_{i,D}(T))].

    Parameters
    ----------
    T : array_like
        Maturities.
    r : float or YieldCurve
        Continuously compounded risk-free rate, or a ``fixed_income`` discount curve.
    Q : array_like, shape (k, k)
        Generator with default as the last, absorbing state.
    rating : int
        Index of the issuer's current rating.
    recovery : float
    premia : array_like, shape (k,), optional
        Risk-premium multipliers per rating.

    Returns
    -------
    float or ndarray
    """
    Q = np.asarray(Q, dtype=np.float64)
    if premia is not None:
        Q = np.asarray(premia, dtype=np.float64)[:, None] * Q
    Ts = np.atleast_1d(np.asarray(T, dtype=np.float64))
    pd = np.array([linalg.expm(Q * t)[rating, -1] for t in Ts])
    out = discount_factors(r, Ts) * (recovery + (1.0 - recovery) * (1.0 - pd))
    return float(out[0]) if np.ndim(T) == 0 else out

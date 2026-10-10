"""Rating-transition data and checks shared by the migration and CreditMetrics modules.

:func:`creditmetrics_matrix` is the one-year transition matrix printed in
the CreditMetrics Technical Document (Table 6.1, from Standard & Poor's
1996 data), with ratings ``AAA, AA, A, BBB, BB, B, CCC`` and default
``D`` absorbing.

References: J. P. Morgan, *CreditMetrics -- Technical Document* (1997),
Table 6.1 and the BBB bond example of Chapter 3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["RATINGS", "creditmetrics_matrix", "creditmetrics_bbb_values", "check_transition_matrix"]

RATINGS = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D")
"""Rating labels of :func:`creditmetrics_matrix`, best first, default last."""

_CM = [
    [90.81, 8.33, 0.68, 0.06, 0.12, 0.00, 0.00, 0.00],
    [0.70, 90.65, 7.79, 0.64, 0.06, 0.14, 0.02, 0.00],
    [0.09, 2.27, 91.05, 5.52, 0.74, 0.26, 0.01, 0.06],
    [0.02, 0.33, 5.95, 86.93, 5.30, 1.17, 0.12, 0.18],
    [0.03, 0.14, 0.67, 7.73, 80.53, 8.84, 1.00, 1.06],
    [0.00, 0.11, 0.24, 0.43, 6.48, 83.46, 4.07, 5.20],
    [0.22, 0.00, 0.22, 1.30, 2.38, 11.24, 64.86, 19.79],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 100.0],
]


def creditmetrics_matrix(normalize: bool = True) -> np.ndarray:
    """The CreditMetrics one-year rating transition matrix.

    Parameters
    ----------
    normalize : bool
        Rescale rows to sum to one exactly (the published percentages are
        rounded, and two rows sum to 99.99% and 100.01%).

    Returns
    -------
    ndarray, shape (8, 8)
        Row ``i`` holds the probabilities of moving from ``RATINGS[i]`` to each rating.

    Examples
    --------
    >>> P = creditmetrics_matrix()
    >>> round(float(P[3, 7]), 4)
    0.0018
    """
    P = np.array(_CM) / 100.0
    return P / P.sum(axis=1, keepdims=True) if normalize else P


def creditmetrics_bbb_values() -> np.ndarray:
    """One-year forward values of CreditMetrics' five-year 6% BBB bond in each rating.

    Returns
    -------
    ndarray, shape (8,)
        Values per 100 face, ordered as :data:`RATINGS` (default value is the
        mean senior-unsecured recovery, 51.13).
    """
    return np.array([109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13])


def check_transition_matrix(P: ArrayLike, atol: float = 1e-8) -> np.ndarray:
    """Validate a stochastic matrix: square, non-negative, rows summing to one.

    Parameters
    ----------
    P : array_like, shape (k, k)
    atol : float

    Returns
    -------
    ndarray
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("a transition matrix must be square")
    if np.any(P < -atol) or not np.allclose(P.sum(axis=1), 1.0, atol=atol):
        raise ValueError("a transition matrix must be non-negative with rows summing to one")
    return P

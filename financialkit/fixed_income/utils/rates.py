"""Yield derivatives of the discount factor, for duration and convexity.

The discount factor itself is :func:`financialkit.time_value.discount_factor`.

The year fractions behind ``t`` are taken as given: converting dates to
year fractions under a day-count convention is
:mod:`financialkit.time_value`'s job.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["yield_discount_derivatives"]


def _check_compounding(compounding: Optional[int]) -> None:
    if compounding is not None and int(compounding) <= 0:
        raise ValueError("compounding must be a positive integer or None (continuous)")


def yield_discount_derivatives(y, t, compounding: Optional[int] = None):
    r"""First and second derivatives of :func:`~financialkit.time_value.discount_factor` with respect to the yield `y`.

    Continuous: :math:`-t e^{-yt}` and :math:`t^2 e^{-yt}`. Periodic:
    :math:`-t (1+y/m)^{-mt-1}` and :math:`t (t + 1/m) (1+y/m)^{-mt-2}`.

    Returns
    -------
    d1, d2 : ndarray
    """
    _check_compounding(compounding)
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if compounding is None:
        df = np.exp(-y * t)
        return -t * df, t**2 * df
    m = float(compounding)
    base = 1.0 + y / m
    return -t * base ** (-m * t - 1.0), t * (t + 1.0 / m) * base ** (-m * t - 2.0)

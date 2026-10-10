"""Small cash-flow helpers shared by the present-value and IRR modules."""

from __future__ import annotations

import numpy as np

__all__ = ["as_cash_flows", "sign_changes"]


def as_cash_flows(cash_flows, times=None) -> tuple[np.ndarray, np.ndarray]:
    """Validate a cash-flow stream and its payment times.

    Parameters
    ----------
    cash_flows : array-like, shape (n,)
        Amounts, negative for outflows.
    times : array-like, shape (n,), optional
        Payment times in years (or periods). Defaults to ``0, 1, ..., n - 1``.

    Returns
    -------
    cash_flows, times : ndarray, shape (n,)
    """
    c = np.atleast_1d(np.asarray(cash_flows, dtype=np.float64))
    if c.ndim != 1 or c.size == 0:
        raise ValueError("cash_flows must be a non-empty 1-D array")
    t = np.arange(c.size, dtype=np.float64) if times is None else np.atleast_1d(np.asarray(times, dtype=np.float64))
    if t.shape != c.shape:
        raise ValueError("times must have the same shape as cash_flows")
    return c, t


def sign_changes(cash_flows) -> int:
    """Number of sign changes in a cash-flow stream, ignoring zeros.

    By Descartes' rule of signs this bounds the number of positive roots
    of the NPV polynomial in the discount factor, i.e. the number of
    internal rates of return above -100%.

    Parameters
    ----------
    cash_flows : array-like

    Returns
    -------
    int

    Examples
    --------
    >>> sign_changes([-1600, 10000, -10000])
    2
    """
    s = np.sign(np.asarray(cash_flows, dtype=np.float64))
    s = s[s != 0]
    return int(np.count_nonzero(s[1:] != s[:-1]))

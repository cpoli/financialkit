"""Shared input validation, so every pricer rejects bad input with the same message.

Each check takes its inputs as keywords, so the error can name them, and
raises :class:`ValueError` on the first input with a bad entry, quoting
that entry: ``"S must be positive, got -1"``. Inputs may be scalars or
arrays. NaN passes the sign and interval checks (a vectorized caller may
mask with it); :func:`finite` is the check that rejects it.

Examples
--------
>>> positive(S=100.0, K=[90.0, 110.0])
>>> nonnegative(T=[1.0, -0.5])
Traceback (most recent call last):
    ...
ValueError: T must be non-negative, got -0.5
>>> option_type("put")
-1
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["positive", "nonnegative", "finite", "in_interval", "one_of", "option_type"]

_BRACKETS = {"neither": ("(", ")"), "left": ("[", ")"), "right": ("(", "]"), "both": ("[", "]")}


def _raise_first(name: str, x: np.ndarray, bad: np.ndarray, requirement: str) -> None:
    if np.any(bad):
        raise ValueError(f"{name} must {requirement}, got {x[bad].flat[0]:.6g}")


def positive(**inputs: ArrayLike) -> None:
    """Raise :class:`ValueError` unless every entry of every input is above zero.

    Parameters
    ----------
    **inputs : array_like
        Named inputs, for example ``S=S, K=K``.

    Raises
    ------
    ValueError
        Naming the first input with an entry at or below zero.

    Examples
    --------
    >>> positive(K=0.0)
    Traceback (most recent call last):
        ...
    ValueError: K must be positive, got 0
    """
    for name, x in inputs.items():
        x = np.asarray(x, dtype=np.float64)
        _raise_first(name, x, x <= 0, "be positive")


def nonnegative(**inputs: ArrayLike) -> None:
    """Raise :class:`ValueError` unless every entry of every input is at least zero.

    Parameters
    ----------
    **inputs : array_like
        Named inputs, for example ``T=T, sigma=sigma``.

    Raises
    ------
    ValueError
        Naming the first input with a negative entry.

    Examples
    --------
    >>> nonnegative(sigma=-0.2)
    Traceback (most recent call last):
        ...
    ValueError: sigma must be non-negative, got -0.2
    """
    for name, x in inputs.items():
        x = np.asarray(x, dtype=np.float64)
        _raise_first(name, x, x < 0, "be non-negative")


def finite(**inputs: ArrayLike) -> None:
    """Raise :class:`ValueError` unless every entry of every input is finite.

    Parameters
    ----------
    **inputs : array_like
        Named inputs, for example ``r=r, q=q``.

    Raises
    ------
    ValueError
        Naming the first input with a NaN or infinite entry.

    Examples
    --------
    >>> finite(r=float("nan"))
    Traceback (most recent call last):
        ...
    ValueError: r must be finite, got nan
    """
    for name, x in inputs.items():
        x = np.asarray(x, dtype=np.float64)
        _raise_first(name, x, ~np.isfinite(x), "be finite")


def in_interval(low: float, high: float, /, *, closed: str = "neither", **inputs: ArrayLike) -> None:
    """Raise :class:`ValueError` unless every entry of every input lies between `low` and `high`.

    Parameters
    ----------
    low, high : float
        The interval's end points.
    closed : {"neither", "left", "right", "both"}
        Which end points belong to the interval.
    **inputs : array_like
        Named inputs, for example ``rho=rho``.

    Raises
    ------
    ValueError
        Naming the first input with an entry outside the interval.

    Examples
    --------
    >>> in_interval(-1, 1, rho=[0.5, 1.0])
    Traceback (most recent call last):
        ...
    ValueError: rho must lie in (-1, 1), got 1
    >>> in_interval(0, 1, closed="both", p=1.0)
    """
    one_of("closed", closed, tuple(_BRACKETS))
    left, right = _BRACKETS[closed]
    for name, x in inputs.items():
        x = np.asarray(x, dtype=np.float64)
        below = x < low if left == "[" else x <= low
        above = x > high if right == "]" else x >= high
        _raise_first(name, x, below | above, f"lie in {left}{low:g}, {high:g}{right}")


def one_of(name: str, value: object, choices: Sequence[object]) -> None:
    """Raise :class:`ValueError` unless `value` is one of `choices`.

    Parameters
    ----------
    name : str
        The argument's name, quoted in the message.
    value : object
    choices : sequence

    Raises
    ------
    ValueError

    Examples
    --------
    >>> one_of("kind", "median", ("max", "min"))
    Traceback (most recent call last):
        ...
    ValueError: kind must be 'max' or 'min', got 'median'
    """
    if value not in choices:
        quoted = [repr(c) for c in choices]
        allowed = " or ".join(quoted) if len(quoted) <= 2 else "one of " + ", ".join(quoted)
        raise ValueError(f"{name} must be {allowed}, got {value!r}")


def option_type(value: str, name: str = "option_type") -> int:
    """Check a call/put flag and return its sign, ``+1`` for a call and ``-1`` for a put.

    Parameters
    ----------
    value : {"call", "put"}
    name : str
        The argument's name, quoted in the message.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If `value` is neither ``"call"`` nor ``"put"``.

    Examples
    --------
    >>> option_type("call")
    1
    >>> option_type("straddle", name="underlying_type")
    Traceback (most recent call last):
        ...
    ValueError: underlying_type must be 'call' or 'put', got 'straddle'
    """
    one_of(name, value, ("call", "put"))
    return 1 if value == "call" else -1

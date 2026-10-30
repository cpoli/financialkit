"""Shared helpers for the options pricers: input validation, the forward,
Black's :math:`d_1, d_2`, payoffs, and finite-difference Greeks of any
pricer, used to check the analytic ones.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from financialkit import _validation
from financialkit.options.core.base import Greeks

__all__ = ["option_sign", "as_output", "check_nonnegative", "forward_price", "black_d1_d2", "payoff", "bump_greek", "finite_difference_greeks"]


def option_sign(option_type: str) -> int:
    """``+1`` for a call, ``-1`` for a put; see ``financialkit._validation.option_type``.

    Parameters
    ----------
    option_type : {"call", "put"}

    Returns
    -------
    int
    """
    return _validation.option_type(option_type)


def as_output(x: ArrayLike) -> np.ndarray | float:
    """Return a Python float for a 0-d result, the array otherwise."""
    x = np.asarray(x, dtype=np.float64)
    return float(x) if x.ndim == 0 else x


def check_nonnegative(**inputs: ArrayLike) -> None:
    """Raise :class:`ValueError` naming the first input with a negative entry; see ``financialkit._validation.nonnegative``.

    Parameters
    ----------
    **inputs : array_like
        Named inputs, for example ``S=S, T=T``.

    Raises
    ------
    ValueError
        If any entry of any input is below zero.

    Examples
    --------
    >>> check_nonnegative(S=100.0, T=0.0)
    >>> check_nonnegative(S=-1.0)
    Traceback (most recent call last):
        ...
    ValueError: S must be non-negative, got -1
    """
    _validation.nonnegative(**inputs)


def forward_price(S: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0) -> np.ndarray | float:
    """Forward price :math:`F = S e^{(r - q) T}`.

    Parameters
    ----------
    S, T, r, q : array_like
        Spot, maturity, risk-free rate and dividend yield.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(forward_price(100.0, 1.0, 0.05), 4)
    105.1271
    """
    S, T, r, q = (np.asarray(a, dtype=np.float64) for a in (S, T, r, q))
    return as_output(S * np.exp((r - q) * T))


def black_d1_d2(F: ArrayLike, K: ArrayLike, sigma: ArrayLike, T: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    r"""Black's :math:`d_{1,2} = [\ln(F/K) \pm \tfrac12 \sigma^2 T] / (\sigma\sqrt{T})`.

    When the total volatility :math:`\sigma\sqrt{T}` is zero, both are
    their limits: :math:`+\infty` in the money (:math:`F > K`),
    :math:`-\infty` out of it and 0 at the money, so :math:`N(d_{1,2})` is
    the probability of exercise, 1, 0 or 1/2. ``F = 0`` gives
    :math:`-\infty` and ``K = 0`` gives :math:`+\infty`.

    Parameters
    ----------
    F, K, sigma, T : array_like
        Forward, strike, volatility and maturity, all non-negative.

    Returns
    -------
    d1, d2 : ndarray

    Examples
    --------
    >>> d1, d2 = black_d1_d2([110.0, 100.0, 90.0], 100.0, 0.2, 0.0)
    >>> d1.tolist(), d2.tolist()
    ([inf, 0.0, -inf], [inf, 0.0, -inf])
    """
    F, K, sigma, T = (np.asarray(a, dtype=np.float64) for a in (F, K, sigma, T))
    v = sigma * np.sqrt(T)
    positive = (F > 0) & (K > 0)
    log_fk = np.where(positive, np.log(np.where(positive, F / np.where(K > 0, K, 1.0), 1.0)), np.where(F > 0, np.inf, -np.inf))
    limit = np.where(log_fk > 0, np.inf, np.where(log_fk < 0, -np.inf, 0.0))
    d1 = np.where(v > 0, (log_fk + 0.5 * v**2) / np.where(v > 0, v, 1.0), limit)
    return d1, d1 - v


def payoff(S: ArrayLike, K: ArrayLike, option_type: str = "call") -> np.ndarray | float:
    """Exercise value :math:`\\max(\\pm(S - K), 0)`.

    Parameters
    ----------
    S, K : array_like
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray
    """
    sign = option_sign(option_type)
    return as_output(np.maximum(sign * (np.asarray(S, dtype=np.float64) - np.asarray(K, dtype=np.float64)), 0.0))


def finite_difference_greeks(
    pricer: Callable[..., ArrayLike],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    h: float = 1e-4,
) -> Greeks:
    """All the Black-Scholes Greeks of ``pricer`` by central finite differences.

    Used to check analytic Greeks, and to get Greeks from any pricer with
    the ``pricer(S, K, T, r, sigma, q, option_type)`` signature. Each Greek
    is one :func:`bump_greek`; call that directly for other pricers, other
    parameters or Monte Carlo pricers.

    Parameters
    ----------
    pricer : callable
        ``pricer(S, K, T, r, sigma, q, option_type) -> price``.
    S, K, T, r, sigma, q : float
    option_type : {"call", "put"}
    h : float
        Relative bump for ``S`` and absolute bump for the other inputs.

    Returns
    -------
    Greeks
    """

    def V(S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str) -> ArrayLike:
        return pricer(S, K, T, r, sigma, q, option_type)

    bumps = {"S": h * S, "T": h, "r": h, "sigma": h, "q": h}

    def d(wrt: str | tuple[str, str]) -> float:
        return float(bump_greek(V, wrt, S, K, T, r, sigma, q, option_type, h=bumps))

    return Greeks(
        delta=d("S"),
        gamma=d(("S", "S")),
        vega=d("sigma"),
        theta=-d("T"),
        rho=d("r"),
        epsilon=d("q"),
        vanna=d(("S", "sigma")),
        volga=d(("sigma", "sigma")),
    )


def _bump(arguments: dict[str, Any], name: str, dx: float) -> dict[str, Any]:
    """Copy of `arguments` with `name` (an argument, or ``"arg.attr"``) moved by `dx`."""
    head, _, attr = name.partition(".")
    out = dict(arguments)
    if attr:
        obj = out[head]
        value = getattr(obj, attr) + dx
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            out[head] = dataclasses.replace(obj, **{attr: value})
        else:
            out[head] = copy.copy(obj)
            setattr(out[head], attr, value)
    else:
        x = np.asarray(out[head], dtype=np.float64) + dx
        out[head] = float(x) if x.ndim == 0 else x
    return out


def bump_greek(
    pricer: Callable[..., Any],
    wrt: str | Sequence[str],
    *args: Any,
    h: float | Mapping[str, float] = 1e-4,
    relative: bool = False,
    **kwargs: Any,
) -> np.ndarray | float:
    """Sensitivity of any pricer to any of its inputs, by central finite differences.

    ``pricer(*args, **kwargs)`` is re-evaluated with inputs bumped up and
    down. `wrt` names a parameter of the pricer (``"S"``, ``"sigma"``,
    ``"H"``, ...) or an attribute of a model argument (``"model.v0"``,
    bumped on a copy, so the caller's model is untouched). One name gives
    :math:`(V_+ - V_-)/(2h)`; a pair gives a second derivative,
    :math:`(V_+ - 2V_0 + V_-)/h^2` when both names are the same (gamma,
    volga) and :math:`(V_{++} - V_{+-} - V_{-+} + V_{--})/(4 h_1 h_2)` when
    they differ (vanna). The bias is :math:`O(h^2)` (Glasserman 2004,
    Sec. 7.1). Theta is minus the derivative in ``"T"``.

    A pricer may return a number, an array (over a strike vector, say) or
    a result with an ``estimate`` attribute, such as
    :class:`~financialkit.monte_carlo.MonteCarloResult`. If it takes an
    ``rng`` argument, every evaluation gets the same integer seed (drawn
    once from a generator or ``None``, used as given if already an
    integer), so the bumped prices share their random numbers and the
    difference has variance :math:`O(1)` rather than :math:`O(h^{-2})`
    (Glasserman 2004, Sec. 7.1.2). Only the difference of the estimates is
    returned, not its standard error. Under common random numbers the
    estimate is only as smooth in the input as the simulated payoff, so
    give a payoff with a jump (a discretely monitored barrier, a digital) a
    larger bump.

    Parameters
    ----------
    pricer : callable
        Any function returning a price.
    wrt : str or pair of str
        Input to differentiate in, or two inputs for a second derivative.
    *args, **kwargs
        Arguments of ``pricer``, positional or by name.
    h : float or mapping of str to float
        Bump size, or a bump per name in `wrt`.
    relative : bool
        Read `h` as a fraction of the input's value rather than an
        absolute bump.

    Returns
    -------
    float or ndarray
        The derivative, shaped like the price.

    Raises
    ------
    ValueError
        If `wrt` is not one or two names or names no parameter of
        `pricer`, or a bump is zero.

    Examples
    --------
    Black-Scholes delta and gamma of Hull's (10th ed.) Example 19.1 call:

    >>> from financialkit.options import black_scholes_price
    >>> round(bump_greek(black_scholes_price, "S", 49.0, 50.0, 0.3846, 0.05, 0.20), 3)
    0.522
    >>> round(bump_greek(black_scholes_price, ("S", "S"), 49.0, 50.0, 0.3846, 0.05, 0.20, h=1e-2), 3)
    0.066
    """
    names = (wrt,) if isinstance(wrt, str) else tuple(wrt)
    if len(names) not in (1, 2):
        raise ValueError(f"wrt must be one name or two, got {wrt!r}")
    signature = inspect.signature(pricer)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    base = dict(bound.arguments)
    for name in names:
        if name.partition(".")[0] not in base:
            raise ValueError(f"{getattr(pricer, '__name__', 'pricer')} has no parameter {name.partition('.')[0]!r}")
    if "rng" in base and not isinstance(base["rng"], (int, np.integer)):
        base["rng"] = int(np.random.default_rng(base["rng"]).integers(2**63))

    def step(name: str) -> float:
        hx = float(h[name] if isinstance(h, Mapping) else h)
        if relative:
            head, _, attr = name.partition(".")
            hx *= abs(float(getattr(base[head], attr) if attr else base[head]))
        if hx == 0:
            raise ValueError(f"the bump of {name!r} is zero")
        return hx

    def V(*bumps: tuple[str, float]) -> np.ndarray:
        arguments = base
        for name, dx in bumps:
            arguments = _bump(arguments, name, dx)
        ba = inspect.BoundArguments(signature, arguments)
        out = pricer(*ba.args, **ba.kwargs)
        return np.asarray(getattr(out, "estimate", out), dtype=np.float64)

    a, b = names[0], names[-1]
    ha = step(a)
    if len(names) == 1:
        return as_output((V((a, ha)) - V((a, -ha))) / (2 * ha))
    if a == b:
        return as_output((V((a, ha)) - 2 * V() + V((a, -ha))) / ha**2)
    hb = step(b)
    return as_output((V((a, ha), (b, hb)) - V((a, ha), (b, -hb)) - V((a, -ha), (b, hb)) + V((a, -ha), (b, -hb))) / (4 * ha * hb))

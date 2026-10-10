"""Annuities, perpetuities, life annuities and sinking funds.

An annuity-certain of 1 per period for ``n`` periods at effective rate
``i`` is worth :math:`a_{\\overline{n}|} = (1 - v^n)/i`, :math:`v = 1/(1+i)`,
and accumulates to :math:`s_{\\overline{n}|} = ((1+i)^n - 1)/i`
(Kellison, *The Theory of Interest*, 3rd ed., 2009, Ch. 3). Letting
:math:`n \\to \\infty` gives the perpetuity :math:`1/i`, the price of the
British consols created in 1751 (Homer and Sylla, *A History of Interest
Rates*, 4th ed., 2005, Ch. 10). Weighting each payment by the probability
that the annuitant is alive gives Johan de Witt's life annuity
(*Waerdye van Lyf-renten*, 1671; Hald, *A History of Probability and
Statistics*, 1990, Sec. 9.2). Richard Price's sinking fund (*An Appeal to
the Public on the Subject of the National Debt*, 1772) is the
accumulation side: the level deposit :math:`F / s_{\\overline{n}|}` that
grows to repay a debt ``F``.

``rate`` is the effective rate per payment period throughout.
"""

from __future__ import annotations

import numpy as np

from financialkit.time_value.core.base import SinkingFundSchedule, _finalize

__all__ = [
    "annuity_factor",
    "annuity_present_value",
    "annuity_future_value",
    "annuity_payment",
    "perpetuity_present_value",
    "de_witt_survival",
    "life_annuity_value",
    "sinking_fund_payment",
    "sinking_fund_schedule",
]


def annuity_factor(rate, n, due: bool = False):
    """Present value of 1 per period for `n` periods, :math:`a_{\\overline{n}|}`.

    Parameters
    ----------
    rate : float or array-like
        Effective rate per period; zero is allowed (the factor is then ``n``).
    n : int or array-like
        Number of payments.
    due : bool
        ``True`` for payments at the start of each period (annuity-due).

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(annuity_factor(0.05, 10), 6)
    7.721735
    """
    i = np.asarray(rate, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(i == 0.0, n, -np.expm1(-n * np.log1p(i)) / i)
    return _finalize(a * (1.0 + i) if due else a)


def annuity_present_value(payment, rate, n, due: bool = False):
    """Present value of `n` level payments.

    Parameters
    ----------
    payment : float or array-like
    rate : float or array-like
        Effective rate per period.
    n : int or array-like
    due : bool
        Payments at the start of each period.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(annuity_present_value(1000.0, 0.06, 5), 2)
    4212.36
    """
    return _finalize(np.asarray(payment, dtype=np.float64) * annuity_factor(rate, n, due))


def annuity_future_value(payment, rate, n, due: bool = False):
    """Accumulated value after `n` level payments, :math:`P s_{\\overline{n}|}`.

    Parameters
    ----------
    payment : float or array-like
    rate : float or array-like
        Effective rate per period.
    n : int or array-like
    due : bool

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(annuity_future_value(100.0, 0.05, 10), 2)
    1257.79
    """
    i = np.asarray(rate, dtype=np.float64)
    return _finalize(np.asarray(annuity_present_value(payment, i, n, due)) * (1.0 + i) ** np.asarray(n, dtype=np.float64))


def annuity_payment(present_value, rate, n, due: bool = False):
    """Level payment whose `n`-period annuity is worth `present_value`.

    Parameters
    ----------
    present_value : float or array-like
    rate : float or array-like
        Effective rate per period.
    n : int or array-like
    due : bool

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(annuity_payment(10_000.0, 0.01, 12), 2)
    888.49
    """
    return _finalize(np.asarray(present_value, dtype=np.float64) / np.asarray(annuity_factor(rate, n, due)))


def perpetuity_present_value(payment, rate, growth=0.0, due: bool = False):
    """Value of a payment that continues forever, :math:`P / (i - g)`.

    Parameters
    ----------
    payment : float or array-like
        First payment, made one period from now (or now if `due`).
    rate : float or array-like
        Effective rate per period, greater than `growth`.
    growth : float or array-like
        Growth rate of the payments per period.
    due : bool

    Returns
    -------
    float or ndarray

    Examples
    --------
    A 3% consol paying 3 a year, when long yields are 4%:

    >>> round(perpetuity_present_value(3.0, 0.04), 6)
    75.0
    """
    i = np.asarray(rate, dtype=np.float64)
    g = np.asarray(growth, dtype=np.float64)
    if np.any(i <= g):
        raise ValueError("a perpetuity needs rate > growth")
    pv = np.asarray(payment, dtype=np.float64) / (i - g)
    return _finalize(pv * (1.0 + i) if due else pv)


# De Witt's hypothesis: relative chances of death in the age bands
# [3, 53), [53, 63), [63, 73), [73, 80); nobody survives past 80.
_DE_WITT_BANDS = np.array([3.0, 53.0, 63.0, 73.0, 80.0])
_DE_WITT_WEIGHTS = np.array([1.0, 1.5, 2.0, 3.0])


def de_witt_survival(age: float, times, base_hazard: float = 0.01) -> np.ndarray:
    """Survival probabilities under De Witt's (1671) mortality hypothesis.

    De Witt assumed that between ages 3 and 53 the chance of dying is the
    same in every half-year, and that it is 1.5, 2 and 3 times as large in
    the bands 53-63, 63-73 and 73-80, with no one living beyond 80. Here
    this is a piecewise-constant force of mortality ``base_hazard * w``.

    Parameters
    ----------
    age : float
        Age of the annuitant today, at least 3.
    times : array-like
        Years from today.
    base_hazard : float
        Annual force of mortality in the youngest band.

    Returns
    -------
    ndarray
        Probability of being alive at each time.

    Examples
    --------
    >>> de_witt_survival(70.0, [0.0, 5.0, 9.0, 10.0]).round(4).tolist()
    [1.0, 0.8869, 0.7866, 0.0]
    """
    if age < _DE_WITT_BANDS[0]:
        raise ValueError("De Witt's hypothesis starts at age 3")
    x = age + np.asarray(times, dtype=np.float64)

    def cumulative_hazard(a):
        lo, hi = _DE_WITT_BANDS[:-1], _DE_WITT_BANDS[1:]
        exposure = np.clip(np.asarray(a)[..., None], lo, hi) - lo
        return base_hazard * (exposure * _DE_WITT_WEIGHTS).sum(axis=-1)

    s = np.exp(-(cumulative_hazard(x) - cumulative_hazard(age)))
    return np.where(x >= _DE_WITT_BANDS[-1], 0.0, s)


def life_annuity_value(payment, rate, survival, due: bool = False):
    """Expected present value of a life annuity, :math:`P \\sum_k {}_k p_x v^k`.

    Parameters
    ----------
    payment : float
        Payment per period while the annuitant lives.
    rate : float or array-like
        Effective rate per period.
    survival : array-like, shape (n,)
        ``survival[k - 1]`` is the probability of being alive at period
        ``k`` (or at ``k - 1`` if `due`), e.g. from :func:`de_witt_survival`.
    due : bool

    Returns
    -------
    float or ndarray

    Examples
    --------
    With certain survival the life annuity is an annuity-certain:

    >>> round(life_annuity_value(1.0, 0.04, [1.0, 1.0, 1.0]), 6) == round(annuity_factor(0.04, 3), 6)
    True
    """
    p = np.asarray(survival, dtype=np.float64)
    k = np.arange(p.size) + (0.0 if due else 1.0)
    i = np.asarray(rate, dtype=np.float64)
    v = (1.0 + i[..., None]) ** -k
    return _finalize(payment * (p * v).sum(axis=-1))


def sinking_fund_payment(target, rate, n):
    """Level deposit that accumulates to `target` after `n` periods (Price, 1772).

    Parameters
    ----------
    target : float or array-like
        Amount to accumulate, e.g. the face value of a debt.
    rate : float or array-like
        Effective rate earned by the fund per period.
    n : int or array-like

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(sinking_fund_payment(1000.0, 0.05, 10), 4)
    79.5046
    """
    return _finalize(np.asarray(target, dtype=np.float64) / np.asarray(annuity_future_value(1.0, rate, n)))


def sinking_fund_schedule(target: float, rate: float, n: int) -> SinkingFundSchedule:
    """Period-by-period growth of a sinking fund.

    Parameters
    ----------
    target : float
    rate : float
        Effective rate per period.
    n : int

    Returns
    -------
    SinkingFundSchedule

    Examples
    --------
    >>> round(float(sinking_fund_schedule(1000.0, 0.05, 10).balance[-1]), 6)
    1000.0
    """
    d = float(sinking_fund_payment(target, rate, n))
    balance = d * np.asarray(annuity_future_value(1.0, rate, np.arange(1, n + 1)))
    previous = np.concatenate(([0.0], balance[:-1]))
    return SinkingFundSchedule(periods=np.arange(1, n + 1), deposit=d, interest=previous * rate, balance=balance)

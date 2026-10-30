"""Barrier options: knock-outs and knock-ins, single and double, continuous and discrete.

* :func:`merton_down_and_out_call` -- Merton (1973): a down-and-out call
  is the call minus the reflected call, by the method of images.
* :func:`barrier_price` -- Reiner and Rubinstein (1991): closed forms for
  all eight single-barrier calls and puts, with a cash rebate.
* :func:`double_barrier_price` -- Ikeda and Kunitomo (1992): knock-out
  and knock-in calls and puts between two flat barriers, as a rapidly
  converging series of images.
* :func:`discrete_barrier_price` -- Broadie, Glasserman and Kou (1997):
  a barrier monitored at :math:`m` dates is priced by the continuous
  formula with the barrier shifted away from the spot by
  :math:`e^{\\beta\\sigma\\sqrt{T/m}}`, :math:`\\beta = -\\zeta(1/2)/\\sqrt{2\\pi}`.
* :func:`barrier_option_mc` -- Monte Carlo with discrete monitoring, or
  with the Brownian-bridge crossing probability for unbiased continuous
  monitoring.

References: R. C. Merton, "Theory of Rational Option Pricing," *Bell J.
Econ. Manag. Sci.* 4 (1973) 141-183, Section 8; E. Reiner and M.
Rubinstein, "Breaking Down the Barriers," *Risk* 4(8) (1991) 28-35; N.
Ikeda and M. Kunitomo, "Pricing Options with Curved Boundaries,"
*Mathematical Finance* 2 (1992) 275-298; M. Broadie, P. Glasserman and
S. Kou, "A Continuity Correction for Discrete Barrier Options,"
*Mathematical Finance* 7 (1997) 325-349; E. G. Haug, *The Complete Guide
to Option Pricing Formulas*, 2nd ed. (McGraw-Hill, 2007), Section 4.17.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from financialkit._validation import nonnegative, one_of, positive
from financialkit._validation import option_type as check_option_type
from financialkit.exotic_options.utils.numerics import barrier_crossing_probability, deterministic_hit_time, parse_barrier_type, spot_paths
from financialkit.integrators import RNGLike
from financialkit.monte_carlo import MonteCarloResult, sample_mean, vanilla_payoff
from financialkit.options import black_scholes_price
from financialkit.options.utils.common import as_output

__all__ = [
    "BGK_BETA",
    "merton_down_and_out_call",
    "barrier_price",
    "double_barrier_price",
    "discrete_barrier_price",
    "barrier_option_mc",
]

BGK_BETA = 0.5825971579390106
"""float: Broadie-Glasserman-Kou's constant :math:`\\beta = -\\zeta(1/2)/\\sqrt{2\\pi}`."""


def _arrays(*args: ArrayLike) -> list[np.ndarray]:
    return list(np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in args)))


def _check(S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, **barriers: np.ndarray) -> None:
    nonnegative(S=S, K=K, T=T, sigma=sigma)
    positive(**barriers)


def _deterministic_barrier(S, K, H, T, r, q, phi: int, is_down: bool, is_out: bool, R) -> np.ndarray:
    """Single-barrier price on the riskless path :math:`S e^{bt}` (``sigma = 0``, ``T = 0`` or ``S = 0``), spot live."""
    hit, t_hit = deterministic_hit_time(S, H, T, r - q, is_down)
    vanilla = np.exp(-r * T) * np.maximum(phi * (S * np.exp((r - q) * T) - K), 0.0)
    if is_out:
        return np.where(hit, R * np.exp(-r * t_hit), vanilla)
    return np.where(hit, vanilla, R * np.exp(-r * T))


def merton_down_and_out_call(S: ArrayLike, K: ArrayLike, H: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> np.ndarray | float:
    """Merton's (1973) down-and-out call with the barrier at or below the strike.

    The barrier is an absorbing boundary for the log-price, and the
    method of images subtracts a call started from the reflected spot
    :math:`H^2 / S`:

    .. math:: C_{do} = c(S, K) - \\left(\\frac{H}{S}\\right)^{2b/\\sigma^2 - 1} c\\!\\left(\\frac{H^2}{S}, K\\right),
              \\qquad b = r - q,

    with :math:`c` the Black-Scholes-Merton call. It is zero once
    :math:`S \\le H`. With ``sigma = 0`` or ``T = 0`` it is the payoff of
    the riskless path :math:`S e^{bt}`.

    Parameters
    ----------
    S, K, H, T, r, sigma : array_like
        Spot, strike, barrier (``H <= K``), maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.

    Returns
    -------
    float or ndarray

    Examples
    --------
    A barrier far below the spot hardly matters:

    >>> from financialkit.options import black_scholes_price
    >>> bool(abs(merton_down_and_out_call(100.0, 100.0, 50.0, 1.0, 0.05, 0.2) - black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)) < 1e-6)
    True
    """
    S, K, H, T, r, sigma, q = _arrays(S, K, H, T, r, sigma, q)
    _check(S, K, T, sigma, H=H)
    if np.any(H > K):
        raise ValueError("merton_down_and_out_call needs H <= K; use barrier_price for H > K")
    alive = S > H
    regular = alive & (sigma * np.sqrt(T) > 0)
    Ss, sig = np.where(regular, S, H), np.where(regular, sigma, 1.0)
    image = (H / Ss) ** (2.0 * (r - q) / sig**2 - 1.0) * black_scholes_price(H**2 / Ss, K, T, r, sig, q)
    live = black_scholes_price(Ss, K, T, r, sig, q) - image
    deterministic = _deterministic_barrier(S, K, H, T, r, q, 1, True, True, 0.0)
    return as_output(np.where(alive, np.where(regular, live, deterministic), 0.0))


def barrier_price(
    S: ArrayLike,
    K: ArrayLike,
    H: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    rebate: ArrayLike = 0.0,
) -> np.ndarray | float:
    """Reiner-Rubinstein (1991) price of a continuously monitored single-barrier option.

    With :math:`b = r - q`, :math:`\\mu = (b - \\sigma^2/2)/\\sigma^2` and
    :math:`\\lambda = \\sqrt{\\mu^2 + 2r/\\sigma^2}`, every one of the eight
    calls and puts is a sum of the building blocks :math:`A, \\dots, F` of
    Haug (2007, Sec. 4.17.1): :math:`A` is the vanilla option, :math:`B`
    the vanilla struck at the barrier, :math:`C` and :math:`D` their
    reflections through the barrier, :math:`E` a knock-in's rebate paid at
    expiry if the barrier is never hit and :math:`F` a knock-out's rebate
    paid when it is hit. For example, for :math:`K > H`,

    .. math:: C_{do} = A - C + F, \\qquad C_{di} = C + E.

    In-out parity holds when the rebate is zero:
    ``knock-in + knock-out = vanilla``. If the spot is already through
    the barrier, a knock-out is worth its rebate and a knock-in the
    vanilla option. With ``sigma = 0``, at ``T = 0`` or at ``S = 0`` the
    path :math:`S e^{bt}` is known: a knock-out pays the vanilla payoff, or
    its rebate on the date the path reaches the barrier, and a knock-in the
    vanilla payoff or its rebate at expiry.

    Parameters
    ----------
    S, K, H, T, r, sigma : array_like
        Spot, strike, barrier, maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    barrier_type : {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}
    rebate : array_like
        Cash paid when a knock-out is knocked out, or at expiry when a
        knock-in never knocks in.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Table 4-13: S = 100, H = 95, rebate 3, T = 0.5, r = 8%,
    b = 4%, sigma = 25%:

    >>> round(barrier_price(100.0, 90.0, 95.0, 0.5, 0.08, 0.25, q=0.04, barrier_type="down-and-out", rebate=3.0), 4)
    9.0246
    """
    phi = check_option_type(option_type)
    is_down, is_out = parse_barrier_type(barrier_type)
    eta = 1.0 if is_down else -1.0
    S, K, H, T, r, sigma, q, R = _arrays(S, K, H, T, r, sigma, q, rebate)
    _check(S, K, T, sigma, H=H)
    knocked = S <= H if is_down else S >= H
    regular = ~knocked & (sigma * np.sqrt(T) > 0) & (S > 0)
    deterministic = _deterministic_barrier(S, K, H, T, r, q, phi, is_down, is_out, R)
    dead = R if is_out else np.asarray(black_scholes_price(S, K, T, r, sigma, q, option_type))
    # Any live spot, volatility and maturity where the closed form does not apply; overwritten below.
    Ss = np.where(regular, S, H * (1.0 + 1e-9 * eta))
    sigma, T = np.where(regular, sigma, 1.0), np.where(regular, T, 1.0)
    b = r - q
    v = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    lam = np.sqrt(mu**2 + 2.0 * r / sigma**2)
    with np.errstate(divide="ignore"):  # K = 0 sends x1 and y1 to +inf, which the formula handles
        x1 = np.log(Ss / K) / v + (1.0 + mu) * v
        y1 = np.log(H**2 / (Ss * K)) / v + (1.0 + mu) * v
    x2 = np.log(Ss / H) / v + (1.0 + mu) * v
    y2 = np.log(H / Ss) / v + (1.0 + mu) * v
    z = np.log(H / Ss) / v + lam * v
    fS, fK, hs = Ss * np.exp((b - r) * T), K * np.exp(-r * T), H / Ss
    N = norm.cdf
    A = phi * fS * N(phi * x1) - phi * fK * N(phi * x1 - phi * v)
    B = phi * fS * N(phi * x2) - phi * fK * N(phi * x2 - phi * v)
    C = phi * fS * hs ** (2 * (mu + 1)) * N(eta * y1) - phi * fK * hs ** (2 * mu) * N(eta * y1 - eta * v)
    D = phi * fS * hs ** (2 * (mu + 1)) * N(eta * y2) - phi * fK * hs ** (2 * mu) * N(eta * y2 - eta * v)
    E = R * np.exp(-r * T) * (N(eta * x2 - eta * v) - hs ** (2 * mu) * N(eta * y2 - eta * v))
    F = R * (hs ** (mu + lam) * N(eta * z) + hs ** (mu - lam) * N(eta * z - 2 * eta * lam * v))
    high = K > H
    # (strike above barrier, strike at or below barrier) for each of the eight options
    table = {
        (1, True, False): (C + E, A - B + D + E),
        (1, False, False): (A + E, B - C + D + E),
        (-1, True, False): (B - C + D + E, A + E),
        (-1, False, False): (A - B + D + E, C + E),
        (1, True, True): (A - C + F, B - D + F),
        (1, False, True): (F, A - B + C - D + F),
        (-1, True, True): (A - B + C - D + F, F),
        (-1, False, True): (B - D + F, A - C + F),
    }
    above, below = table[(phi, is_down, is_out)]
    live = np.where(high, above, below)
    return as_output(np.where(knocked, dead, np.where(regular, live, deterministic)))


def double_barrier_price(
    S: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    U: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    knock: str = "out",
    n_terms: int = 5,
) -> np.ndarray | float:
    """Ikeda-Kunitomo (1992) double-barrier option with flat barriers :math:`L < S < U`.

    The knock-out call is a series of images reflected alternately
    through both barriers (Haug 2007, Sec. 4.17.3, with zero curvature):

    .. math:: c = S e^{(b - r)T} \\sum_{n=-N}^{N} \\left[\\left(\\tfrac{U^n}{L^n}\\right)^{\\mu}
              (N(d_1) - N(d_2)) - \\left(\\tfrac{L^{n+1}}{U^n S}\\right)^{\\mu} (N(d_3) - N(d_4))\\right]
              - K e^{-rT} \\sum_{n=-N}^{N} [\\dots]_{\\mu - 2,\\; d_i - \\sigma\\sqrt T},

    with :math:`\\mu = 2b/\\sigma^2 + 1`. The terms decay like
    :math:`e^{-2n^2 \\ln^2(U/L)/\\sigma^2 T}`, so a handful suffice. The
    knock-in is the vanilla option minus the knock-out. With ``sigma = 0``
    or ``T = 0`` the knock-out pays the vanilla payoff of the path
    :math:`S e^{bt}` if that path stays between the barriers.

    Parameters
    ----------
    S, K, L, U, T, r, sigma : array_like
        Spot, strike, lower and upper barriers, maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    knock : {"out", "in"}
    n_terms : int
        The series runs over :math:`n = -N, \\dots, N`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Far-away barriers leave the vanilla option:

    >>> from financialkit.options import black_scholes_price
    >>> bool(abs(double_barrier_price(100.0, 100.0, 20.0, 500.0, 0.5, 0.05, 0.2) - black_scholes_price(100.0, 100.0, 0.5, 0.05, 0.2)) < 1e-8)
    True
    """
    one_of("knock", knock, ("out", "in"))
    phi = check_option_type(option_type)
    S, K, L, U, T, r, sigma, q = _arrays(S, K, L, U, T, r, sigma, q)
    _check(S, K, T, sigma, L=L)
    if np.any(L >= U):
        raise ValueError("need L < U")
    alive = (S > L) & (S < U)
    vanilla = np.asarray(black_scholes_price(S, K, T, r, sigma, q, option_type))
    fwd = S * np.exp((r - q) * T)
    deterministic = np.where((fwd > L) & (fwd < U), np.exp(-r * T) * np.maximum(phi * (fwd - K), 0.0), 0.0)
    regular = sigma * np.sqrt(T) > 0
    Ss = np.where(alive, S, np.sqrt(L * U))
    sigma, T = np.where(regular, sigma, 1.0), np.where(regular, T, 1.0)
    b = r - q
    v = sigma * np.sqrt(T)
    mu = 2.0 * b / sigma**2 + 1.0
    n = np.arange(-int(n_terms), int(n_terms) + 1).reshape((-1,) + (1,) * S.ndim)
    drift = (b + 0.5 * sigma**2) * T
    lo, hi = (K, U) if phi == 1 else (L, K)  # the payoff is non-zero for lo < S_T < hi
    lo, hi = np.maximum(lo, L), np.minimum(hi, U)
    nonempty = lo < hi
    hi = np.where(nonempty, hi, 2.0 * lo)  # an empty payoff range (a put struck at or below L) is worth 0

    def d(num):
        return (np.log(num) + drift) / v

    UL = (U / L) ** n
    refl = L ** (n + 1) / (U**n * Ss)
    d1, d2 = d(Ss * UL**2 / lo), d(Ss * UL**2 / hi)
    d3, d4 = d(L ** (2 * n + 2) / (lo * Ss * U ** (2 * n))), d(L ** (2 * n + 2) / (hi * Ss * U ** (2 * n)))
    N = norm.cdf
    asset = np.sum(UL**mu * (N(d1) - N(d2)) - refl**mu * (N(d3) - N(d4)), axis=0)
    cash = np.sum(UL ** (mu - 2) * (N(d1 - v) - N(d2 - v)) - refl ** (mu - 2) * (N(d3 - v) - N(d4 - v)), axis=0)
    out = phi * (Ss * np.exp((b - r) * T) * asset - K * np.exp(-r * T) * cash)
    out = np.where(alive & nonempty, np.where(regular, np.maximum(out, 0.0), deterministic), 0.0)
    if knock == "in":
        out = vanilla - out
    return as_output(out)


def discrete_barrier_price(
    S: ArrayLike,
    K: ArrayLike,
    H: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    n_monitor: int = 252,
    rebate: ArrayLike = 0.0,
) -> np.ndarray | float:
    """Broadie-Glasserman-Kou (1997) price of a barrier monitored at `n_monitor` equally spaced dates.

    A path can cross a discretely monitored barrier between two dates
    without being caught, so the option behaves like a continuously
    monitored one with the barrier moved away from the spot:

    .. math:: H \\to H e^{\\pm \\beta \\sigma \\sqrt{\\Delta t}}, \\qquad
              \\beta = -\\zeta(1/2)/\\sqrt{2\\pi} \\approx 0.5826,

    with ``+`` for an up barrier and ``-`` for a down barrier. The error
    is :math:`o(1/\\sqrt{m})`, against :math:`O(1/\\sqrt{m})` for ignoring
    the difference.

    Parameters
    ----------
    S, K, H, T, r, sigma : array_like
        Spot, strike, barrier, maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    barrier_type : {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}
    n_monitor : int
        Number of monitoring dates :math:`m`, the last at expiry.
    rebate : array_like
        Cash rebate, as in :func:`barrier_price`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Daily monitoring is worth more to the holder of a knock-out than
    continuous monitoring:

    >>> args = (100.0, 100.0, 85.0, 1.0, 0.05, 0.3)
    >>> bool(discrete_barrier_price(*args, n_monitor=252) > barrier_price(*args))
    True
    """
    is_down, _ = parse_barrier_type(barrier_type)
    H, T, sigma = (np.asarray(a, dtype=np.float64) for a in (H, T, sigma))
    nonnegative(T=T, sigma=sigma)
    shift = np.exp((-1.0 if is_down else 1.0) * BGK_BETA * sigma * np.sqrt(T / int(n_monitor)))
    return barrier_price(S, K, H * shift, T, r, sigma, q, option_type, barrier_type, rebate)


def barrier_option_mc(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float | None = None,
    q: float = 0.0,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    n_steps: int = 50,
    n_paths: int = 100_000,
    continuous: bool = True,
    rng: RNGLike = None,
    confidence: float = 0.95,
    paths: ArrayLike | None = None,
    discount: ArrayLike | None = None,
) -> MonteCarloResult:
    """Monte Carlo price of a single-barrier option without rebate.

    Paths come from :func:`financialkit.integrators.gbm_paths`. With
    ``continuous=False`` the barrier is checked only at the `n_steps`
    grid dates, which prices a discretely monitored option exactly. With
    ``continuous=True`` each step's Brownian-bridge crossing probability
    :math:`p_i` (:func:`~financialkit.exotic_options.utils.numerics.barrier_crossing_probability`)
    weights the payoff by the survival probability :math:`\\prod_i (1 - p_i)`,
    which prices continuous monitoring without discretisation bias
    (Glasserman 2004, Sec. 6.4). Any other model's spot paths can be given
    as `paths`; the bridge then assumes volatility `sigma` within each step.

    Parameters
    ----------
    S, K, H, T, r : float
        Spot, strike, barrier, maturity and rate.
    sigma : float, optional
        Volatility of the GBM paths and of the bridge; required unless
        `paths` are given with ``continuous=False``.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    barrier_type : {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}
    n_steps : int
        Time steps (monitoring dates when ``continuous=False``); ignored with `paths`.
    n_paths : int
        Number of paths; ignored with `paths`.
    continuous : bool
        Continuous monitoring through the bridge correction, or discrete.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.
    paths : array_like, shape (n_paths, n_steps + 1), optional
        Risk-neutral spot paths starting at `S` on the grid :math:`t_i = iT/n`,
        in place of the GBM ones.
    discount : array_like, optional
        Discount factors :math:`D(0, t_i)` for `paths`, of shape
        ``(n_steps + 1,)`` or one row per path; only :math:`D(0, T)` is
        used. :math:`e^{-rT}` by default.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> res = barrier_option_mc(100.0, 100.0, 90.0, 1.0, 0.05, 0.2, n_steps=20, n_paths=40_000, rng=0)
    >>> bool(abs(res.estimate - barrier_price(100.0, 100.0, 90.0, 1.0, 0.05, 0.2)) < 4 * res.std_error)
    True
    """
    is_down, is_out = parse_barrier_type(barrier_type)
    spot, D = spot_paths(S, T, r - q, sigma, int(n_steps), int(n_paths), paths, discount, rng=rng)
    dt = T / (spot.shape[1] - 1)
    if continuous:
        if sigma is None:
            raise ValueError("continuous monitoring needs sigma for the bridge")
        p = barrier_crossing_probability(np.log(spot[:, :-1]), np.log(spot[:, 1:]), float(np.log(H)), sigma, dt)
        survive = np.prod(1.0 - p, axis=1)
    else:
        monitored = spot[:, 1:]
        survive = np.all(monitored > H if is_down else monitored < H, axis=1).astype(np.float64)
    weight = survive if is_out else 1.0 - survive
    disc = np.exp(-r * T) if D is None else D[:, -1]
    payoff = disc * vanilla_payoff(spot[:, -1], K, option_type) * weight
    return sample_mean(payoff, confidence, "continuous barrier" if continuous else "discrete barrier")

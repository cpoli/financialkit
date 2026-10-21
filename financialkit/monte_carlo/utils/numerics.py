"""Supporting numerics for financialkit.monte_carlo: Black-Scholes terminal
prices from given normals, vanilla payoffs, the sample-mean estimator and
SDE callbacks for :mod:`financialkit.integrators`.

Terminal prices go through :func:`financialkit.integrators.gbm_paths` with
a one-step ``dW``, so every estimator here can be fed pseudo-random,
antithetic, stratified, importance-sampled or quasi-random normals and
still simulate through the shared integrators.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import qmc

from financialkit._jit import njit
from financialkit.integrators import gbm_paths
from financialkit.monte_carlo.core.base import MonteCarloResult

__all__ = [
    "sample_mean",
    "terminal_prices",
    "vanilla_payoff",
    "gbm_drift",
    "gbm_diffusion",
    "qmc_engine",
]


def sample_mean(samples: ArrayLike, confidence: float = 0.95, method: str = "crude", n_evaluations: int | None = None) -> MonteCarloResult:
    """The sample mean of i.i.d. draws with its standard error :math:`s / \\sqrt{n}`.

    Parameters
    ----------
    samples : array_like, shape (n,)
        Independent draws of the quantity whose expectation is estimated.
    confidence : float
        Coverage of the confidence interval.
    method : str
        Label stored in the result.
    n_evaluations : int, optional
        Function evaluations behind the draws, if not one each (an
        antithetic pair or a replication costs several).

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> res = sample_mean([1.0, 2.0, 3.0, 4.0])
    >>> res.estimate, round(res.std_error, 4)
    (2.5, 0.6455)
    """
    y = np.asarray(samples, dtype=np.float64).ravel()
    if y.size < 2:
        raise ValueError("need at least two samples for a standard error")
    n_eval = y.size if n_evaluations is None else int(n_evaluations)
    return MonteCarloResult(float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size)), n_eval, confidence, method)


def terminal_prices(S: float, T: float, r: float, sigma: float, z: ArrayLike, q: float = 0.0) -> np.ndarray:
    """Risk-neutral Black-Scholes prices :math:`S_T = S e^{(r - q - \\sigma^2/2)T + \\sigma\\sqrt{T} z}`.

    Parameters
    ----------
    S, T, r, sigma : float
        Spot, maturity, risk-free rate and volatility.
    z : array_like, shape (n,)
        Standard normal draws, from any source.
    q : float
        Continuous dividend yield.

    Returns
    -------
    ndarray, shape (n,)
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    _, paths = gbm_paths(S, r - q, sigma, T, n_steps=1, n_paths=z.size, dW=np.sqrt(T) * z[:, None])
    return paths[:, -1]


def vanilla_payoff(S_T: ArrayLike, K: float, option_type: str = "call", digital: bool = False) -> np.ndarray:
    """Undiscounted payoff :math:`(S_T - K)^+`, :math:`(K - S_T)^+`, or the cash-or-nothing digital.

    Parameters
    ----------
    S_T : array_like
        Terminal prices.
    K : float
        Strike.
    option_type : {"call", "put"}
    digital : bool
        Pay 1 if in the money instead of the intrinsic value.

    Returns
    -------
    ndarray
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    S_T = np.asarray(S_T, dtype=np.float64)
    itm = S_T > K if option_type == "call" else S_T < K
    if digital:
        return itm.astype(np.float64)
    return np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)


@njit
def gbm_drift(x, t, p):
    """``(r - q) x`` with ``p = [r - q, sigma]``."""
    return p[0] * x


@njit
def gbm_diffusion(x, t, p):
    """``sigma x`` with ``p = [r - q, sigma]``."""
    return p[1] * x


def qmc_engine(kind: str, d: int, scramble: bool = True, rng=None):
    """A :mod:`scipy.stats.qmc` engine: ``"sobol"``, ``"halton"`` or ``"lhs"`` (Latin hypercube).

    Parameters
    ----------
    kind : {"sobol", "halton", "lhs"}
    d : int
        Dimension.
    scramble : bool
        Randomize the sequence (Owen scrambling for Sobol and Halton);
        ignored for ``"lhs"``, which is always random.
    rng : numpy.random.Generator or int, optional
        Seeds the randomization.

    Returns
    -------
    scipy.stats.qmc.QMCEngine
    """
    classes = {"sobol": qmc.Sobol, "halton": qmc.Halton, "lhs": qmc.LatinHypercube}
    if kind not in classes:
        raise ValueError(f"kind must be one of {sorted(classes)}, got {kind!r}")
    kwargs = {} if kind == "lhs" else {"scramble": scramble}
    gen = np.random.default_rng(rng)
    try:
        return classes[kind](d=int(d), rng=gen, **kwargs)
    except TypeError:  # pragma: no cover - scipy < 1.15 names the argument `seed`
        # Old Halton scrambles from a copy of a Generator `seed` without
        # advancing it, so engines sharing `gen` would repeat one scrambling;
        # an integer drawn from `gen` keeps them independent.
        return classes[kind](d=int(d), seed=int(gen.integers(2**63)), **kwargs)

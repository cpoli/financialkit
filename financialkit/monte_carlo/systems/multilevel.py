"""Multilevel Monte Carlo (Giles 2008).

Standard Monte Carlo with an Euler scheme of weak order 1 needs
:math:`O(\\varepsilon^{-2})` paths of :math:`O(\\varepsilon^{-1})` steps for
a root-mean-square error :math:`\\varepsilon`, a cost of
:math:`O(\\varepsilon^{-3})`. Writing the finest-level expectation as a
telescoping sum

.. math:: E[P_L] = E[P_0] + \\sum_{l=1}^{L} E[P_l - P_{l-1}],

and estimating each correction from paths on which the fine (``M**l``
steps) and coarse (``M**(l-1)`` steps) Euler solutions share their
Brownian increments, makes the corrections' variances fall like
:math:`M^{-l}`. Most samples are then taken on the cheap coarse levels,
and the total cost drops to :math:`O(\\varepsilon^{-2} (\\log \\varepsilon)^2)`.

References: M. B. Giles, "Multilevel Monte Carlo Path Simulation,"
*Operations Research* 56 (2008) 607-617; M. B. Giles, "Multilevel Monte
Carlo Methods," *Acta Numerica* 24 (2015) 259-328.
"""

from __future__ import annotations

import numpy as np

from financialkit.integrators import RNGLike, brownian_increments, euler_maruyama
from financialkit.monte_carlo.core.base import MLMCResult
from financialkit.monte_carlo.utils.numerics import gbm_diffusion, gbm_drift, vanilla_payoff

__all__ = ["mlmc_level", "multilevel_monte_carlo"]

#: Largest ``n_paths * n_steps`` simulated in one Euler call, to bound memory.
_CHUNK = 1_000_000


def _euler_terminal(S: float, T: float, params: np.ndarray, dW: np.ndarray) -> np.ndarray:
    _, paths = euler_maruyama(gbm_drift, gbm_diffusion, S, T, dW.shape[1], dW.shape[0], params=params, dW=dW)
    return paths[:, -1]


def mlmc_level(
    level: int,
    n_paths: int,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    M: int = 2,
    rng: RNGLike = None,
) -> np.ndarray:
    """Samples of the level-``l`` correction :math:`P_l - P_{l-1}` for a European option.

    :math:`P_l` is the discounted payoff of the Euler-Maruyama solution of
    risk-neutral GBM with ``M**level`` steps; the coarse solution sums the
    fine Brownian increments in groups of `M`, so the two are coupled.
    Level 0 returns :math:`P_0` itself (one Euler step).

    Parameters
    ----------
    level : int
        Level :math:`l \\ge 0`.
    n_paths : int
        Number of coupled samples.
    S, K, T, r, sigma : float
        Spot, strike, maturity, risk-free rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    M : int
        Refinement factor between levels.
    rng : numpy.random.Generator or int, optional
        Generator or seed.

    Returns
    -------
    ndarray, shape (n_paths,)

    Examples
    --------
    The corrections vary far less than the payoff itself:

    >>> y0 = mlmc_level(0, 20_000, 100.0, 100.0, 1.0, 0.05, 0.2, rng=0)
    >>> y4 = mlmc_level(4, 20_000, 100.0, 100.0, 1.0, 0.05, 0.2, rng=0)
    >>> bool(y4.var() < 0.01 * y0.var())
    True
    """
    gen = np.random.default_rng(rng)
    nf = M ** int(level)
    params = np.array([r - q, sigma])
    disc = np.exp(-r * T)
    out = np.empty(int(n_paths))
    step = max(1, _CHUNK // nf)
    for start in range(0, int(n_paths), step):
        m = min(step, int(n_paths) - start)
        dW = brownian_increments(T, nf, m, rng=gen)
        fine = disc * vanilla_payoff(_euler_terminal(S, T, params, dW), K, option_type)
        if level == 0:
            out[start : start + m] = fine
        else:
            coarse_dW = dW.reshape(m, nf // M, M).sum(axis=2)
            out[start : start + m] = fine - disc * vanilla_payoff(_euler_terminal(S, T, params, coarse_dW), K, option_type)
    return out


def multilevel_monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    eps: float = 0.05,
    M: int = 2,
    L_min: int = 2,
    L_max: int = 10,
    N0: int = 2000,
    rng: RNGLike = None,
    confidence: float = 0.95,
) -> MLMCResult:
    """European option by Giles's adaptive multilevel Monte Carlo, to root-mean-square error `eps`.

    The algorithm of Giles (2008), Sec. 5: start with levels
    ``0 .. L_min`` and `N0` samples each; set the optimal sample sizes

    .. math:: N_l = \\left\\lceil 2\\varepsilon^{-2} \\sqrt{V_l / C_l} \\sum_k \\sqrt{V_k C_k} \\right\\rceil,

    which split :math:`\\varepsilon^2` evenly between variance and squared
    bias; top up any level that is short; and once none is, add a level
    unless the bias estimate :math:`\\max(|\\hat Y_L|, |\\hat Y_{L-1}| / M) /
    (M - 1)` (weak order 1) is below :math:`\\varepsilon / \\sqrt 2`.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, risk-free rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    eps : float
        Target root-mean-square error.
    M : int
        Refinement factor.
    L_min, L_max : int
        Smallest and largest finest level.
    N0 : int
        Initial samples on a new level.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval (sampling error only).

    Returns
    -------
    MLMCResult
    """
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    gen = np.random.default_rng(rng)
    args = (S, K, T, r, sigma, q, option_type)
    L = int(L_min)
    n = np.zeros(L + 1, dtype=np.int64)
    s1 = np.zeros(L + 1)
    s2 = np.zeros(L + 1)
    dN = np.full(L + 1, int(N0), dtype=np.int64)
    while True:
        for l in np.flatnonzero(dN > 0):
            y = mlmc_level(int(l), int(dN[l]), *args, M=M, rng=gen)
            n[l] += dN[l]
            s1[l] += y.sum()
            s2[l] += y @ y
        mean = s1 / n
        var = np.maximum(s2 / n - mean**2, 0.0)
        C = np.array([1.0] + [M**l + M ** (l - 1) for l in range(1, L + 1)])
        target = np.ceil(2.0 / eps**2 * np.sqrt(var / C) * np.sum(np.sqrt(var * C))).astype(np.int64)
        dN = np.maximum(target - n, 0)
        if np.any(dN > 0):
            continue
        bias = max(abs(mean[L]), abs(mean[L - 1]) / M) / (M - 1)
        if bias < eps / np.sqrt(2.0) or L >= L_max:
            break
        L += 1
        n, s1, s2 = np.append(n, 0), np.append(s1, 0.0), np.append(s2, 0.0)
        dN = np.append(np.zeros(L, dtype=np.int64), int(N0))
    se = float(np.sqrt(np.sum(var / n)))
    return MLMCResult(
        float(mean.sum()),
        se,
        int(n.sum()),
        confidence,
        "multilevel",
        mean,
        var,
        n.copy(),
        C,
        float(n @ C),
        float(eps),
        int(M),
    )

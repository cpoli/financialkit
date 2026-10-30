"""Recombining lattices for European and American options: the
Cox-Ross-Rubinstein, Rendleman-Bartter and Leisen-Reimer binomial trees,
and Boyle's trinomial tree.

Every tree replaces geometric Brownian motion by a few-state random walk
whose risk-neutral mean matches :math:`e^{(r - q)\\Delta t}` and whose
variance matches :math:`\\sigma^2 \\Delta t` as :math:`\\Delta t \\to 0`, then
values the option by backward induction, taking the larger of the
continuation and exercise values at each node for an American option.
They differ only in how they place the nodes:

* CRR (1979): :math:`u = e^{\\sigma\\sqrt{\\Delta t}} = 1/d`, so the tree is
  centred on the spot. European prices converge like :math:`1/n`, with an
  oscillation from where the strike falls between nodes.
* Rendleman-Bartter (1979): the log-step is centred on the drift,
  :math:`\\ln u, \\ln d = (r - q - \\sigma^2/2)\\Delta t \\pm \\sigma\\sqrt{\\Delta
  t}`, which makes the up and down probabilities nearly equal.
* Leisen-Reimer (1996): the probabilities are Peizer-Pratt inversions of
  Black-Scholes's :math:`N(d_2)` and :math:`N(d_1)`, which centres the tree on
  the strike and gives smooth :math:`1/n^2` convergence (odd ``n`` only).
* Boyle (1986): three branches :math:`u, 1, d = 1/u` with
  :math:`u = e^{\\lambda\\sigma\\sqrt{\\Delta t}}` and probabilities matching
  the first two moments exactly; fewer steps for the same accuracy.

Backward induction is vectorised over the nodes of each time step, so
the trees run fast without Numba; there is no ``@njit`` kernel here.

References: J. C. Cox, S. A. Ross and M. Rubinstein, *J. Financial
Economics* 7 (1979) 229-263; R. J. Rendleman and B. J. Bartter,
*J. Finance* 34 (1979) 1093-1110; P. P. Boyle, *Int. Options J.* 3 (1986)
7-12; D. P. J. Leisen and M. Reimer, *Applied Mathematical Finance* 3
(1996) 319-346; Hull, *Options, Futures, and Other Derivatives*, 10th ed.,
Chs. 13 and 21.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from financialkit._validation import option_type as check_option_type
from financialkit._validation import positive
from financialkit.options.core.base import LatticePricer, TreeResult
from financialkit.options.utils.common import black_d1_d2

__all__ = ["BinomialTree", "CRRTree", "RendlemanBartterTree", "LeisenReimerTree", "TrinomialTree"]


def _backward_induction(
    log_S: np.ndarray, K: float, disc: float, probs: tuple[float, ...], log_steps: tuple[float, ...], sign: int, american: bool, n: int
) -> tuple[float, float, float, np.ndarray]:
    """Shared induction for binomial (2 branches) and trinomial (3 branches) lattices.

    ``log_S(i)`` nodes at step ``i`` are ``log_S0 + j * log_steps`` combinations;
    ``probs``/``log_steps`` are ordered from the lowest branch up. Returns the
    root value, delta, gamma and the exercise boundary.
    """
    b = len(probs) - 1  # nodes at step i: b * i + 1
    log_up, log_down = log_steps[-1], log_steps[0]

    def nodes(i: int) -> np.ndarray:
        if b == 1:
            j = np.arange(i + 1)
            return log_S[0] + j * log_up + (i - j) * log_down
        return log_S[0] + np.arange(-i, i + 1) * log_up

    boundary = np.full(n + 1, np.nan)
    S_T = np.exp(nodes(n))
    V = np.maximum(sign * (S_T - K), 0.0)
    if american:
        ex = (sign * (S_T - K)) > 0
        if np.any(ex):
            boundary[n] = S_T[ex].max() if sign < 0 else S_T[ex].min()
    level1 = level2 = V
    for i in range(n - 1, -1, -1):
        V = disc * np.sum([p * V[k : k + b * i + 1] for k, p in enumerate(probs)], axis=0)
        if american:
            S_i = np.exp(nodes(i))
            exercise = np.maximum(sign * (S_i - K), 0.0)
            ex = (exercise > V) & (exercise > 0)
            if np.any(ex):
                boundary[i] = S_i[ex].max() if sign < 0 else S_i[ex].min()
            V = np.maximum(V, exercise)
        if i == 2:
            level2 = V
        if i == 1:
            level1 = V
    S1 = np.exp(nodes(1)) if n >= 1 else np.array([np.exp(log_S[0])])
    delta = float((level1[-1] - level1[0]) / (S1[-1] - S1[0])) if n >= 1 else np.nan
    gamma = np.nan
    if n >= 2:
        S2 = np.exp(nodes(2))
        if b == 1:
            d_up = (level2[2] - level2[1]) / (S2[2] - S2[1])
            d_dn = (level2[1] - level2[0]) / (S2[1] - S2[0])
            gamma = float((d_up - d_dn) / (0.5 * (S2[2] - S2[0])))
        else:
            d_up = (level1[2] - level1[1]) / (S1[2] - S1[1])
            d_dn = (level1[1] - level1[0]) / (S1[1] - S1[0])
            gamma = float((d_up - d_dn) / (0.5 * (S1[2] - S1[0])))
    if not american:
        boundary = np.empty(0)
    return float(V[0]), delta, gamma, boundary


class BinomialTree(LatticePricer):
    """A recombining binomial tree; subclasses choose ``u``, ``d`` and ``p``."""

    @abstractmethod
    def _parameters(self, S: float, K: float, T: float, r: float, sigma: float, q: float, n: int) -> tuple[float, float, float, int]:
        """Return ``(u, d, p, n)`` for a tree of (about) ``n`` steps."""

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        american: bool = False,
    ) -> TreeResult:
        """Value a vanilla option by backward induction on the tree.

        Parameters
        ----------
        S, K, T, r, sigma : float
            Spot, strike, maturity (years), risk-free rate and volatility.
        q : float
            Continuous dividend yield.
        option_type : {"call", "put"}
        american : bool
            Allow early exercise at every node.

        Returns
        -------
        TreeResult
        """
        sign = check_option_type(option_type)
        positive(S=S, K=K, T=T, sigma=sigma)
        u, d, p, n = self._parameters(S, K, T, r, sigma, q, self.n_steps)
        if not 0.0 < p < 1.0:
            raise ValueError(f"risk-neutral probability {p:.4g} outside (0, 1); increase n_steps")
        dt = T / n
        price, delta, gamma, boundary = _backward_induction(np.array([np.log(S)]), K, np.exp(-r * dt), (1.0 - p, p), (np.log(d), np.log(u)), sign, american, n)
        return TreeResult(price=price, delta=delta, gamma=gamma, n_steps=n, method=self.method, times=np.linspace(0.0, T, n + 1), exercise_boundary=boundary)


class CRRTree(BinomialTree):
    """Cox-Ross-Rubinstein (1979) binomial tree, :math:`u = e^{\\sigma\\sqrt{\\Delta t}} = 1/d`.

    Parameters
    ----------
    n_steps : int

    Examples
    --------
    Hull, 10th ed., Example 21.1 (Figure 21.3): an American put with
    S = K = 50, r = 10%, sigma = 40% and five months, on a five-step tree.

    >>> round(CRRTree(5).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True).price, 2)
    4.49
    """

    method = "crr"

    def _parameters(self, S, K, T, r, sigma, q, n):
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        return u, d, (np.exp((r - q) * dt) - d) / (u - d), n


class RendlemanBartterTree(BinomialTree):
    """Rendleman-Bartter (1979) drift-centred binomial tree.

    :math:`\\ln u, \\ln d = (r - q - \\sigma^2/2)\\Delta t \\pm \\sigma\\sqrt{\\Delta t}`,
    with the no-arbitrage probability :math:`p = (e^{(r-q)\\Delta t} - d)/(u - d)`,
    which is :math:`1/2 + O(\\Delta t^{3/2})`.

    Parameters
    ----------
    n_steps : int

    Examples
    --------
    >>> res = RendlemanBartterTree(500).price(100.0, 100.0, 1.0, 0.05, 0.2)
    >>> round(res.price, 1)
    10.5
    """

    method = "rendleman_bartter"

    def _parameters(self, S, K, T, r, sigma, q, n):
        dt = T / n
        nu = (r - q - 0.5 * sigma**2) * dt
        u, d = np.exp(nu + sigma * np.sqrt(dt)), np.exp(nu - sigma * np.sqrt(dt))
        return u, d, (np.exp((r - q) * dt) - d) / (u - d), n


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt method 2 inversion of the normal CDF onto a binomial (``n`` odd)."""
    a = z / (n + 1.0 / 3.0 + 0.1 / (n + 1))
    return 0.5 + np.copysign(0.5, z) * np.sqrt(1.0 - np.exp(-(a**2) * (n + 1.0 / 6.0)))


class LeisenReimerTree(BinomialTree):
    """Leisen-Reimer (1996) binomial tree with Peizer-Pratt probabilities.

    :math:`p = h(d_2)`, :math:`p' = h(d_1)`, :math:`u = e^{(r-q)\\Delta t} p'/p`
    and :math:`d = (e^{(r-q)\\Delta t} - p u)/(1 - p)`, with :math:`h` the
    Peizer-Pratt inversion. An even ``n_steps`` is rounded up to the next
    odd number, as the method requires.

    Parameters
    ----------
    n_steps : int

    Examples
    --------
    >>> from financialkit.options import black_scholes_price
    >>> lr = LeisenReimerTree(51).price(100.0, 100.0, 1.0, 0.05, 0.2).price
    >>> bool(abs(lr - black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)) < 1e-3)
    True
    """

    method = "leisen_reimer"

    def _parameters(self, S, K, T, r, sigma, q, n):
        n = n if n % 2 else n + 1
        dt = T / n
        d1, d2 = (float(x) for x in black_d1_d2(S * np.exp((r - q) * T), K, sigma, T))
        p, p_bar = _peizer_pratt(d2, n), _peizer_pratt(d1, n)
        growth = np.exp((r - q) * dt)
        u = growth * p_bar / p
        return u, (growth - p * u) / (1.0 - p), p, n


class TrinomialTree(LatticePricer):
    """Boyle's (1986) trinomial tree, :math:`u = e^{\\lambda\\sigma\\sqrt{\\Delta t}}`, middle branch flat.

    With :math:`M = e^{(r-q)\\Delta t}` and :math:`V = M^2(e^{\\sigma^2\\Delta t} - 1)`,

    .. math::

        p_u = \\frac{(V + M^2 - M)u - (M - 1)}{(u - 1)(u^2 - 1)}, \\quad
        p_d = \\frac{(V + M^2 - M)u^2 - (M - 1)u^3}{(u - 1)(u^2 - 1)}, \\quad
        p_m = 1 - p_u - p_d,

    which match the mean and variance of the next price exactly.

    Parameters
    ----------
    n_steps : int
    stretch : float
        Boyle's :math:`\\lambda > 1`; the default :math:`\\sqrt{3/2}` puts
        about a third of the probability on the middle branch.

    Examples
    --------
    >>> from financialkit.options import black_scholes_price
    >>> tri = TrinomialTree(200).price(100.0, 100.0, 1.0, 0.05, 0.2).price
    >>> bool(abs(tri - black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)) < 0.01)
    True
    """

    method = "trinomial"

    def __init__(self, n_steps: int = 200, stretch: float = float(np.sqrt(1.5))):
        super().__init__(n_steps)
        if not stretch > 1.0:
            raise ValueError(f"stretch must exceed 1, got {stretch}")
        self.stretch = stretch

    def probabilities(self, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float, float, float]:
        """Return ``(u, p_u, p_m, p_d)`` for one step of this tree.

        Parameters
        ----------
        T, r, sigma, q : float

        Returns
        -------
        tuple of float
        """
        dt = T / self.n_steps
        u = float(np.exp(self.stretch * sigma * np.sqrt(dt)))
        M = np.exp((r - q) * dt)
        V = M**2 * np.expm1(sigma**2 * dt)
        denom = (u - 1.0) * (u**2 - 1.0)
        pu = ((V + M**2 - M) * u - (M - 1.0)) / denom
        pd = ((V + M**2 - M) * u**2 - (M - 1.0) * u**3) / denom
        return u, float(pu), float(1.0 - pu - pd), float(pd)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        american: bool = False,
    ) -> TreeResult:
        """Value a vanilla option by backward induction on the trinomial tree.

        Parameters
        ----------
        S, K, T, r, sigma : float
        q : float
        option_type : {"call", "put"}
        american : bool

        Returns
        -------
        TreeResult
        """
        sign = check_option_type(option_type)
        positive(S=S, K=K, T=T, sigma=sigma)
        u, pu, pm, pd = self.probabilities(T, r, sigma, q)
        if min(pu, pm, pd) < 0:
            raise ValueError("negative branch probability; increase n_steps or the stretch")
        n = self.n_steps
        dt = T / n
        lu = np.log(u)
        price, delta, gamma, boundary = _backward_induction(np.array([np.log(S)]), K, np.exp(-r * dt), (pd, pm, pu), (-lu, 0.0, lu), sign, american, n)
        return TreeResult(price=price, delta=delta, gamma=gamma, n_steps=n, method=self.method, times=np.linspace(0.0, T, n + 1), exercise_boundary=boundary)

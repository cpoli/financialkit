"""Duffie and Kan's (1996) affine term-structure models.

The state :math:`X \\in \\mathbb{R}^d` follows

.. math:: dX = (K_0 + K_1 X)\\,dt + \\operatorname{diag}(\\sigma)\\sqrt{S(X)}\\,dW, \\quad
          S_{kk}(X) = \\alpha_k + \\beta_k \\cdot X,

with independent Brownian drivers and short rate
:math:`r = \\delta_0 + \\delta_1 \\cdot X`. Duffie and Kan showed that
bond prices are then exponential-affine, :math:`P(t, T) = \\exp(A(\\tau) +
B(\\tau) \\cdot X_t)`, where :math:`A` and :math:`B` solve the Riccati
equations

.. math:: B' = -\\delta_1 + K_1^\\top B + \\tfrac12 \\sum_k \\sigma_k^2 B_k^2 \\beta_k, \\qquad
          A' = -\\delta_0 + K_0 \\cdot B + \\tfrac12 \\sum_k \\sigma_k^2 B_k^2 \\alpha_k,

from :math:`A(0) = 0, B(0) = 0`. Vasicek is the case :math:`\\beta = 0`,
CIR the case :math:`\\alpha = 0`, and a multi-factor model mixes them.
The ODEs are integrated by :func:`scipy.integrate.solve_ivp`.

References: D. Duffie and R. Kan, "A Yield-Factor Model of Interest
Rates," Mathematical Finance 6(4) (1996) 379-406; Q. Dai and K. J.
Singleton, Journal of Finance 55(5) (2000) 1943-1978.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from financialkit._validation import nonnegative
from financialkit.integrators import RNGLike, euler_maruyama
from financialkit.interest_rate_models.core.base import ShortRateModel
from financialkit.interest_rate_models.utils.numerics import affine_diffusion, affine_drift

__all__ = ["AffineModel"]


class AffineModel(ShortRateModel):
    r"""A :math:`d`-factor affine short-rate model, priced through its Riccati ODEs.

    Parameters
    ----------
    x0 : array_like, shape (d,)
        Initial state.
    K0 : array_like, shape (d,)
        Drift intercept.
    K1 : array_like, shape (d, d)
        Drift matrix.
    sigma : array_like, shape (d,)
        Volatility scale of each factor.
    alpha : array_like, shape (d,)
        Constant part of each factor's variance.
    beta : array_like, shape (d, d)
        ``beta[k]`` is the loading of factor ``k``'s variance on the state.
    delta0 : float
        Constant part of the short rate.
    delta1 : array_like, shape (d,)
        Loadings of the short rate on the state.

    Examples
    --------
    Vasicek written as an affine model gives Vasicek's bond price:

    >>> from financialkit.interest_rate_models.systems.short_rate import VasicekModel
    >>> m = AffineModel([0.03], [0.5 * 0.05], [[-0.5]], [0.01], [1.0], [[0.0]], 0.0, [1.0])
    >>> bool(np.isclose(m.zero_coupon_bond(5.0), VasicekModel(0.03, 0.5, 0.05, 0.01).zero_coupon_bond(5.0)))
    True
    """

    def __init__(self, x0, K0, K1, sigma, alpha, beta, delta0: float, delta1):
        self.x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        d = self.x0.size
        self.K0 = np.asarray(K0, dtype=np.float64).reshape(d)
        self.K1 = np.asarray(K1, dtype=np.float64).reshape(d, d)
        self.sigma = np.asarray(sigma, dtype=np.float64).reshape(d)
        self.alpha = np.asarray(alpha, dtype=np.float64).reshape(d)
        self.beta = np.asarray(beta, dtype=np.float64).reshape(d, d)
        self.delta0 = float(delta0)
        self.delta1 = np.asarray(delta1, dtype=np.float64).reshape(d)
        self.n_factors = d
        self.r0 = float(self.delta0 + self.delta1 @ self.x0)

    @classmethod
    def vasicek(cls, r0: float, kappa: float, theta: float, sigma: float) -> AffineModel:
        """Vasicek's model in affine form (state = short rate, ``beta = 0``)."""
        return cls([r0], [kappa * theta], [[-kappa]], [sigma], [1.0], [[0.0]], 0.0, [1.0])

    @classmethod
    def cir(cls, r0: float, kappa: float, theta: float, sigma: float) -> AffineModel:
        """The CIR model in affine form (state = short rate, ``alpha = 0``)."""
        return cls([r0], [kappa * theta], [[-kappa]], [sigma], [0.0], [[1.0]], 0.0, [1.0])

    def _rhs(self, tau, y):
        B = y[1:]
        q = 0.5 * self.sigma**2 * B**2
        dB = -self.delta1 + self.K1.T @ B + self.beta.T @ q
        dA = -self.delta0 + self.K0 @ B + self.alpha @ q
        return np.concatenate([[dA], dB])

    def riccati(self, tau) -> tuple[np.ndarray, np.ndarray]:
        r"""Solve the Riccati ODEs for :math:`A(\tau)` and :math:`B(\tau)`.

        Parameters
        ----------
        tau : array_like, shape (m,)
            Non-negative times to maturity.

        Returns
        -------
        A : ndarray, shape (m,)
        B : ndarray, shape (m, d)
        """
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        nonnegative(tau=tau)
        order = np.argsort(tau)
        grid = tau[order]
        A = np.zeros(tau.size)
        B = np.zeros((tau.size, self.n_factors))
        if grid[-1] > 0:
            sol = solve_ivp(self._rhs, (0.0, grid[-1]), np.zeros(self.n_factors + 1), t_eval=grid, method="LSODA", rtol=1e-10, atol=1e-12)
            A[order] = sol.y[0]
            B[order] = sol.y[1:].T
        return A, B

    def zero_coupon_bond(self, T, t: float = 0.0, state=None):
        T = np.asarray(T, dtype=np.float64)
        A, B = self.riccati(T.ravel() - t)
        x = self.x0 if state is None else np.asarray(state, dtype=np.float64)
        if x.ndim <= 1:
            out = np.exp(A + B @ x.reshape(self.n_factors))
            return out.reshape(T.shape) if T.ndim else float(out[0])
        return np.exp(A + x @ B.T)

    def simulate_factors(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None) -> tuple[np.ndarray, np.ndarray]:
        """State paths by Euler-Maruyama, with the variances truncated at zero.

        Returns
        -------
        times : ndarray, shape (n_steps + 1,)
        x : ndarray, shape (n_paths, n_steps + 1, d)
        """
        d = self.n_factors
        params = np.concatenate([[d], self.K0, self.K1.T.ravel(), self.sigma, self.alpha, self.beta.T.ravel()])
        return euler_maruyama(affine_drift, affine_diffusion, self.x0, T, n_steps, n_paths, params=params, rng=rng)

    def simulate(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None):
        times, x = self.simulate_factors(T, n_steps, n_paths, rng)
        return times, self.delta0 + x @ self.delta1

    def __repr__(self) -> str:
        return f"AffineModel(n_factors={self.n_factors}, x0={self.x0!r})"

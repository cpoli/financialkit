r"""Operator-splitting time steps for two-factor finite differences.

The semi-discrete system :math:`U'(\tau) = F(\tau, U)` is split as
:math:`F = F_0 + F_1 + F_2`, where :math:`F_0` holds the mixed derivative
and :math:`F_j` (:math:`j = 1, 2`) the derivatives in direction
:math:`j`; each is affine, :math:`F_j(\tau, U) = A_j U + g_j(\tau)`. With
:math:`\Delta\tau` the step and :math:`F^{n-1} = F(\tau_{n-1}, U^{n-1})`,
the Douglas scheme is

.. math:: Y_0 = U^{n-1} + \Delta\tau F^{n-1}, \qquad
          Y_j = Y_{j-1} + \theta\Delta\tau\,\bigl(F_j(\tau_n, Y_j) - F_j^{n-1}\bigr),
          \quad j = 1, 2,

and :math:`U^n = Y_2`. Only :math:`F_0` is explicit, and each
:math:`Y_j` solves a system :math:`(I - \theta\Delta\tau A_j) Y_j = \dots`
that is tridiagonal along the lines of direction :math:`j`. With a mixed
derivative the Douglas scheme is only first order. Craig and Sneyd (1988)
add a corrector that updates the explicit mixed term and repeats the
implicit sweeps,

.. math:: \tilde Y_0 = Y_0 + \tfrac12\Delta\tau\,\bigl(F_0(\tau_n, Y_2) - F_0^{n-1}\bigr), \qquad
          \tilde Y_j = \tilde Y_{j-1} + \theta\Delta\tau\,\bigl(F_j(\tau_n, \tilde Y_j) - F_j^{n-1}\bigr),

which is second order for :math:`\theta = 1/2`. In 't Hout and Welfert's
modified Craig-Sneyd scheme inserts
:math:`\hat Y_0 = Y_0 + \theta\Delta\tau(F_0(\tau_n, Y_2) - F_0^{n-1})`,
:math:`\tilde Y_0 = \hat Y_0 + (\tfrac12 - \theta)\Delta\tau(F(\tau_n, Y_2) - F^{n-1})`
and is second order for every :math:`\theta`; Hundsdorfer and Verwer's
scheme uses :math:`\tilde Y_0 = Y_0 + \tfrac12\Delta\tau(F(\tau_n, Y_2) - F^{n-1})`
and corrects the implicit sweeps against :math:`Y_2` instead of
:math:`U^{n-1}`.

References: I. J. D. Craig and A. D. Sneyd, *Comput. Math. Appl.* 16
(1988) 341-350; K. J. in 't Hout and B. D. Welfert, *Appl. Numer. Math.*
59 (2009) 677-692; K. J. in 't Hout and S. Foulon, *Int. J. Numer. Anal.
Model.* 7 (2010) 303-320.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.sparse.linalg import splu

__all__ = ["ADI_SCHEMES", "default_adi_theta", "difference_matrices", "adi_march"]

ADI_SCHEMES = ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer")
"""Names of the splitting schemes :func:`adi_march` runs."""

_DEFAULT_THETA = {
    "douglas": 0.5,
    "craig_sneyd": 0.5,
    "modified_craig_sneyd": 1.0 / 3.0,
    "hundsdorfer_verwer": 0.5 + np.sqrt(3.0) / 6.0,
}


def default_adi_theta(scheme: str) -> float:
    """The parameter :math:`\\theta` in 't Hout and Foulon (2010) recommend for each scheme.

    Parameters
    ----------
    scheme : {"douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"}

    Returns
    -------
    float

    Examples
    --------
    >>> default_adi_theta("craig_sneyd"), round(default_adi_theta("modified_craig_sneyd"), 4)
    (0.5, 0.3333)
    """
    if scheme not in _DEFAULT_THETA:
        raise ValueError(f"scheme must be one of {ADI_SCHEMES}, got {scheme!r}")
    return float(_DEFAULT_THETA[scheme])


def difference_matrices(x: ArrayLike) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Three-point first- and second-derivative matrices on a non-uniform grid.

    Interior rows use the central formulas of
    :func:`~financialkit.pde_methods.utils.operators.black_scholes_operator`;
    the first and last rows are zero, for the caller to fill with its
    boundary treatment.

    Parameters
    ----------
    x : array_like, shape (m + 1,)
        Strictly increasing grid.

    Returns
    -------
    D1, D2 : scipy.sparse.csr_matrix, shape (m + 1, m + 1)

    Examples
    --------
    Both are exact on quadratics:

    >>> x = np.array([0.0, 0.5, 1.5, 2.0])
    >>> D1, D2 = difference_matrices(x)
    >>> (D1 @ x**2)[1:-1].round(12).tolist(), (D2 @ x**2)[1:-1].round(12).tolist()
    ([1.0, 3.0], [2.0, 2.0])
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("x must be a strictly increasing grid of at least three nodes")
    hm, hp = np.diff(x)[:-1], np.diff(x)[1:]
    denom = hm * hp * (hm + hp)
    zero = np.zeros(1)
    d1 = [np.r_[-(hp**2) / denom, 0.0], np.r_[0.0, (hp**2 - hm**2) / denom, 0.0], np.r_[0.0, hm**2 / denom]]
    d2 = [np.r_[2.0 * hp / denom, 0.0], np.r_[zero, -2.0 * (hm + hp) / denom, zero], np.r_[0.0, 2.0 * hm / denom]]
    return sp.diags(d1, [-1, 0, 1], format="csr"), sp.diags(d2, [-1, 0, 1], format="csr")


def adi_march(
    A: Sequence[sp.spmatrix],
    g: Callable[[float], Sequence[np.ndarray]],
    U0: ArrayLike,
    T: float,
    n_time: int,
    scheme: str = "craig_sneyd",
    theta: float | None = None,
) -> np.ndarray:
    """March :math:`U' = \\sum_j (A_j U + g_j(\\tau))` from :math:`\\tau = 0` to `T` by an ADI scheme.

    Parameters
    ----------
    A : sequence of three sparse matrices, shape (m, m)
        :math:`A_0` (mixed derivative, explicit), :math:`A_1` and
        :math:`A_2` (implicit, one direction each).
    g : callable
        ``g(tau) -> (g0, g1, g2)``, the boundary contributions at time `tau`.
    U0 : array_like, shape (m,)
        Values at :math:`\\tau = 0` (the payoff).
    T : float
        Final time.
    n_time : int
        Number of uniform steps.
    scheme : {"douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"}
    theta : float, optional
        Implicitness of the sweeps; defaults to :func:`default_adi_theta`.

    Returns
    -------
    ndarray, shape (m,)
        :math:`U(T)`.

    Examples
    --------
    Without a mixed term and with :math:`A_1 = A_2 = -\\tfrac12`, every
    scheme approximates :math:`e^{-T}`; Craig-Sneyd is second order:

    >>> half = sp.identity(1, format="csr") * -0.5
    >>> zero = sp.csr_matrix((1, 1))
    >>> no_bc = lambda tau: (np.zeros(1), np.zeros(1), np.zeros(1))
    >>> U = adi_march([zero, half, half], no_bc, [1.0], 1.0, 20)
    >>> bool(abs(U[0] - np.exp(-1.0)) < 1e-4)
    True
    """
    if scheme not in ADI_SCHEMES:
        raise ValueError(f"scheme must be one of {ADI_SCHEMES}, got {scheme!r}")
    if int(n_time) != n_time or n_time < 1:
        raise ValueError(f"n_time must be a positive integer, got {n_time}")
    th = default_adi_theta(scheme) if theta is None else float(theta)
    if not 0.0 < th <= 1.0:
        raise ValueError(f"theta must lie in (0, 1], got {th}")
    A0, A1, A2 = (sp.csc_matrix(a) for a in A)
    U = np.array(U0, dtype=np.float64)
    dt = float(T) / int(n_time)
    eye = sp.identity(U.size, format="csc")
    lu = [splu(eye - th * dt * A1), splu(eye - th * dt * A2)]
    mats = (A0, A1, A2)

    def F(gs, Y):
        return [m @ Y + b for m, b in zip(mats, gs, strict=True)]

    def sweeps(Y, F_ref, g_new):
        # (I - th dt A_j) Y_j = Y_{j-1} - th dt F_j(ref) + th dt g_j(tau_n)
        for j in (1, 2):
            Y = lu[j - 1].solve(Y - th * dt * (F_ref[j] - g_new[j]))
        return Y

    for n in range(int(n_time)):
        g_old, g_new = g(n * dt), g((n + 1) * dt)
        F_old = F(g_old, U)
        Y0 = U + dt * sum(F_old)
        Y2 = sweeps(Y0, F_old, g_new)
        if scheme == "douglas":
            U = Y2
            continue
        F_new = F(g_new, Y2)
        if scheme == "craig_sneyd":
            Z0 = Y0 + 0.5 * dt * (F_new[0] - F_old[0])
        elif scheme == "modified_craig_sneyd":
            Z0 = Y0 + th * dt * (F_new[0] - F_old[0]) + (0.5 - th) * dt * (sum(F_new) - sum(F_old))
        else:
            Z0 = Y0 + 0.5 * dt * (sum(F_new) - sum(F_old))
        U = sweeps(Z0, F_new if scheme == "hundsdorfer_verwer" else F_old, g_new)
    return U

"""SDE path simulation shared across financialkit subpackages.

Drift and diffusion callbacks use the ``f(x, t, params) -> ndarray``
convention, with ``x`` the batch of paths of shape ``(n_paths, d)``, so a
single compiled callback is reused across parameter values without
recompiling (see :mod:`financialkit.integrators.sde`).

* :func:`euler_maruyama` -- strong order 1/2, weak order 1: the baseline.
* :func:`milstein` -- adds the Ito correction ``1/2 b b' (dW^2 - dt)``
  for strong order 1.
* :func:`gbm_paths`, :func:`ou_paths` -- exact sampling of geometric
  Brownian motion and the Ornstein-Uhlenbeck process, with no
  discretisation error.
* :func:`brownian_increments`, :func:`correlated_increments` (Cholesky),
  :func:`antithetic_increments` and :func:`sobol_increments` (scrambled
  Sobol via :mod:`scipy.stats.qmc`, optionally in Brownian-bridge order)
  -- the driving noise, interchangeable through the schemes' ``dW``
  argument.
* :func:`brownian_bridge` -- Brownian paths from normals by bisection,
  coarse shape first.
* :func:`strong_convergence`, :func:`weak_convergence` -- empirical
  convergence orders, returning a :class:`ConvergenceResult`.

Random numbers are drawn in Python from ``np.random.default_rng(rng)``
and passed to the kernels as arrays, so results are identical with and
without Numba. Every domain simulates paths through this module rather
than reimplementing it.

The scheme kernels are plain Python: each step is vectorised over paths,
so compiling the time loop with Numba measured no faster, and callbacks
can be any callable.

:func:`njit` is still re-exported here, for backward compatibility and
for user code that compiles its own loops: it is :func:`numba.njit` when
Numba is installed (``pip install financialkit[fast]``) and a no-op
decorator otherwise, with :data:`HAS_NUMBA` reporting which. Callbacks
decorated with it keep working.
"""

from financialkit._jit import HAS_NUMBA, njit
from financialkit.integrators.brownian import (
    RNGLike,
    antithetic_increments,
    brownian_bridge,
    brownian_increments,
    correlated_increments,
    sobol_increments,
)
from financialkit.integrators.convergence import ConvergenceResult, strong_convergence, weak_convergence
from financialkit.integrators.exact import gbm_paths, ou_paths
from financialkit.integrators.sde import SDEFunc, euler_maruyama, milstein

__all__ = [
    "HAS_NUMBA",
    "njit",
    "RNGLike",
    "SDEFunc",
    "euler_maruyama",
    "milstein",
    "gbm_paths",
    "ou_paths",
    "brownian_increments",
    "correlated_increments",
    "antithetic_increments",
    "sobol_increments",
    "brownian_bridge",
    "ConvergenceResult",
    "strong_convergence",
    "weak_convergence",
]

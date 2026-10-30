financialkit.pde_methods
========================


.. include:: /_generated/nav/pde_methods.rst

Finite-difference methods for the Black-Scholes equation: the
theta-method family (Schwartz's explicit scheme with its stability
check, Laasonen's implicit scheme and Crank-Nicolson with Rannacher's
implicit start) on uniform or Tavella-Randall spot grids, with
tridiagonal systems solved by :func:`scipy.linalg.solve_banded` or the
Thomas algorithm; knock-out barriers on a barrier-aligned grid; American
options as linear complementarity problems solved by projected SOR, the
Brennan-Schwartz algorithm or the penalty method; von Neumann
amplification factors, the explicit log-spot scheme as a trinomial
tree, the transformation to the heat equation and Richardson
extrapolation; the Peaceman-Rachford and Craig-Sneyd ADI schemes for
two-asset options; and the Heston PDE on in 't Hout and Foulon's
non-uniform grids, stepped by the Douglas, Craig-Sneyd, modified
Craig-Sneyd or Hundsdorfer-Verwer splitting.

.. automodule:: financialkit.pde_methods
   :members:
   :undoc-members:

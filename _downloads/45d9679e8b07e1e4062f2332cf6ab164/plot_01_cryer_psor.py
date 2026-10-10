r"""
Cryer's projected SOR
=====================

An American option is either held, where it satisfies the Black-Scholes
equation, or exercised, where it equals its payoff; each implicit step
becomes a linear complementarity problem
:math:`Ax \ge b,\ x \ge g,\ (Ax - b)^\top(x - g) = 0`. Colin Cryer (1971)
solved it by successive over-relaxation, projecting each update back
onto :math:`x \ge g`. :func:`~financialkit.pde_methods.psor` does this
inside :class:`~financialkit.pde_methods.CrankNicolson`; the sweep count
depends strongly on the relaxation parameter :math:`\omega`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import CRRTree, black_scholes_price
from financialkit.pde_methods import CrankNicolson
from financialkit.pde_methods.visualizers import plot_exercise_boundary

# Hull, 10th ed., Example 21.1
S, K, T, r, sigma = 50.0, 50.0, 5 / 12, 0.10, 0.40
res = CrankNicolson(200, 100, rannacher_steps=2, american_method="psor").price(S, K, T, r, sigma, option_type="put", american=True)
tree = CRRTree(2000).price(S, K, T, r, sigma, option_type="put", american=True).price
print(f"PSOR {res.price:.4f}, CRR tree {tree:.4f}, European {black_scholes_price(S, K, T, r, sigma, option_type='put'):.4f}")
print(f"{res.lcp_iterations} PSOR sweeps over 100 time steps")

# %%
# The option and its early-exercise boundary
# ------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
near = res.S < 90
axes[0].plot(res.S[near], res.V[near], label="American put")
axes[0].plot(res.S[near], np.maximum(K - res.S[near], 0.0), "k--", lw=1, label="payoff")
axes[0].set_xlabel("spot S")
axes[0].legend()
plot_exercise_boundary(res, ax=axes[1])

# %%
# Sweeps against the relaxation parameter
# ---------------------------------------

omegas = np.linspace(1.0, 1.8, 9)
sweeps = [
    CrankNicolson(100, 50, rannacher_steps=2, american_method="psor", omega=w).price(S, K, T, r, sigma, option_type="put", american=True).lcp_iterations
    for w in omegas
]
axes[2].plot(omegas, sweeps, "o-")
axes[2].set_xlabel(r"relaxation $\omega$")
axes[2].set_ylabel("total sweeps (M = 100, N = 50)")
fig.tight_layout()
plt.show()

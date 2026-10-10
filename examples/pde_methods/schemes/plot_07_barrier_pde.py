r"""
Zvan-Forsyth-Vetzal barrier options
===================================

A knock-out option is a boundary condition: the value is zero on the
barrier. Zvan, Forsyth and Vetzal (2000) showed that finite differences
price barriers accurately when a grid node sits exactly on the barrier
and the solver handles the discontinuous data smoothly, rather than
letting the barrier fall between nodes.
:meth:`~financialkit.pde_methods.ThetaScheme.price_barrier` starts the
grid at the barrier, and the result matches the method of images,
:math:`C_{do}(S) = C(S) - (H/S)^{2\lambda - 2} C(H^2/S)`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_price
from financialkit.pde_methods import CrankNicolson

S, K, H, T, r, sigma, q = 100.0, 100.0, 90.0, 1.0, 0.05, 0.25, 0.0
lam = (r - q + 0.5 * sigma**2) / sigma**2


def images(s):
    """Down-and-out call by the method of images (Hull, 10th ed., Section 26.9)."""
    return black_scholes_price(s, K, T, r, sigma, q) - (H / s) ** (2 * lam - 2) * black_scholes_price(H**2 / s, K, T, r, sigma, q)


res = CrankNicolson(400, 200, rannacher_steps=2).price_barrier(S, K, H, T, r, sigma, q, "call", "down-and-out")
print(f"PDE {res.price:.4f}, method of images {images(S):.4f}, vanilla {black_scholes_price(S, K, T, r, sigma, q):.4f}")

# %%
# A node on the barrier, against a barrier between nodes
# ------------------------------------------------------
# On a grid that ignores the barrier, the knock-out condition can only be
# imposed at the nearest node, typically half a spacing away. Moving the
# barrier by h/2 mimics that, and costs an error of order h; the aligned
# grid keeps the scheme's second order.

ms = np.array([50, 100, 200, 400])
aligned, between = [], []
for m in ms:
    cn = CrankNicolson(m, 200, rannacher_steps=2)
    h = (400.0 - H) / m
    aligned.append(abs(cn.price_barrier(S, K, H, T, r, sigma).price - images(S)))
    between.append(abs(cn.price_barrier(S, K, H + 0.5 * h, T, r, sigma).price - images(S)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
near = res.S < 160
axes[0].plot(res.S[near], res.V[near], label="down-and-out call (PDE)")
axes[0].plot(res.S[near], black_scholes_price(res.S[near], K, T, r, sigma, q), "--", label="vanilla call")
axes[0].axvline(H, color="k", lw=0.8)
axes[0].set_xlabel("spot S")
axes[0].legend()
axes[1].loglog(ms, aligned, "o-", label="node on the barrier")
axes[1].loglog(ms, between, "s-", label="barrier half a node away")
axes[1].set_xlabel("spot intervals M")
axes[1].set_ylabel("absolute error")
axes[1].legend()
fig.tight_layout()
plt.show()

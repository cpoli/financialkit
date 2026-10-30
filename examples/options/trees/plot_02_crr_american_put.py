r"""
Cox-Ross-Rubinstein: the American put and its exercise boundary
===============================================================

The binomial tree's real advantage over a closed form is early
exercise. At each node an American option is worth the larger of its
continuation value and its immediate exercise value, so backward
induction prices it as easily as a European option. For a put, early
exercise is optimal once the stock falls below a critical price
:math:`S^*(t)` that rises towards the strike as expiry approaches.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import CRRTree, black_scholes_price
from financialkit.options.visualizers import plot_exercise_boundary

# %%
# Hull's Example 21.1: five steps
# -------------------------------

res = CRRTree(5).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True)
print(f"American put on a five-step tree: {res.price:.2f} (Hull: 4.49)")
assert abs(res.price - 4.49) < 0.005
for n in (30, 50, 100, 500):
    price = CRRTree(n).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True).price
    print(f"  {n:3d} steps: {price:.3f}")
assert abs(price - 4.28) < 0.005  # Hull's converged value

# %%
# The early-exercise premium and boundary
# ---------------------------------------

K, T, r, sigma = 100.0, 1.0, 0.08, 0.25
spots = np.linspace(60, 140, 41)
american = [CRRTree(200).price(s, K, T, r, sigma, option_type="put", american=True).price for s in spots]
european = black_scholes_price(spots, K, T, r, sigma, option_type="put")
assert np.all(american >= european) and np.all(american >= np.maximum(K - spots, 0) - 1e-12)
boundary = CRRTree(400).price(100.0, K, T, r, sigma, option_type="put", american=True)
assert np.nanmax(boundary.exercise_boundary) <= K

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(spots, american, label="American (CRR, 200 steps)")
axes[0].plot(spots, european, "--", label="European (Black-Scholes)")
axes[0].plot(spots, np.maximum(K - spots, 0), "k:", label="exercise value")
axes[0].set_xlabel("spot S")
axes[0].set_ylabel("put value")
axes[0].legend()
plot_exercise_boundary(boundary, ax=axes[1])
axes[1].axhline(K, color="k", lw=0.5)
axes[1].set_title("exercise when the stock is below the boundary")
fig.tight_layout()

plt.show()

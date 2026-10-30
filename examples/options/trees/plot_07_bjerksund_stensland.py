r"""
The Bjerksund-Stensland American approximation
==============================================

Petter Bjerksund and Gunnar Stensland (1993) did not approximate the
early-exercise premium. They valued one exercise policy exactly:
exercise the first time the spot touches a flat trigger :math:`I`,
placed between the boundary just before expiry and the perpetual one.
With a flat barrier every term is a closed-form knock-out claim, so the
price needs no root search at all, and since the policy is feasible but
not optimal the result is a lower bound on the American value.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import CRRTree, barone_adesi_whaley, bjerksund_stensland, black_scholes_price

K, T, r, sigma = 100.0, 1.0, 0.08, 0.3

# %%
# A lower bound, just under the tree
# ----------------------------------
# Haug's worked example is a call on a commodity with negative carry
# (b = r - q = -4%), where early exercise captures the convenience yield.

haug = bjerksund_stensland(42.0, 40.0, 0.75, 0.04, 0.35, q=0.08)
print(f"Haug's example: {haug.price:.4f} (published 5.2704), trigger {haug.critical_price:.2f}")
assert abs(haug.price - 5.2704) < 5e-5

spots = np.linspace(70, 130, 13)
bs93 = np.array([bjerksund_stensland(s, K, T, r, sigma, option_type="put").price for s in spots])
baw = np.array([barone_adesi_whaley(s, K, T, r, sigma, option_type="put").price for s in spots])
tree = np.array([CRRTree(1500).price(s, K, T, r, sigma, option_type="put", american=True).price for s in spots])
euro = np.asarray(black_scholes_price(spots, K, T, r, sigma, option_type="put"))
print(f"put, 1 year: max shortfall vs tree {np.max(tree - bs93):.4f}, BAW max |error| {np.max(np.abs(baw - tree)):.4f}")
assert np.all(bs93 <= tree + 2e-3) and np.all(bs93 >= euro - 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(spots, bs93 - tree, "o-", label="Bjerksund-Stensland 1993")
axes[0].plot(spots, baw - tree, "s--", label="Barone-Adesi-Whaley")
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_xlabel("spot S")
axes[0].set_ylabel("approximation - 1500-step tree")
axes[0].set_title("American put: BS93 stays below")
axes[0].legend()

# %%
# The flat trigger against the true exercise boundary
# ---------------------------------------------------
# The tree's boundary rises toward the strike as expiry nears; the
# trigger is one number for the whole life of the option, which is why
# the policy leaves a little value on the table.

res = CRRTree(800).price(100.0, K, T, r, sigma, option_type="put", american=True)
trigger = bjerksund_stensland(100.0, K, T, r, sigma, option_type="put").critical_price
print(f"put trigger {trigger:.2f}; tree boundary from {np.nanmin(res.exercise_boundary):.2f} to {np.nanmax(res.exercise_boundary):.2f}")
assert np.nanmin(res.exercise_boundary) < trigger < K
axes[1].plot(res.times, res.exercise_boundary, label="optimal boundary (tree)")
axes[1].axhline(trigger, color="C1", label="Bjerksund-Stensland trigger")
axes[1].set_xlabel("time")
axes[1].set_ylabel("exercise below this spot")
axes[1].legend()
fig.tight_layout()

plt.show()

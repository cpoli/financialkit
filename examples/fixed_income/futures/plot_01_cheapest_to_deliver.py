r"""
Treasury bond futures: conversion factor and cheapest to deliver
================================================================

The Chicago Board of Trade's Treasury bond future, listed in 1977, lets
the short deliver any bond in a basket. The short is paid the futures
price times the bond's conversion factor, which is the bond's price per
1 of face at a 6% yield. If every bond yielded 6%, the factors would make
the bonds equally cheap to deliver. At other yields one bond is
cheapest, and which bond that is changes as yields move.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import CashFlows, bond_price, cheapest_to_deliver, conversion_factor, treasury_futures_price

# %%
# Hull's conversion factors and cheapest to deliver
# -------------------------------------------------
#
# The maturity is rounded down to whole quarters. 20 years and 2 months
# becomes 20 years, and 18 years and 4 months becomes 18 years and 3
# months.

print(f"10%, 20y 2m: {conversion_factor(0.10, 20 + 2 / 12):.4f}   8%, 18y 4m: {conversion_factor(0.08, 18 + 4 / 12):.4f}")
assert np.allclose([conversion_factor(0.10, 20 + 2 / 12), conversion_factor(0.08, 18 + 4 / 12)], [1.4623, 1.2199], atol=5e-5)

basis = cheapest_to_deliver(93.25, [99.50, 143.50, 119.75], [1.0382, 1.5188, 1.2615])
print("cost of delivering each bond:", basis.costs.round(2), "-> cheapest is bond", basis.cheapest + 1)
assert basis.cheapest == 1

# %%
# The futures price is the cheapest bond's forward price, here for Hull's
# 12% bond with conversion factor 1.6, delivered in 270 days at a flat 10%.

F = treasury_futures_price(0.10, 115.0, 6 * 60 / 182, CashFlows([122 / 365, 305 / 365], [6.0, 6.0]), 270 / 365, 6 * 148 / 183, 1.6)
print(f"quoted futures price {F:.2f}")
assert abs(F - 71.79) < 5e-3

# %%
# The cheapest to deliver switches at the notional yield
# ------------------------------------------------------
#
# At delivery the futures price converges to the lowest price per
# conversion factor in the basket. Above 6%, long-duration bonds lose the
# most value relative to their factors, so the longest bond is
# cheapest. Below 6% the shortest bond is cheapest.

coupons = np.array([0.04, 0.055, 0.07, 0.085])
maturities = np.array([30.0, 25.0, 20.0, 15.0])
cf = conversion_factor(coupons, maturities)
yields = np.linspace(0.03, 0.09, 121)
ratio = np.array([[bond_price(100.0, c, T, y, compounding=2) for c, T in zip(coupons, maturities, strict=True)] for y in yields]) / cf
ctd = ratio.argmin(axis=1)
assert np.allclose(ratio[np.argmin(abs(yields - 0.06))], 100.0, atol=1e-9)  # all equal at the notional yield
assert np.all(ctd[yields > 0.0601] == 0) and np.all(ctd[yields < 0.0599] == 3)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for k, (c, T) in enumerate(zip(coupons, maturities, strict=True)):
    axes[0].plot(100 * yields, ratio[:, k] - ratio.min(axis=1), label=f"{100 * c:g}% {T:g}y (CF {cf[k]:.4f})")
axes[0].axvline(6.0, color="0.6", lw=0.8)
axes[0].set_xlabel("yield (%)")
axes[0].set_ylabel("price / CF less the basket minimum")
axes[0].legend(fontsize=8)
axes[1].step(100 * yields, ctd + 1, where="mid")
axes[1].set_yticks(range(1, 5), [f"{100 * c:g}% {T:g}y" for c, T in zip(coupons, maturities, strict=True)])
axes[1].set_xlabel("yield (%)")
axes[1].set_title("cheapest to deliver")
plt.tight_layout()
plt.show()

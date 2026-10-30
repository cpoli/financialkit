r"""
STRIPS: a bond as a portfolio of zeros
======================================

In February 1985 the US Treasury's STRIPS program (Separate Trading of
Registered Interest and Principal of Securities) let dealers split a note
into one zero-coupon security per payment date. A coupon bond is then
exactly a portfolio of zeros. If the strips are worth more than the bond,
dealers buy bonds and strip them. If they are worth less, dealers buy
strips and reconstitute the bond. Either trade pulls the two prices back
together.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve, bond_cash_flows, strips_curve, strips_decomposition, strips_value, yield_to_maturity

curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
bond = bond_cash_flows(100.0, 0.055, 10.0)
strips = strips_decomposition(bond)
quotes = 100.0 * curve.discount(bond.times)

# %%
# Strip prices and the bond they rebuild
# --------------------------------------

value = strips_value(bond, quotes)
print(f"{len(strips)} strips; replication cost {value:.4f}")
assert abs(value - bond.present_value(curve)) < 1e-10
for market in (value - 0.25, value + 0.25):
    edge = value - market
    trade = "buy the bond, strip it, sell the strips" if edge > 0 else "buy the strips, reconstitute, sell the bond"
    print(f"bond at {market:.4f}: {trade} for {abs(edge):.2f}")

# %%
# One yield versus a curve of strip yields
# ----------------------------------------

implied = strips_curve(bond.times, quotes)
ytm = yield_to_maturity(value, bond)
fig, ax = plt.subplots()
ax.plot(bond.times, 100 * implied.rates, "o-", label="strip (zero) yields")
ax.axhline(100 * ytm, color="k", ls="--", label=f"bond yield to maturity {100 * ytm:.3f}%")
ax.set_xlabel("strip maturity (years)")
ax.set_ylabel("continuously compounded yield (%)")
ax.legend()
strip_error = np.abs(implied.rates - curve.zero_rate(bond.times)).max()
print("max strip-curve error vs the curve:", strip_error)
assert strip_error < 1e-12
assert implied.rates.min() < ytm < implied.rates.max()  # one yield is a blend of the strip yields
plt.show()

r"""
Bierwag-Kaufman-Toevs: convexity
================================

Duration is a tangent, but the price-yield curve bends. Bierwag, Kaufman
and Toevs (1983) brought the second-order term into portfolio practice:

.. math:: \frac{\Delta P}{P} \approx -D_{mod}\,\Delta y + \tfrac12 C\,(\Delta y)^2,
          \qquad C = \frac{1}{P}\frac{d^2P}{dy^2}.

For a long bond the convexity term removes most of the duration
approximation's error, and a barbell of the same duration as a bullet
is more convex.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import CashFlows, bond_cash_flows, bond_risk, price_from_yield
from financialkit.fixed_income.visualizers import plot_price_yield

cf = bond_cash_flows(100.0, 0.04, 30.0)
plot_price_yield(cf, 0.04, dy=0.04).set_title("30-year 4% bond")

# %%
# Approximation errors
# --------------------

risk = bond_risk(cf, 0.04)
dy = np.linspace(-0.03, 0.03, 61)
exact = price_from_yield(cf, 0.04 + dy) - risk.price
fig, ax = plt.subplots()
error_1 = risk.price_change(dy, order=1) - exact
error_2 = risk.price_change(dy, order=2) - exact
assert np.abs(error_2).max() < 0.5 * np.abs(error_1).max()  # convexity removes most of the error
ax.plot(1e4 * dy, error_1, label="duration only")
ax.plot(1e4 * dy, error_2, label="duration + convexity")
ax.set_xlabel("yield change (bp)")
ax.set_ylabel("approximation error")
ax.legend()

# %%
# Bullet versus barbell of equal duration
# ---------------------------------------

y = 0.04
bullet = CashFlows([10.0], [100.0])
w = 0.5  # half in 2-year, half in 18-year zeros: Macaulay duration 10
barbell = CashFlows([2.0, 18.0], [w * 100 * np.exp(-y * 8), (1 - w) * 100 * np.exp(y * 8)])
risks = {name: bond_risk(port, y) for name, port in (("bullet", bullet), ("barbell", barbell))}
for name, r in risks.items():
    print(f"{name:8s} D={r.macaulay_duration:.3f}  C={r.convexity:.2f}")
assert abs(risks["barbell"].macaulay_duration - risks["bullet"].macaulay_duration) < 1e-10
assert risks["barbell"].convexity > risks["bullet"].convexity
plt.show()

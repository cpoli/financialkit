r"""
The internal rate of return
===========================

Irving Fisher's *The Theory of Interest* (1930) defined the "rate of
return over cost": the discount rate at which an investment's net present
value is zero, now called the internal rate of return. For a
conventional project (one outflow followed by inflows) the NPV falls
steadily as the rate rises and crosses zero exactly once, so the IRR is
well defined and accepting projects with IRR above the cost of capital
is the same as accepting positive-NPV ones. financialkit brackets the
crossing on a grid and polishes it with :func:`scipy.optimize.brentq`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_value import irr, npv, npv_profile, plot_cash_flows, plot_npv_profile

# %%
# A conventional project
# ----------------------

cf = np.array([-5000.0, 1200.0, 1500.0, 1800.0, 2000.0])
r = irr(cf)
print(f"IRR = {r:.4%}, NPV at IRR = {npv(r, cf):.2e}")
for cost in (0.08, 0.15):
    verdict = "accept" if r > cost else "reject"
    print(f"cost of capital {cost:.0%}: NPV {npv(cost, cf):8.2f} -> {verdict}")
    assert (npv(cost, cf) > 0) == (r > cost)
assert abs(npv(r, cf)) < 1e-8

# %%
# One crossing
# ------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_cash_flows(cf, ax=axes[0])
grid = np.linspace(0.0, 0.4, 161)
plot_npv_profile(npv_profile(cf, grid), ax=axes[1])
assert np.all(np.diff(npv(grid, cf)) < 0)
axes[1].annotate(f"IRR {r:.1%}", (100 * r, 0), xytext=(100 * r + 3, 600), arrowprops={"arrowstyle": "->"})
fig.tight_layout()

plt.show()

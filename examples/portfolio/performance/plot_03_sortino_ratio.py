r"""
The Sortino ratio
=================

Volatility penalizes a fund for its best months as much as for its
worst. Frank Sortino and Lee Price (1994) measured risk instead by the
*downside deviation*, the root mean square shortfall below a minimum
acceptable return :math:`\tau`,

.. math::

   DD = \sqrt{\tfrac1n \textstyle\sum_t \min(R_t - \tau, 0)^2},

and divided the return above :math:`\tau` by it. For symmetric returns
the two ratios rank funds alike; they disagree when returns are skewed,
as for option-selling or trend-following strategies.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.portfolio import downside_deviation, sharpe_ratio, sortino_ratio

rng = np.random.default_rng(8)
n = 240
z = rng.standard_normal(n)
funds = {
    "symmetric": 0.006 + 0.03 * z,
    "short volatility": 0.012 - 0.03 * np.exp(0.9 * rng.standard_normal(n) - 0.4) * (rng.random(n) < 0.25) + 0.008 * z,
    "trend following": 0.004 + 0.02 * (np.exp(0.8 * rng.standard_normal(n)) - np.exp(0.32)) + 0.005 * z,
}

# %%
# Return distributions and the two ratios
# ---------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
for name, r in funds.items():
    ax.hist(r, bins=50, histtype="step", lw=1.5, density=True, label=f"{name}: Sharpe {sharpe_ratio(r, 0, 12):.2f}, Sortino {sortino_ratio(r, 0, 12):.2f}")
ax.set_xlabel("monthly return")
ax.legend(fontsize=8)
fig.tight_layout()

# %%
# The minimum acceptable return matters
# -------------------------------------

taus = np.linspace(-0.01, 0.01, 41)
fig, ax = plt.subplots(figsize=(7, 3.5))
for name, r in funds.items():
    ax.plot(12 * taus, [sortino_ratio(r, t, 12) for t in taus], label=name)
ax.set_xlabel("minimum acceptable return (annual)")
ax.set_ylabel("Sortino ratio")
ax.legend()
fig.tight_layout()
r = funds["symmetric"]
print(f"symmetric fund: downside deviation about the mean / volatility = {downside_deviation(r, r.mean()) / r.std():.3f} (1/sqrt 2 = 0.707)")
assert abs(downside_deviation(r, r.mean()) / r.std() - 1 / np.sqrt(2)) < 0.05
# Relative to its Sharpe ratio, Sortino penalizes the crash-prone fund and rewards the right-skewed one.
boost = {name: sortino_ratio(f, 0, 12) / sharpe_ratio(f, 0, 12) for name, f in funds.items()}
assert boost["short volatility"] < boost["symmetric"] < boost["trend following"]
plt.show()

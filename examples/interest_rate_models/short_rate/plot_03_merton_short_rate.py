r"""
Merton's Gaussian short rate
============================

In Section 6 of "Theory of Rational Option Pricing" (1973), Robert
Merton let the short rate follow an arithmetic Brownian motion,
:math:`dr = \mu\,dt + \sigma\,dW`, the first stochastic interest rate in
continuous time. Bonds price in closed form,

.. math:: P(0, T) = \exp\left(-r_0 T - \tfrac12\mu T^2 + \tfrac16\sigma^2 T^3\right),

but the cubic convexity term means long yields fall without bound and
the rate itself goes negative with growing probability. The model's
failure is what motivated mean reversion.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from financialkit.interest_rate_models import MertonShortRateModel, monte_carlo_bond_prices
from financialkit.interest_rate_models.visualizers import plot_model_yields, plot_short_rate_paths

model = MertonShortRateModel(r0=0.04, mu=0.001, sigma=0.012)

# %%
# Yields collapse at long maturities
# ----------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for sigma in (0.0, 0.006, 0.012, 0.018):
    plot_model_yields(MertonShortRateModel(0.04, 0.001, sigma), t_max=50, ax=axes[0], label=f"sigma = {100 * sigma:.1f}%")
axes[0].axhline(0, color="k", lw=0.5)
assert MertonShortRateModel(0.04, 0.001, 0.018).yields(50.0) < 0  # convexity drags the 50-year yield below zero
axes[0].set_title("Merton yield curves")

# %%
# Rates drift through zero
# ------------------------

times, r = model.simulate(30.0, 300, 4000, rng=0)
plot_short_rate_paths(times, r, n_show=15, ax=axes[1])
axes[1].axhline(0, color="k", lw=0.5)
p_neg = norm.cdf(-(0.04 + 0.001 * times[1:]) / (0.012 * np.sqrt(times[1:])))
p_neg_mc = np.mean(r[:, 1:] < 0, axis=0)
assert np.max(np.abs(p_neg_mc - p_neg)) < 0.03  # 4000 paths: a binomial SE of at most 0.8%
axes[2].plot(times[1:], 100 * p_neg, label="exact")
axes[2].plot(times[1:], 100 * p_neg_mc, ":", label="simulated")
axes[2].set_xlabel("t (years)")
axes[2].set_ylabel("P(r(t) < 0) (%)")
axes[2].legend()
fig.tight_layout()

# %%
# The closed form agrees with simulation
# --------------------------------------

res = monte_carlo_bond_prices(model, [5.0, 15.0, 30.0], n_steps=300, n_paths=10_000, rng=1)
for T, z in zip(res.maturities, res.z_scores, strict=True):
    print(f"T = {T:4.0f}: closed form {model.zero_coupon_bond(T):.5f}, Monte Carlo off by {z:+.2f} standard errors")
assert np.abs(res.z_scores).max() < 3

plt.show()

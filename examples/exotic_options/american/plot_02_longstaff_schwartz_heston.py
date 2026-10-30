r"""
Longstaff-Schwartz on Heston paths
==================================

Longstaff and Schwartz's regression needs only simulated paths and a
payoff, not a model. Nothing ties it to geometric Brownian motion: the
same backward induction prices an American put on any model that can be
simulated. Here the paths come from Heston's stochastic-volatility model,
simulated with Andersen's QE scheme, and go into
:func:`~financialkit.exotic_options.longstaff_schwartz` through its
``paths=`` argument. The regression sees only the spot, not the variance,
so the policy is slightly suboptimal and the price is a lower bound.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import longstaff_schwartz, plot_exercise_boundary
from financialkit.options import black_scholes_price
from financialkit.stochastic_volatility import HestonModel, heston_paths, heston_price

S, K, T, r = 100.0, 100.0, 1.0, 0.05
n_steps, n_paths = 50, 20_000
model = HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)

# %%
# Heston paths into the least-squares pricer
# ------------------------------------------
# The European put on the same paths checks the simulation against
# Heston's closed form; the American put adds the early-exercise premium.

sim = heston_paths(model, S, T, r, n_steps=n_steps, n_paths=n_paths, rng=0)
american = longstaff_schwartz(S, K, T, r, paths=sim.spot)
payoff = np.exp(-r * T) * np.maximum(K - sim.spot[:, -1], 0.0)
european_mc, european_se = payoff.mean(), payoff.std(ddof=1) / np.sqrt(n_paths)
european = float(heston_price(model, S, K, T, r, option_type="put"))
print(f"Heston European put: closed form {european:.4f}, on the paths {european_mc:.4f} +/- {european_se:.4f}")
print(f"Heston American put: {american.estimate:.4f} +/- {american.std_error:.4f}")
# The QE paths reproduce the European price, and early exercise is worth more.
assert abs(european_mc - european) < 4 * european_se
assert american.estimate - 4 * american.std_error > european

# %%
# Against Black-Scholes at the same long-run volatility
# -----------------------------------------------------
# With sigma = sqrt(theta) = 20% the GBM pricer is the default call. An
# at-the-money price is nearly linear in volatility, so concave in
# variance; by Jensen's inequality, random variance with the same mean
# lowers it (Hull and White's mixing argument). The Heston put is worth
# less than the Black-Scholes one, early exercise included.

sigma = np.sqrt(model.theta)
gbm = longstaff_schwartz(S, K, T, r, sigma, n_steps=n_steps, n_paths=n_paths, rng=1)
bs_european = black_scholes_price(S, K, T, r, sigma, option_type="put")
print(f"GBM American put: {gbm.estimate:.4f} +/- {gbm.std_error:.4f} (European {bs_european:.4f})")
premium = {"Heston": american.estimate - european, "Black-Scholes": gbm.estimate - bs_european}
print("early-exercise premium:", {k: round(v, 4) for k, v in premium.items()})
assert american.estimate + 4 * np.hypot(american.std_error, gbm.std_error) < gbm.estimate

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_exercise_boundary(american, ax=axes[0], label="Heston")
plot_exercise_boundary(gbm, ax=axes[0], label="Black-Scholes")
axes[0].set_ylim(70, 101)
axes[0].legend()
labels = list(premium)
axes[1].bar(labels, [european, bs_european], label="European")
axes[1].bar(labels, list(premium.values()), bottom=[european, bs_european], label="early-exercise premium")
axes[1].set_ylabel("put value")
axes[1].legend()
fig.tight_layout()
plt.show()

r"""
Feynman-Kac: bond prices by Monte Carlo
=======================================

Richard Feynman's path integrals (1948) and Mark Kac's proof (1949) tie
a parabolic PDE to an expectation over Brownian paths. For a short-rate
model the bond-pricing PDE
:math:`\partial_t P + \mu\,\partial_r P + \tfrac12\sigma^2\partial_{rr} P = rP`
is solved by

.. math:: P(0, T) = E^Q\left[\exp\left(-\int_0^T r(s)\,ds\right)\right],

so a bond can be priced by simulating the short rate and averaging the
discount factors. Every closed form in this domain is checked that way.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.interest_rate_models import VasicekModel, monte_carlo_bond_prices, path_discount_factors
from financialkit.interest_rate_models.visualizers import plot_bond_monte_carlo, plot_short_rate_paths

model = VasicekModel(r0=0.02, kappa=0.4, theta=0.05, sigma=0.015)

# %%
# Paths and pathwise discount factors
# -----------------------------------

times, r = model.simulate(10.0, 200, 2000, rng=0)
D = path_discount_factors(times, r)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
plot_short_rate_paths(times, r, n_show=15, ax=axes[0], label="Vasicek")
axes[0].set_title("simulated short rates")
axes[1].plot(times, D[:15].T, lw=0.6)
axes[1].plot(times, model.zero_coupon_bond(np.maximum(times, 1e-12)), "k", lw=2, label="closed-form P(0, T)")
axes[1].set_xlabel("T (years)")
axes[1].set_ylabel(r"$e^{-\int_0^T r\,dt}$")
axes[1].legend()
axes[1].set_title("discount factors along paths")

# %%
# The average of the discount factors is the bond price
# -----------------------------------------------------

res = monte_carlo_bond_prices(model, np.arange(1.0, 11.0), n_steps=200, n_paths=10_000, rng=1)
for T, p, se, ex in zip(res.maturities[::3], res.prices[::3], res.std_errors[::3], res.exact[::3], strict=True):
    print(f"T = {T:4.1f}: Monte Carlo {p:.5f} +/- {se:.5f}, closed form {ex:.5f}")
plot_bond_monte_carlo(res, ax=axes[2])
axes[2].set_title("Feynman-Kac vs closed form")
fig.tight_layout()

# %%
# The error falls like one over the square root of the path count
# ---------------------------------------------------------------

n = np.array([250, 1000, 4000, 16000])
se = [monte_carlo_bond_prices(model, [5.0], n_steps=50, n_paths=k, rng=2).std_errors[0] for k in n]
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(n, se, "o-", label="standard error")
ax.loglog(n, se[0] * np.sqrt(n[0] / n), "k:", label=r"$1/\sqrt{n}$")
ax.set_xlabel("paths")
ax.legend()

plt.show()

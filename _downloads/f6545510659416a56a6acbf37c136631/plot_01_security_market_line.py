r"""
Sharpe's CAPM and the security market line
==========================================

William Sharpe (1964), with Lintner and Mossin, asked what prices must
be if every investor follows Markowitz. With a riskless asset everyone
holds the same tangency portfolio, so in equilibrium it is the market
portfolio, and each asset's expected return is set by its beta on the
market alone:

.. math::

   E[r_i] = r_f + \beta_i\,(E[r_m] - r_f), \qquad \beta_i = \frac{\mathrm{Cov}(r_i, r_m)}{\sigma_m^2}.

Total volatility is not priced: only the part of an asset's risk that
the market portfolio cannot diversify away.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import capm_equilibrium, estimate_beta
from financialkit.asset_pricing.visualizers import plot_sml

names = ["utilities", "staples", "industrials", "energy", "tech"]
mu = np.array([0.055, 0.065, 0.085, 0.095, 0.12])
vol = np.array([0.14, 0.15, 0.20, 0.28, 0.30])
corr = 0.35 + 0.65 * np.eye(5)
cov = corr * np.outer(vol, vol)
eq = capm_equilibrium(mu, cov, risk_free=0.03)
print("market weights:", eq.market_weights.round(3))
print("betas:         ", eq.betas.round(3))
print("SML errors:    ", np.abs(eq.sml_errors).max())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_sml(eq, ax=axes[0], labels=names)
axes[0].set_title("expected return is linear in beta")
axes[1].plot(vol, mu, "o", color="C1")
for i, nm in enumerate(names):
    axes[1].annotate(nm, (vol[i], mu[i]), textcoords="offset points", xytext=(5, -10), fontsize=8)
axes[1].set_xlabel("volatility")
axes[1].set_title("but not in total volatility")
fig.tight_layout()

# %%
# Betas estimated from returns
# ----------------------------
#
# Five years of monthly returns give noisy betas.

rng = np.random.default_rng(1964)
R = rng.multivariate_normal(mu / 12, cov / 12, size=60)
b_hat = estimate_beta(R, R @ eq.market_weights)
print("estimated betas:", np.round(b_hat, 2))

plt.show()

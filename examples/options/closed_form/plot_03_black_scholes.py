r"""
The Black-Scholes formula
=========================

Fischer Black and Myron Scholes (1973) showed that a continuously
rebalanced position in the stock and a bond replicates a European
option when the stock follows geometric Brownian motion, so the option
price cannot depend on investors' expected return:

.. math:: c = S N(d_1) - K e^{-rT} N(d_2).

The formula equals the discounted expected payoff under the
risk-neutral drift :math:`r`, which a Monte Carlo over exact GBM paths
confirms.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.integrators import gbm_paths
from financialkit.options import black_scholes_price
from financialkit.options.visualizers import plot_option_value

# %%
# Risk-neutral Monte Carlo agrees with the formula
# ------------------------------------------------

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
_, paths = gbm_paths(S0, r, sigma, T, n_steps=1, n_paths=400_000, rng=0)
discounted = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)
print(f"Black-Scholes: {black_scholes_price(S0, K, T, r, sigma):.4f}")
print(f"Monte Carlo:   {discounted.mean():.4f} +/- {1.96 * discounted.std() / np.sqrt(discounted.size):.4f}")

# %%
# The price as a function of spot, approaching the payoff at expiry
# -----------------------------------------------------------------

S = np.linspace(60, 150, 181)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for tau in (1.0, 0.25, 0.02):
    plot_option_value(S, black_scholes_price(S, K, tau, r, sigma), K, ax=axes[0], label=f"T = {tau}")
axes[0].set_title("call value against spot")
for vol in (0.1, 0.2, 0.4):
    axes[1].plot(S, black_scholes_price(S, K, T, r, vol, option_type="put"), label=f"sigma = {vol}")
axes[1].set_xlabel("spot S")
axes[1].set_ylabel("put value")
axes[1].set_title("higher volatility, higher price")
axes[1].legend()
fig.tight_layout()

plt.show()

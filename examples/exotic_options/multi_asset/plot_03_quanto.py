r"""
Reiner quantos
==============

A quanto pays a foreign asset's option payoff in domestic currency at a
fixed exchange rate, so the investor keeps the asset's risk without the
currency's. Eric Reiner (1992) showed that, under the domestic measure,
the asset's drift picks up a correction :math:`-\rho\sigma_S\sigma_X`
from its correlation with the exchange rate:

.. math:: V = \bar X e^{-r_d T}[\hat F N(d_1) - K N(d_2)], \qquad
          \hat F = S e^{(r_f - q - \rho\sigma_S\sigma_X)T}.

A Monte Carlo under the *foreign* measure, which converts the payoff at
the floating rate, checks the drift adjustment independently.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import quanto_price
from financialkit.integrators import correlated_increments

S, K, T, rd, rf, sS, sX, q, Xbar, X0 = 100.0, 105.0, 0.5, 0.08, 0.05, 0.2, 0.1, 0.04, 1.5, 1.2
print(f"Haug (2007), 5.16: {quanto_price(S, K, T, rd, rf, sS, sX, 0.3, q=q, fixed_rate=Xbar):.4f}")
rho = np.linspace(-0.9, 0.9, 50)

fig, ax = plt.subplots(figsize=(8, 4.5))
for option_type in ("call", "put"):
    ax.plot(rho, quanto_price(S, K, T, rd, rf, sS, sX, rho, q, Xbar, option_type), label=f"quanto {option_type}")
    for c in (-0.6, 0.0, 0.6):
        W = correlated_increments([[1.0, -c], [-c, 1.0]], T, 1, 20_000, rng=int(10 * c) + 20)[:, 0, :]
        ST = S * np.exp((rf - q - 0.5 * sS**2) * T + sS * W[:, 0])
        YT = np.exp((rf - rd - 0.5 * sX**2) * T + sX * W[:, 1]) / X0  # foreign price of one domestic unit
        y = X0 * np.exp(-rf * T) * Xbar * np.maximum((1 if option_type == "call" else -1) * (ST - K), 0.0) * YT
        ax.errorbar(c, y.mean(), yerr=2 * y.std() / np.sqrt(y.size), fmt="k.", capsize=3)
ax.set_xlabel(r"correlation $\rho$ between asset and exchange rate")
ax.set_ylabel("value (domestic currency)")
ax.legend()
plt.show()

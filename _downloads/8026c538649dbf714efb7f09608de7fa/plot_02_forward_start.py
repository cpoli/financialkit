r"""
Rubinstein forward-start options
================================

A forward-start option is struck at :math:`\alpha S_t` on a future date
:math:`t`, the way executive options are granted at the money. Mark
Rubinstein (1990) noticed that on the start date it is worth
:math:`S_t` times the option on a unit spot, so today it is worth

.. math:: V = S e^{-qt}\, \mathrm{BS}(1, \alpha, T - t),

independent of where the price goes before :math:`t`. Its value is
proportional to the spot, so it has no gamma until it starts, and a
string of them is a cliquet.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import forward_start_price
from financialkit.options import black_scholes_price

S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.25, 0.02
print(f"Haug (2007), 4.3: {forward_start_price(60.0, 1.1, 0.25, 1.0, 0.08, 0.30, q=0.04):.4f}")

# %%
# Value against the start date, with a Monte Carlo check
# ------------------------------------------------------

t = np.linspace(0.0, 0.95, 40)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for alpha in (0.9, 1.0, 1.1):
    axes[0].plot(t, forward_start_price(S, alpha, t, T, r, sigma, q), label=rf"$\alpha$ = {alpha}")
rng = np.random.default_rng(0)
for ts in (0.25, 0.5, 0.75):
    z = rng.standard_normal((2, 20_000))
    St = S * np.exp((r - q - 0.5 * sigma**2) * ts + sigma * np.sqrt(ts) * z[0])
    ST = St * np.exp((r - q - 0.5 * sigma**2) * (T - ts) + sigma * np.sqrt(T - ts) * z[1])
    y = np.exp(-r * T) * np.maximum(ST - St, 0.0)
    axes[0].errorbar(ts, y.mean(), yerr=2 * y.std() / np.sqrt(y.size), fmt="k.", capsize=3)
axes[0].set_xlabel("start date t")
axes[0].set_ylabel("forward-start call")
axes[0].legend()

# %%
# Homogeneous in the spot: no gamma before the start
# --------------------------------------------------

Ss = np.linspace(50.0, 150.0, 100)
axes[1].plot(Ss, forward_start_price(Ss, 1.0, 0.5, T, r, sigma, q), label="forward start (t = 0.5)")
axes[1].plot(Ss, black_scholes_price(Ss, S, T, r, sigma, q), "--", label="vanilla call, K = 100")
axes[1].set_xlabel("spot S")
axes[1].legend()
fig.tight_layout()
plt.show()

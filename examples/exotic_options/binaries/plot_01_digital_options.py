r"""
Reiner-Rubinstein binary options
================================

Mark Rubinstein and Eric Reiner's "Unscrambling the Binary Code" (1991)
priced the all-or-nothing payoffs and showed that every vanilla and
barrier option is assembled from them. A cash-or-nothing call pays 1 if
:math:`S_T > K` and is worth :math:`e^{-rT}N(d_2)`; an asset-or-nothing
call pays the share and is worth :math:`Se^{-qT}N(d_1)`. A vanilla call
is one asset-or-nothing call minus :math:`K` cash-or-nothing calls. A
gap option uses different trigger and payoff strikes and can be worth
less than nothing, and a one-touch pays as soon as a barrier is hit.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import asset_or_nothing_price, cash_or_nothing_price, gap_option_price, one_touch_price
from financialkit.options import black_scholes_price

K, T, r, sigma = 100.0, 0.5, 0.05, 0.25
S = np.linspace(60.0, 150.0, 300)

# %%
# A vanilla call from two digitals
# --------------------------------

asset = asset_or_nothing_price(S, K, T, r, sigma)
cash = cash_or_nothing_price(S, K, T, r, sigma, cash=K)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(S, asset, label="asset-or-nothing call")
axes[0].plot(S, cash, label=f"{K:g} cash-or-nothing calls")
axes[0].plot(S, asset - cash, lw=3, label="difference")
axes[0].plot(S, black_scholes_price(S, K, T, r, sigma), "k--", label="Black-Scholes call")
assert np.allclose(asset - cash, black_scholes_price(S, K, T, r, sigma), rtol=1e-10, atol=1e-10)
axes[0].set_xlabel("spot S")
axes[0].legend()

# %%
# The digital's delta spikes at the strike near expiry
# ----------------------------------------------------

for tau in (0.5, 0.1, 0.02):
    axes[1].plot(S, cash_or_nothing_price(S, K, tau, r, sigma), label=f"T = {tau:g}")
axes[1].set_xlabel("spot S")
axes[1].set_ylabel("cash-or-nothing call (pays 1)")
axes[1].legend()

# %%
# Gap options and one-touches
# ---------------------------
# Haug (2007), Section 4.19.1: a gap call triggered at 50 that pays
# S_T - 57 is worth a fraction of a cent less than zero. The one-touch
# pays 1 the first time the price reaches 120 and is worth roughly twice
# the digital that pays at expiry if S_T > 120 (the reflection principle).

gap = gap_option_price(50.0, 50.0, 57.0, 0.5, 0.09, 0.20)
print(f"gap call: {gap:.4f}")
assert abs(gap - -0.0053) < 1e-4
# With drift the one-touch is a little under twice the digital: within 5%.
touch_ratio = one_touch_price(S[S < 120], 120.0, T, r, sigma) / (2 * cash_or_nothing_price(S[S < 120], 120.0, T, r, sigma))
assert np.all((0.95 < touch_ratio) & (touch_ratio < 1.0))
axes[2].plot(S[S < 120], one_touch_price(S[S < 120], 120.0, T, r, sigma), label="one-touch at 120")
axes[2].plot(S, cash_or_nothing_price(S, 120.0, T, r, sigma), label="digital call, K = 120")
axes[2].plot(S[S < 120], 2 * cash_or_nothing_price(S[S < 120], 120.0, T, r, sigma), ":", label="twice the digital")
axes[2].set_xlabel("spot S")
axes[2].legend()
fig.tight_layout()
plt.show()

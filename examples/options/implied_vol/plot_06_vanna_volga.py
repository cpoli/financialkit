r"""
Castagna-Mercurio: the vanna-volga FX smile
===========================================

An FX options broker quotes three numbers per expiry: the at-the-money
volatility, the 25-delta risk reversal (call minus put volatility) and
the 25-delta butterfly (their average minus at the money). Traders
priced every other strike by hedging: at the at-the-money volatility,
plus the market cost of the three pillar options that neutralise the
option's vega, vanna (:math:`\partial^2 V/\partial S\,\partial\sigma`)
and volga (:math:`\partial^2 V/\partial\sigma^2`). Antonio Castagna and
Fabio Mercurio (2007) gave the hedge weights in closed form, proved that
the method reproduces the three quotes, and derived simple
approximations of the smile it implies.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_greeks, fx_pillar_strikes, garman_kohlhagen_price, vanna_volga_price, vanna_volga_smile

# Illustrative three-month EUR/USD-style quotes (in domestic USD per EUR).
S, T, r_d, r_f = 1.10, 0.25, 0.045, 0.030
atm, rr, bf = 0.075, -0.006, 0.0025
vols = np.array([atm + bf - rr / 2, atm, atm + bf + rr / 2])  # 25-delta put, ATM, 25-delta call
pillars = fx_pillar_strikes(S, T, r_d, r_f, *vols)
print("pillar strikes:", np.round(pillars, 4), "vols:", np.round(vols, 4))

# %%
# The hedge that defines the price
# --------------------------------
# At every strike, the weights on the three pillars match the option's
# vega, vanna and volga, all computed at the at-the-money volatility.

K = 1.16
greeks = black_scholes_greeks(S, np.r_[K, pillars], T, r_d, atm, r_f)
target, pillar_greeks = (np.c_[greeks.vega, greeks.vanna, greeks.volga][i] for i in (0, slice(1, None)))
weights = np.linalg.solve(pillar_greeks.T, target)
market = garman_kohlhagen_price(S, pillars, T, r_d, r_f, vols) - garman_kohlhagen_price(S, pillars, T, r_d, r_f, atm)
by_hand = garman_kohlhagen_price(S, K, T, r_d, r_f, atm) + weights @ market
print(f"weights {np.round(weights, 4)}; price by hand {by_hand:.6f}, vanna_volga_price {vanna_volga_price(K, S, T, r_d, r_f, pillars, vols):.6f}")
assert abs(by_hand - vanna_volga_price(K, S, T, r_d, r_f, pillars, vols)) < 1e-12

# %%
# The smile and its two approximations
# ------------------------------------
# The exact smile inverts the vanna-volga prices. The first-order
# approximation is the parabola in :math:`\ln K` through the quotes,
# fine between the pillars and too steep beyond; the second-order
# formula follows the exact smile into the wings.

strikes = np.linspace(1.02, 1.18, 81)
smiles = {m: vanna_volga_smile(strikes, S, T, r_d, r_f, pillars, vols, m).vols for m in ("price", "first_order", "second_order")}
assert np.allclose(vanna_volga_smile(pillars, S, T, r_d, r_f, pillars, vols).vols, vols)
err1, err2 = (np.max(np.abs(smiles[m] - smiles["price"])) for m in ("first_order", "second_order"))
print(f"max error: first order {err1 * 1e4:.1f} bp, second order {err2 * 1e4:.2f} bp")
assert err2 < err1 / 5

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(strikes, 100 * smiles["price"], label="vanna-volga (exact)")
axes[0].plot(strikes, 100 * smiles["first_order"], "--", label="first order")
axes[0].plot(strikes, 100 * smiles["second_order"], ":", lw=2, label="second order")
axes[0].plot(pillars, 100 * vols, "ko", label="quotes")
axes[0].set_xlabel("strike")
axes[0].set_ylabel("implied volatility (%)")
axes[0].legend()

# %%
# The cost of the smile
# ---------------------
# The vanna-volga price minus the flat at-the-money price is what the
# smile adds: the 25-delta risk reversal makes downside puts dearer
# than upside calls.

premium = vanna_volga_price(strikes, S, T, r_d, r_f, pillars, vols) - garman_kohlhagen_price(S, strikes, T, r_d, r_f, atm)
axes[1].plot(strikes, 1e4 * premium)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_xlabel("strike")
axes[1].set_ylabel("smile premium on a call (pips)")
fig.tight_layout()

plt.show()

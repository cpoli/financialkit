r"""
Negative oil futures: Bachelier returns to the exchange
=======================================================

On 20 April 2020, the day before it expired, the May WTI crude oil
futures contract settled at :math:`-37.63` dollars a barrel: storage at
Cushing was full and holders paid to be rid of delivery. Black's
lognormal model cannot price an option on a negative futures price,
and CME Clearing had already announced that from 22 April it would
value its crude oil options with Bachelier's 1900 normal model, in
which prices can take any sign. The example prices options around a
negative futures price, inverts the normal implied volatility, and
compares the two models where both apply.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import bachelier_implied_volatility, bachelier_price, black76_price, implied_volatility

r, T = 0.002, 20 / 365  # near-zero rates and a 20-day option, as in April 2020

# %%
# Black-76 breaks, Bachelier does not
# -----------------------------------

F_neg = -37.63
try:
    black76_price(F_neg, 5.0, T, r, 1.0)
except ValueError as err:
    print("Black-76:", err)
else:
    raise AssertionError("Black-76 should reject a negative forward")

strikes = np.linspace(-60.0, 20.0, 81)
sigma_n = 120.0  # dollars per barrel per sqrt(year): a 20-day standard deviation of about 28 dollars
calls = bachelier_price(F_neg, strikes, T, r, sigma_n, q=r)  # q = r: an option on a futures price
puts = bachelier_price(F_neg, strikes, T, r, sigma_n, q=r, option_type="put")
assert np.allclose(calls - puts, np.exp(-r * T) * (F_neg - strikes))  # put-call parity with any sign

recovered = bachelier_implied_volatility(puts, F_neg, strikes, T, r, r, "put")
print(f"normal vol recovered from the puts: max error {np.max(np.abs(recovered - sigma_n)):.1e}")
assert np.max(np.abs(recovered - sigma_n)) < 1e-8

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(strikes, calls, label="call")
axes[0].plot(strikes, puts, label="put")
axes[0].axvline(F_neg, color="k", lw=0.5, ls="--", label=f"futures {F_neg}")
axes[0].axvline(0.0, color="grey", lw=0.5)
axes[0].set_xlabel("strike ($/bbl)")
axes[0].set_ylabel("Bachelier price ($/bbl)")
axes[0].set_title("options on a negative futures price")
axes[0].legend()

# %%
# The same prices in the two models before the crash
# --------------------------------------------------
# At a positive futures price both models work. A flat normal volatility
# seen through Black-76 is a lognormal smile that rises steeply as the
# strike falls toward zero, and it has no lognormal volatility at all
# once the futures price itself approaches zero.

for F in (20.0, 5.0):
    K = np.linspace(0.3 * F, 1.7 * F, 41)
    prices = bachelier_price(F, K, T, r, 0.5 * F * 1.2, q=r, option_type="put")  # same 60% lognormal-equivalent at the money
    black_vols = implied_volatility(prices, F, K, T, r, q=r, option_type="put", method="jaeckel")
    axes[1].plot(K / F, black_vols, label=f"F = {F:g}")
    assert np.all(np.diff(black_vols) < 0)  # the normal model is a downward lognormal skew
axes[1].set_xlabel("strike / futures")
axes[1].set_ylabel("Black-76 implied volatility")
axes[1].set_title("flat Bachelier vol as a lognormal smile")
axes[1].legend()
fig.tight_layout()

plt.show()

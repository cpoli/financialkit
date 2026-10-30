r"""
Haug-Haug-Lewis: discrete cash dividends
========================================

A stock that pays a cash dividend :math:`D` at :math:`t_D` drops by
:math:`D`, so it is not a geometric Brownian motion over the option's
life. The usual fix, Hull's escrowed model, subtracts the dividend's
present value from the spot and keeps the volatility. Espen Haug, Jørgen
Haug and Alan Lewis (2003) went back to the definition: just after the
dividend the option is a plain European on the reduced stock, so today's
value is its expectation over the stock just before,

.. math:: c_0 = e^{-r t_D}\,\mathbb{E}\big[c_{BS}(\max(S_{t_D} - D, 0), K, T - t_D)\big],

one integral per dividend.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_price, discrete_dividend_price

S, K, T, r, sigma, D = 100.0, 100.0, 1.0, 0.06, 0.3, 7.0

# %%
# Checking the integral
# ---------------------
# A dividend at expiry turns the call into a call struck at K + D, and a
# simulation of the dropping stock agrees.

at_expiry = discrete_dividend_price(S, K, T, r, sigma, [T], [D])
print(f"dividend at T: {at_expiry:.6f}, call struck at K + D: {black_scholes_price(S, K + D, T, r, sigma):.6f}")
assert abs(at_expiry - black_scholes_price(S, K + D, T, r, sigma)) < 1e-10

rng = np.random.default_rng(0)
n, t_D = 200_000, 0.5
z1, z2 = rng.standard_normal((2, n))
S_tD = S * np.exp((r - 0.5 * sigma**2) * t_D + sigma * np.sqrt(t_D) * z1)
S_T = np.maximum(S_tD - D, 0.0) * np.exp((r - 0.5 * sigma**2) * (T - t_D) + sigma * np.sqrt(T - t_D) * z2)
payoff = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
hhl = discrete_dividend_price(S, K, T, r, sigma, [t_D], [D])
print(f"mid-life dividend: HHL {hhl:.4f}, Monte Carlo {payoff.mean():.4f} +/- {payoff.std() / np.sqrt(n):.4f}")
assert abs(payoff.mean() - hhl) < 4 * payoff.std() / np.sqrt(n)

# %%
# Where the escrowed model goes wrong
# -----------------------------------
# Subtracting PV(D) from the spot shrinks the volatility of the whole
# path. That is right for a dividend paid today and increasingly wrong
# as the dividend moves toward expiry, where the stock carried its full
# volatility almost all the way.

times = np.linspace(0.01, 0.99, 25)
exact = np.array([discrete_dividend_price(S, K, T, r, sigma, [t], [D]) for t in times])
escrowed = np.asarray(black_scholes_price(S - D * np.exp(-r * times), K, T, r, sigma))
print(f"escrowed error: {exact[0] - escrowed[0]:.3f} for an early dividend, {exact[-1] - escrowed[-1]:.3f} for a late one")
assert abs(exact[0] - escrowed[0]) < 0.05 and exact[-1] - escrowed[-1] > 0.5

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(times, exact, label="Haug-Haug-Lewis")
axes[0].plot(times, escrowed, "--", label="escrowed (S - PV(D))")
axes[0].set_xlabel("dividend date $t_D$")
axes[0].set_ylabel("call price")
axes[0].set_title(f"at-the-money call, D = {D:g}")
axes[0].legend()

# %%
# Several dividends
# -----------------
# Semi-annual dividends nest the integrals (the cost grows like the
# number of nodes to the power of the number of dividends). The call
# loses less than the present value of the payments, because only the
# part of the drop that lands in the money costs its holder anything.

strikes = np.linspace(70, 130, 13)
semiannual = discrete_dividend_price(S, strikes, T, r, sigma, [0.25, 0.75], [3.5, 3.5])
none = np.asarray(black_scholes_price(S, strikes, T, r, sigma))
pv = 3.5 * (np.exp(-r * 0.25) + np.exp(-r * 0.75))
assert np.all((semiannual < none) & (none - semiannual < pv))
axes[1].plot(strikes, none - semiannual, "o-")
axes[1].axhline(pv, color="k", lw=0.5, label="PV of the dividends")
axes[1].legend()
axes[1].set_xlabel("strike")
axes[1].set_ylabel("value lost to two 3.5 dividends")
fig.tight_layout()

plt.show()

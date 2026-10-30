r"""
Gatheral and Jacquier's Arbitrage-Free SSVI Surfaces
====================================================

Fitting SVI one expiry at a time can leave calendar-spread arbitrage
between slices and butterfly arbitrage inside them. Gatheral and
Jacquier (2014) parametrised the whole surface at once by the
at-the-money total variance :math:`\theta_T`, a skew :math:`\rho` and a
curvature function :math:`\varphi(\theta)`,

.. math::

   w(k, \theta_T) = \frac{\theta_T}{2}\left[1 + \rho\varphi(\theta_T) k
   + \sqrt{(\varphi(\theta_T) k + \rho)^2 + 1 - \rho^2}\right],

and proved simple conditions for no static arbitrage. With the power law
:math:`\varphi = \eta\,\theta^{-\gamma}(1 + \theta)^{\gamma - 1}`, a rising
:math:`\theta_T`, :math:`\gamma \le 1/2` and :math:`\eta(1 + |\rho|) \le 2`
suffice. Here SSVI is fitted to Heston smiles at four expiries with
those bounds built in, checked by Durrleman's function on every slice,
and contrasted with a curvature that breaks the butterfly condition.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import implied_volatility
from financialkit.stochastic_volatility import HestonModel, SSVISurface, heston_price, ssvi_surface

S, r = 100.0, 0.0
model = HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
expiries = np.array([0.25, 0.5, 1.0, 2.0])
strikes = np.linspace(70.0, 140.0, 15)
K, T = np.meshgrid(strikes, expiries)
vols = np.asarray(implied_volatility(np.array([heston_price(model, S, strikes, t, r) for t in expiries]), S, K, T, r))

fit = ssvi_surface(K.ravel(), T.ravel(), vols.ravel(), S, r)
print(f"SSVI: rho = {fit.rho:.3f}, eta = {fit.eta:.3f}, gamma = {fit.gamma:.3f}; rms vol error {fit.rmse * 1e4:.0f} bp")
print("theta_T =", np.round(fit.theta, 4), "| eta (1 + |rho|) =", round(fit.eta * (1 + abs(fit.rho)), 3))
print("Gatheral-Jacquier conditions hold:", fit.is_arbitrage_free())
print("Durrleman and calendar checks on the raw SVI slices:", fit.to_svi_surface().is_arbitrage_free(tol=1e-12))
assert fit.is_arbitrage_free() and fit.to_svi_surface().is_arbitrage_free(tol=1e-12) and fit.rmse < 0.01

# %%
# The fitted smiles, and :math:`w(k)` rising with :math:`T` at every
# log-moneyness: no calendar spread has a negative price.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
kk = np.linspace(-0.6, 0.5, 200)
for t, v in zip(expiries, vols, strict=True):
    F = float(fit.forward(t))
    line = ax1.plot(np.log(strikes / F), v, "o", ms=3)[0]
    ax1.plot(kk, np.sqrt(np.asarray(fit.total_variance(kk, t)) / t), color=line.get_color(), label=f"T = {t}")
    ax2.plot(kk, fit.total_variance(kk, t), color=line.get_color(), label=f"T = {t}")
ax1.set_xlabel("log-moneyness k")
ax1.set_ylabel("implied volatility")
ax1.set_title("Heston quotes and the SSVI fit")
ax1.legend()
ax2.set_xlabel("log-moneyness k")
ax2.set_ylabel("total variance w(k, T)")
ax2.set_title("No crossing: no calendar arbitrage")
ax2.legend()

# %%
# A curvature with :math:`\eta(1 + |\rho|) = 6.4`, far above 2, makes
# Durrleman's :math:`g` negative in the wings: the implied density would
# be negative there.
steep = SSVISurface(fit.rho, fit.expiries, fit.theta, S, r, eta=4.0, gamma=0.5)
kg = np.linspace(-1.5, 1.5, 301)
g_fit = fit.to_svi_surface().butterfly_arbitrage(kg)
g_steep = steep.to_svi_surface().butterfly_arbitrage(kg)
ax3.plot(kg, g_fit[-1], label=f"fitted, eta = {fit.eta:.2f}")
ax3.plot(kg, g_steep[-1], label="eta = 4")
ax3.axhline(0.0, color="k", lw=0.8)
ax3.set_xlabel("log-moneyness k")
ax3.set_ylabel("Durrleman g(k), T = 2")
ax3.set_title("Butterfly arbitrage")
ax3.legend()
print("eta = 4 passes the conditions:", steep.is_arbitrage_free(), "| min g:", round(float(g_steep.min()), 3))
assert not steep.is_arbitrage_free() and g_steep.min() < 0
fig.tight_layout()

plt.show()

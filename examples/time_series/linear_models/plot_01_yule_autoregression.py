r"""
Yule's autoregression
=====================

Schuster's periodogram treated sunspot numbers as a sum of exact cycles
plus measurement error. Udny Yule (1927) proposed instead a pendulum
"bombarded by peas": a damped oscillation kept alive by random shocks,

.. math:: x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \varepsilon_t,

with :math:`\phi_1 \approx 1.34` and :math:`\phi_2 \approx -0.65` for
Wolfer's sunspot series. Complex roots of :math:`1 - \phi_1 z - \phi_2 z^2`
give a *pseudo*-cycle whose phase and amplitude wander, which is exactly
what the sunspot record looks like. The coefficients come from the
Yule-Walker equations, a Toeplitz system in the autocorrelations.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series import ar_cycle_period, arma_acf, autocorrelation, simulate_arma, yule_walker

# %%
# A disturbed pendulum
# --------------------
# Two hundred "years" of Yule's AR(2) look cyclical, but no two cycles have
# the same length or height.

phi = [1.34, -0.65]
x = simulate_arma(ar=phi, n=200, rng=1927)
print(f"pseudo-period implied by the roots: {ar_cycle_period(phi):.1f} years")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(1750, 1950), x, color="C0")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("year")
ax.set_ylabel("deviation")
ax.set_title("Yule's AR(2): a pendulum bombarded by random shocks")
fig.tight_layout()

# %%
# Estimating it by Yule-Walker
# ----------------------------
# The sample autocorrelations oscillate and decay, as the model's do. Solving
# the Yule-Walker equations on a longer simulated record recovers the
# coefficients and the period.

long = simulate_arma(ar=phi, n=5000, rng=1)
fit = yule_walker(long, 2)
print(f"Yule-Walker estimates: phi = {fit.ar.round(3)}, sigma^2 = {fit.sigma2:.3f}")
print(f"estimated pseudo-period: {ar_cycle_period(fit.ar):.1f} years")
assert np.allclose(fit.ar, phi, atol=0.03)
assert abs(ar_cycle_period(fit.ar) - ar_cycle_period(phi)) < 0.5
assert np.max(np.abs(autocorrelation(long, 30) - arma_acf(ar=phi, nlags=30))) < 0.1

fig, ax = plt.subplots()
k = np.arange(31)
ax.bar(k, autocorrelation(long, 30), color="0.7", label="sample")
ax.plot(k, arma_acf(ar=phi, nlags=30), "C3o-", ms=3, label="AR(2) theory")
ax.set_xlabel("lag (years)")
ax.set_ylabel("autocorrelation")
ax.legend()
plt.show()

r"""
Gatheral, Jaisson and Rosenbaum: Volatility Is Rough
====================================================

Gatheral, Jaisson and Rosenbaum (2018) took realized-variance estimates
of daily volatility for thousands of days and many indices, and measured
how log-volatility increments scale with the lag:

.. math::

   m(q, \Delta) = \overline{|\log\sigma_{t+\Delta} - \log\sigma_t|^q} \propto \Delta^{\zeta_q}, \qquad
   \zeta_q = qH.

The lines in :math:`\log\Delta` were straight, the slopes linear in
:math:`q`, and :math:`H \approx 0.1` everywhere: log-volatility behaves
like a fractional Brownian motion much rougher than Brownian motion. They
called the model RFSV (rough fractional stochastic volatility). This
example uses synthetic data, so the result follows from how the data
were generated. It runs the estimator on a simulated RFSV log-volatility
with :math:`H = 0.14` (sampled exactly by Davies and Harte's circulant
embedding), and on the log-variance of a Heston model sampled daily,
whose short-lag scaling is Brownian, with :math:`H = 1/2`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility import HestonModel, fbm_paths, heston_paths, volatility_roughness
from financialkit.stochastic_volatility.visualizers import plot_roughness

n_days = 4000
days = n_days / 252
# RFSV: log sigma_t = log sigma_0 + nu W^H_t, with nu as estimated for the S&P 500.
H_true, nu = 0.14, 0.3
_, B = fbm_paths(H_true, n_days, n_days, 1, rng=2018)
log_vol = np.log(0.2) + nu * B[0]
rfsv = volatility_roughness(log_vol)
print(f"RFSV log-volatility: H = {rfsv.H:.3f} (true {H_true}); zeta_q / q =", np.round(rfsv.zeta / rfsv.q, 3))

# %%
# Heston's variance, sampled once a day by Andersen's QE scheme with the
# Feller condition satisfied, so the log is defined.
heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
hp = heston_paths(heston, 100.0, days, 0.0, n_steps=n_days, n_paths=1, rng=7)
smooth = volatility_roughness(0.5 * np.log(hp.variance[0]), lags=np.arange(1, 21))
print(f"Heston log-volatility: H = {smooth.H:.3f}")
assert abs(rfsv.H - H_true) < 0.03 and abs(smooth.H - 0.5) < 0.1

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
t = np.arange(n_days + 1) / 252
axes[0].plot(t, np.exp(log_vol), lw=0.6, label=f"RFSV, H = {H_true}")
axes[0].plot(t, np.sqrt(hp.variance[0]), lw=0.6, label="Heston")
axes[0].set_xlabel("years")
axes[0].set_ylabel("volatility")
axes[0].legend()
plot_roughness(rfsv, ax=axes[1])
axes[1].set_title(f"RFSV: H = {rfsv.H:.3f}")
plot_roughness(smooth, ax=axes[2])
axes[2].set_title(f"Heston: H = {smooth.H:.3f}")
fig.tight_layout()

plt.show()

r"""
In 't Hout-Foulon: the Heston PDE by ADI
========================================

Heston's model adds a second state variable, the variance, and a strong
correlation between it and the spot, so its pricing PDE has a large
mixed derivative, coefficients that vanish at :math:`v = 0` and a
boundary where Feller's condition may fail. Karel in 't Hout and Sven
Foulon (2010) put the ADI family on this problem: non-uniform grids
clustered at the strike and at zero variance, an upwind difference
where the variance drifts down, and four splittings, Douglas,
Craig-Sneyd, modified Craig-Sneyd and Hundsdorfer-Verwer, compared on
six parameter sets. :class:`~financialkit.pde_methods.HestonADI`
follows their recipe; Heston's semi-closed formula checks it.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.pde_methods import HestonADI
from financialkit.pde_methods.visualizers import plot_heston_surface
from financialkit.stochastic_volatility import HestonModel, heston_price

# in 't Hout and Foulon's case A: a strong negative correlation, rho = -0.9
K, T, r, v0, kappa, theta, xi, rho = 100.0, 1.0, 0.025, 0.04, 1.5, 0.04, 0.3, -0.9
model = HestonModel(v0, kappa, theta, xi, rho)
exact = heston_price(model, 100.0, K, T, r)
res = HestonADI(100, 50, 50).price(100.0, K, T, r, v0, kappa, theta, xi, rho)
print(f"HestonADI {res.price:.4f}, Heston formula {exact:.4f}")
assert abs(res.price - exact) < 0.01

# %%
# The value over spot and variance, and the grid
# ----------------------------------------------
#
# Every tenth grid line is drawn: the spot nodes are uniform between
# :math:`e^{-T/4}K` and :math:`K` and stretch out beyond, and the
# variance nodes crowd towards zero, where the PDE degenerates.

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_heston_surface(res, ax=axes[0], s_range=(0, 200), v_max=0.5)
for s in res.S[::5]:
    axes[0].axvline(s, color="w", lw=0.3)
for v in res.v[::3]:
    axes[0].axhline(v, color="w", lw=0.3)
axes[0].set_xlim(0, 200)
axes[0].set_title("call value, case A")

spots = np.linspace(60.0, 160.0, 11)
pde = np.array([float(np.interp(s, res.S, res.V[:, np.searchsorted(res.v, v0)])) for s in spots])
formula = np.array([heston_price(model, s, K, T, r) for s in spots])
axes[1].plot(spots, formula, "k", lw=1, label="Heston formula")
axes[1].plot(spots, pde, "o", label="ADI grid (v = 0.04)")
axes[1].set_xlabel("S")
axes[1].set_ylabel("call value")
axes[1].legend()
fig.tight_layout()
plt.show()

# %%
# Which splitting?
# ----------------
#
# On a fixed 100 x 50 grid, the error against a 1000-step solve over
# :math:`[K/2, 3K/2] \times [0, 1]`, the region in 't Hout and Foulon
# measure. Douglas is first order. Craig-Sneyd with :math:`\theta = 1/2`
# barely damps the high-frequency error from the payoff kink until the
# step is small; the modified Craig-Sneyd scheme with
# :math:`\theta = 1/3` and Hundsdorfer-Verwer are second order from the
# start, which is why in 't Hout and Foulon recommend them.

ref = HestonADI(100, 50, 1000, scheme="hundsdorfer_verwer").price(100.0, K, T, r, v0, kappa, theta, xi, rho).V
window = np.ix_((res.S >= 50) & (res.S <= 150), res.v <= 1.0)
steps = np.array([5, 10, 20, 40, 80])
fig, ax = plt.subplots(figsize=(6, 4.5))
errors = {}


def window_error(n, scheme):
    V = HestonADI(100, 50, n, scheme=scheme).price(100.0, K, T, r, v0, kappa, theta, xi, rho).V
    return np.abs(V - ref)[window].max()


for scheme in ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"):
    errors[scheme] = np.array([window_error(n, scheme) for n in steps])
    ax.loglog(steps, errors[scheme], "o-", label=scheme.replace("_", " "))
ax.loglog(steps, 2.0 / steps**2, "k:", lw=1, label="slope -2")
ax.set_xlabel("time steps")
ax.set_ylabel("max error on the window")
ax.legend(fontsize=8)
assert errors["modified_craig_sneyd"][-1] < 0.05 * errors["douglas"][-1]
plt.show()

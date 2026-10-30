r"""
Gârleanu-Pedersen: trading with transaction costs
=================================================

A fund trades on two return predictors: a fast one that decays with a
half-life of five days and a slow one with a half-life of a year. With
no trading costs the best position is Markowitz's,
:math:`(\gamma\Sigma)^{-1}Bf_t`, and it chases every move of the fast
signal. Nicolae Gârleanu and Lasse Pedersen (2013) solved the dynamic
problem with quadratic costs :math:`\tfrac12\lambda\,\Delta x'\Sigma\Delta x`
in closed form. The optimal rule is to *aim in front of the target* and
*trade partially towards the aim*:

.. math:: x_t = \left(1 - \tfrac{a}{\lambda}\right)x_{t-1} + \tfrac{a}{\lambda}\,\mathrm{aim}_t,
          \qquad \mathrm{aim}_t = (\gamma\Sigma)^{-1}B\left(I + \tfrac{a}{\gamma}\Phi\right)^{-1}f_t.

The aim discounts each signal by how fast it decays: by the time a
slowly traded position gets there, a fast signal has faded and a slow
one has not.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from financialkit.portfolio import garleanu_pedersen

rng = np.random.default_rng(8)
half_life = np.array([5.0, 250.0])
phi = 1 - 2 ** (-1 / half_life)
sigma, gamma, lam = 0.02, 5.0, 5000.0
B = np.array([[0.002, 0.0005]])
S = np.array([[sigma**2]])

T = 100_000
f = np.empty((T, 2))
f[0] = rng.standard_normal(2)
shocks = rng.standard_normal((T, 2)) * np.sqrt(1 - (1 - phi) ** 2)
for t in range(1, T):
    f[t] = (1 - phi) * f[t - 1] + shocks[t]
r = f[:-1] @ B.T[:, 0] + sigma * rng.standard_normal(T - 1)  # r[t] is earned on the position held after f[t]

policy = garleanu_pedersen(B, S, phi, gamma, lam)
print(f"trade rate a / lambda = {policy.trade_rate:.3f} per day")
print("aim / Markowitz loading, fast and slow:", np.round(policy.aim_loadings / policy.markowitz_loadings, 3))
assert (policy.aim_loadings / policy.markowitz_loadings)[0, 0] < 0.5 < (policy.aim_loadings / policy.markowitz_loadings)[0, 1]


def utility(x):
    """Average daily mean-variance utility net of trading costs."""
    x = x[:-1]
    dx = np.diff(np.r_[0.0, x])
    return np.mean(x * r - 0.5 * gamma * sigma**2 * x**2 - 0.5 * lam * sigma**2 * dx**2)


# %%
# Aim in front of the target, trade partially towards it
# ------------------------------------------------------
# Markowitz's position jumps with the fast signal. The aim leans on the
# slow signal, and the optimal position follows the aim smoothly.

days = slice(2000, 2400)
markowitz = policy.markowitz(f)[:, 0]
aim = policy.aim(f)[:, 0]
gp = policy.positions(f)[:, 0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
t = np.arange(T)[days]
axes[0].plot(t, markowitz[days], color="0.7", lw=0.8, label="Markowitz")
axes[0].plot(t, aim[days], "C1", lw=1.2, label="aim portfolio")
axes[0].plot(t, gp[days], "C0", lw=2, label="optimal position")
axes[0].set_xlabel("day")
axes[0].set_ylabel("position (fraction of capital)")
axes[0].legend(fontsize=8)
turnover = {name: np.abs(np.diff(x)).mean() for name, x in (("Markowitz", markowitz), ("optimal", gp))}
print("mean daily turnover:", {k: round(v, 4) for k, v in turnover.items()})
assert turnover["optimal"] < turnover["Markowitz"] / 10

# %%
# Better than any static compromise
# ---------------------------------
# A natural fix is to trade a fixed fraction :math:`\tau` of the way to
# Markowitz each day. Even at its best :math:`\tau` it earns less than
# the Gârleanu-Pedersen rule over 100,000 simulated days, because it
# gives the fast signal the same weight as the slow one. Trading straight
# to Markowitz loses money once costs are paid.

taus = np.geomspace(2e-4, 1.0, 30)
partial = [utility(signal.lfilter([tau], [1, tau - 1], markowitz)) for tau in taus]
u_gp, u_mark = utility(gp), utility(markowitz)
axes[1].semilogx(taus, 1e4 * np.array(partial), "o-", ms=3, label=r"trade $\tau$ of the way to Markowitz")
axes[1].axhline(1e4 * u_gp, color="C0", lw=2, label="Gârleanu-Pedersen")
axes[1].axvline(policy.trade_rate, color="C0", ls=":", lw=1)
axes[1].set_ylim(-0.5, 1.3e4 * u_gp)
axes[1].set_xlabel(r"daily trade rate $\tau$")
axes[1].set_ylabel("net utility (bp per day)")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
print(f"net utility, bp/day: optimal {1e4 * u_gp:.3f}, best partial {1e4 * max(partial):.3f}, Markowitz {1e4 * u_mark:.3f}")
assert u_gp > max(partial) and u_mark < 0 < u_gp
assert 0 < np.argmax(partial) < len(taus) - 1  # the best static rate is inside the grid

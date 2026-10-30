r"""
Lee's moment formula
====================

Roger Lee (2004) proved that the total implied variance can grow at
most linearly in the log strike, and that the slope is fixed by how
many moments the price has. If :math:`\tilde p` is the supremum of the
:math:`p` with :math:`E[S_T^{1+p}] < \infty`, then

.. math:: \limsup_{k \to \infty} \frac{\sigma_{BS}^2(k)\,T}{k} = 2 - 4\left(\sqrt{\tilde p^2 + \tilde p} - \tilde p\right),

and the left wing obeys the same formula with the negative moments.
Kou's model makes this concrete: its exponential jump tails give
:math:`\tilde p = \eta_1 - 1` and :math:`\tilde q = \eta_2`. The bound
is approached slowly, with corrections of order :math:`\ln|k|/|k|`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility import KouModel, VarianceGammaModel, lee_wing_slope

S, r, T = 100.0, 0.0, 1.0
k = np.linspace(0.25, 6.0, 24)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for eta1, eta2 in ((3.0, 2.0), (6.0, 4.0)):
    m = KouModel(0.1, 1.0, 0.5, eta1, eta2)
    p, q = m.critical_moments()
    right = m.implied_vol(S, S * np.exp(k), T, r) ** 2 * T / k
    left = m.implied_vol(S, S * np.exp(-k), T, r) ** 2 * T / k
    line = ax1.plot(k, right, label=f"right wing, eta1 = {eta1:g}")[0]
    ax1.axhline(lee_wing_slope(p), color=line.get_color(), ls="--", lw=1)
    line = ax2.plot(k, left, label=f"left wing, eta2 = {eta2:g}")[0]
    ax2.axhline(lee_wing_slope(q), color=line.get_color(), ls="--", lw=1)
    print(f"eta1 = {eta1:g}, eta2 = {eta2:g}: critical moments p = {p:g}, q = {q:g}; Lee slopes {lee_wing_slope(p):.4f}, {lee_wing_slope(q):.4f}")
    assert (p, q) == (eta1 - 1, eta2)
    # Each wing closes in on its bound: nearer at |k| = 6 than at |k| = 2.75, and within 0.1.
    for ratio, bound in ((right, lee_wing_slope(p)), (left, lee_wing_slope(q))):
        assert abs(ratio[-1] - bound) < min(0.1, abs(ratio[10] - bound))
for ax in (ax1, ax2):
    ax.set_xlabel("|log-moneyness| |k|")
    ax.set_ylabel("w(k) / |k|")
    ax.set_ylim(0, 1)
    ax.legend()
ax1.set_title("dashed: Lee's bound")
fig.tight_layout()

# %%
# Variance gamma
# --------------

sigma, nu, theta = 0.2, 0.5, -0.2
p, q = VarianceGammaModel(sigma, nu, theta).critical_moments()
print(f"variance gamma: p = {p:.3f}, q = {q:.3f}, wing slopes {lee_wing_slope(p):.4f} (right), {lee_wing_slope(q):.4f} (left)")
# The moment generating function blows up at the roots of 1 - theta nu u - sigma^2 nu u^2 / 2.
u_minus, u_plus = np.sort(np.roots([-0.5 * sigma**2 * nu, -theta * nu, 1.0]))
assert np.isclose(p, u_plus - 1) and np.isclose(q, -u_minus)

plt.show()

r"""
Giles-Glasserman adjoint Greeks ("smoking adjoints")
====================================================

A rates desk wants the delta of its swaption book to every point of the
forward curve. Pathwise differentiation gives each one without bumping,
but propagated forward through the simulation it needs one tangent per
input: 80 forward rates means 80 tangents carried along every path.
Michael Giles and Paul Glasserman (2006) ran the chain rule the other
way. One adjoint vector :math:`\bar L = \partial V / \partial L`, swept
back from the payoff to time zero through the stored path, gives every
delta (and every vega) at once:

.. math:: \bar L(n) = \left(\frac{\partial L(n+1)}{\partial L(n)}\right)^{\!\top} \bar L(n+1),
          \qquad \bar L(N_{\text{mat}}) = \frac{\partial V}{\partial L(N_{\text{mat}})}.

In the LIBOR market model each transposed Jacobian product reduces to a
reverse cumulative sum, :math:`O(N)` work, so the whole gradient costs a
small multiple of one price. This is their test case: 80 quarterly
forward rates at 5% with 20% volatility, and 15 payer swaptions
expiring in 10 years on swaps of 1 to 10 years at three strikes.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from financialkit.monte_carlo import lmm_swaption_greeks

N, m, delta = 80, 40, 0.25
L0, sigma = np.full(N, 0.05), np.full(N, 0.2)
book = [(k, K) for k in (4, 8, 20, 28, 40) for K in (0.045, 0.05, 0.055)]

g = lmm_swaption_greeks(L0, sigma, delta, m, book, n_paths=10_000, rng=0)
print(f"book value {g.price:.5f} +/- {g.price_se:.5f}")

# %%
# Eighty deltas and eighty vegas from one simulation
# --------------------------------------------------
# Rates that fix before the expiry enter only through the spot numeraire,
# so their deltas are small and negative. Rates inside the shortest swap
# carry all 15 swaptions, and the delta steps down where the 1-, 2-, 5-
# and 7-year swaps end.

T = delta * np.arange(N)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
panels = ((g.delta, g.delta_se, "C0", r"$\partial V / \partial L_i(0)$"), (g.vega, g.vega_se, "C1", r"$\partial V / \partial \sigma_i$"))
for ax, (value, se, color, label) in zip(axes[:2], panels, strict=True):
    ax.bar(T, value, width=0.8 * delta, yerr=2 * se, color=color)
    ax.axvline(m * delta, color="k", ls=":", label="expiry")
    ax.set_xlabel(r"start of accrual period $T_i$ (years)")
    ax.set_ylabel(label)
    ax.legend()
assert np.all(g.delta[1:m] < 0) and np.all(g.delta[m : m + 4] > g.delta[m + 4])

# %%
# The same numbers as bump-and-reprice
# ------------------------------------
# On common random numbers a central difference in one forward rate
# reproduces the adjoint delta to the accuracy of the difference itself;
# bumping would need 160 more simulations for the whole gradient.

z = np.random.default_rng(1).standard_normal((2000, m))
base = lmm_swaption_greeks(L0, sigma, delta, m, book, z=z)
h = 1e-6
for i in (10, 45, 70):
    e = h * np.eye(N)[i]
    up, down = (lmm_swaption_greeks(L0 + s * e, sigma, delta, m, book, z=z).price for s in (1, -1))
    fd = (up - down) / (2 * h)
    print(f"L_{i}: adjoint {base.delta[i]:+.6f}, bump-and-reprice {fd:+.6f}")
    assert abs(fd - base.delta[i]) < 1e-6

# %%
# Cost against the number of rates
# --------------------------------
# Forward mode carries :math:`2N` tangents and its cost per path grows
# like :math:`N^2`; the adjoint grows like :math:`N`, as the price does.
# Both give the same deltas to rounding.

sizes = (10, 20, 40, 80)
times = {"forward": [], "adjoint": []}
for n_rates in sizes:
    mm = n_rates // 2
    bk = [(min(k, n_rates - mm), K) for k, K in book]
    res = {}
    for method in times:
        t0 = time.perf_counter()
        res[method] = lmm_swaption_greeks(np.full(n_rates, 0.05), 0.2, delta, mm, bk, n_paths=400, rng=2, method=method)
        times[method].append(time.perf_counter() - t0)
    assert np.allclose(res["forward"].delta, res["adjoint"].delta, rtol=1e-10, atol=1e-14)
ratio = np.array(times["forward"]) / np.array(times["adjoint"])
print("forward / adjoint time:", np.round(ratio, 1))
assert ratio[-1] > ratio[0]

for method, marker in (("forward", "s-"), ("adjoint", "o-")):
    axes[2].loglog(sizes, times[method], marker, label=f"{method} mode")
axes[2].loglog(sizes, times["adjoint"][-1] * (np.array(sizes) / sizes[-1]) ** 2 * ratio[-1], "k:", label=r"$\propto N^2$")
axes[2].set_xlabel("forward rates N")
axes[2].set_ylabel("seconds for all deltas and vegas")
axes[2].legend()
fig.tight_layout()
plt.show()

r"""
Koksma-Hlawka and discrepancy
=============================

Jurjen Koksma (1942) in one dimension and Edmund Hlawka (1961) in many
bounded the error of an equal-weight rule by a product of two numbers,
one about the integrand and one about the points:

.. math:: \left|\frac1N\sum_i f(x_i) - \int_{[0,1]^d} f\right| \le V_{HK}(f)\,D_N^*,

the Hardy-Krause variation times the star discrepancy, the worst
mismatch between the fraction of points in an anchored box and its
volume. It is why quasi-Monte Carlo works: low-discrepancy sequences
have :math:`D_N^* = O((\log N)^d / N)`, against :math:`N^{-1/2}` for
random points.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.monte_carlo import halton, koksma_hlawka_bound, sobol, star_discrepancy

# %%
# The local discrepancy of one box
# --------------------------------

x = halton(64, 2, scramble=True, rng=0)
a, b = 0.6, 0.45
inside = (x[:, 0] < a) & (x[:, 1] < b)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(*x[~inside].T, s=8, color="0.5")
axes[0].scatter(*x[inside].T, s=8, color="C3")
axes[0].add_patch(plt.Rectangle((0, 0), a, b, fill=False, lw=2))
axes[0].set_aspect("equal")
axes[0].set_title(f"{inside.sum()}/64 = {inside.mean():.3f} points vs area {a * b:.3f}")
print(f"star discrepancy of these 64 points: {star_discrepancy(x):.4f}")

# %%
# Discrepancy falls faster for low-discrepancy points
# ---------------------------------------------------

ns = 2 ** np.arange(4, 11)
sets = {
    "random": lambda n: np.random.default_rng(n).random((n, 2)),
    "Halton": lambda n: halton(n, 2, scramble=True, rng=n),
    "Sobol": lambda n: sobol(n, 2, rng=n),
}
disc = {name: [star_discrepancy(gen(n)) for n in ns] for name, gen in sets.items()}
assert disc["Halton"][-1] < disc["random"][-1] / 3 and disc["Sobol"][-1] < disc["random"][-1] / 3
for name, d in disc.items():
    axes[1].loglog(ns, d, "o-", label=name)
axes[1].loglog(ns, 1 / np.sqrt(ns), "k:", label=r"$N^{-1/2}$")
axes[1].loglog(ns, np.log(ns) ** 2 / ns / 4, "k--", label=r"$(\log N)^2 / N$")
axes[1].set_xlabel("N")
axes[1].set_ylabel(r"$D_N^*$")
axes[1].legend()

# %%
# The bound holds and is tight up to a constant
# ---------------------------------------------
# :math:`f(x, y) = xy` has :math:`V_{HK} = 3` (the variations of
# :math:`x`, of :math:`y` and of the mixed term) and integral 1/4.

for name, gen in sets.items():
    err = [abs(np.mean(np.prod(gen(n), axis=1)) - 0.25) for n in ns]
    bound = [koksma_hlawka_bound(gen(n), 3.0) for n in ns]
    assert np.all(np.array(err) <= bound)
    (line,) = axes[2].loglog(ns, err, "o", label=f"{name} error")
    axes[2].loglog(ns, bound, "-", color=line.get_color(), alpha=0.6)
axes[2].set_xlabel("N")
axes[2].set_title(r"errors (dots) under $V_{HK} D_N^*$ (lines)")
axes[2].legend()
fig.tight_layout()
plt.show()

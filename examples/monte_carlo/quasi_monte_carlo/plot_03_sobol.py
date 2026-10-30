r"""
Sobol sequences
===============

Ilya Sobol (1967) constructed sequences in base 2 whose leading points
form :math:`(t, m, s)`-nets: every dyadic box of volume
:math:`2^{t - m}` holds exactly :math:`2^t` of the first :math:`2^m`
points. With well-chosen direction numbers (Joe and Kuo's, which
:class:`scipy.stats.qmc.Sobol` uses) they stay balanced in thousands of
dimensions and are the workhorse of quasi-Monte Carlo in finance.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.monte_carlo import halton, sobol
from financialkit.monte_carlo.visualizers import plot_error_convergence, plot_point_set

# %%
# Every elementary box holds one point
# ------------------------------------
# The first 64 points hit each cell of an 8 x 8, a 4 x 16 and a 16 x 4 grid
# exactly once.

x = sobol(64, 2, scramble=False)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
plot_point_set(x, ax=axes[0], grid=8, title="64 Sobol points, 8 x 8 cells")
plot_point_set(x, ax=axes[1], title="the same points, 16 x 4 cells")
for k in range(1, 16):
    axes[1].axvline(k / 16, color="0.8", lw=0.5)
for k in range(1, 4):
    axes[1].axhline(k / 4, color="0.8", lw=0.5)
counts = np.histogram2d(x[:, 0], x[:, 1], bins=[16, 4], range=[[0, 1], [0, 1]])[0]
print("points per 16 x 4 cell:", np.unique(counts))
for bins in ([8, 8], [4, 16], [16, 4]):
    assert np.all(np.histogram2d(x[:, 0], x[:, 1], bins=bins, range=[[0, 1], [0, 1]])[0] == 1)

# %%
# Integration error in eight dimensions
# -------------------------------------
# A smooth product integrand with integral 1: Sobol keeps close to
# :math:`1/N` where Halton's high bases start to lag.

d = 8
f = lambda u: np.prod(1 + 0.5 * (u - 0.5), axis=1)  # noqa: E731
ns = 2 ** np.arange(6, 15)
errs = {
    "random": [abs(f(np.random.default_rng(n).random((n, d))).mean() - 1) for n in ns],
    "Halton": [abs(f(halton(n, d)).mean() - 1) for n in ns],
    "Sobol": [abs(f(sobol(n, d, rng=0)).mean() - 1) for n in ns],
}
assert errs["Sobol"][-1] < errs["Halton"][-1] < errs["random"][-1]
for name, e in errs.items():
    plot_error_convergence(ns, e, ax=axes[2], label=name, rates=(1.0,) if name == "Sobol" else ())
axes[2].set_title(f"d = {d}")
fig.tight_layout()
plt.show()

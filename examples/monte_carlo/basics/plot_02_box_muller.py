r"""
Box-Muller normal variates
==========================

Every Black-Scholes simulation needs normal random numbers, and a
computer produces uniforms. George Box and Mervin Muller (1958) gave an
exact transform: if :math:`u_1, u_2` are independent uniforms,

.. math:: z_1 = \sqrt{-2\ln u_1}\cos(2\pi u_2), \qquad z_2 = \sqrt{-2\ln u_1}\sin(2\pi u_2)

are independent standard normals. The squared radius of a standard
bivariate normal is exponential with mean 2 and its angle is uniform, so
inverting both gives the pair.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, norm

from financialkit.monte_carlo import box_muller, box_muller_normals

rng = np.random.default_rng(0)
u1, u2 = rng.random((2, 3000))
z1, z2 = box_muller(1.0 - u1, u2)

# %%
# Uniform square to normal plane
# ------------------------------
# Colouring each uniform pair by its angle :math:`2\pi u_2` shows the map:
# :math:`u_2` sets the direction and :math:`u_1` the radius.

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].scatter(u1, u2, c=u2, s=3, cmap="twilight")
axes[0].set_title("uniforms $(u_1, u_2)$")
axes[1].scatter(z1, z2, c=u2, s=3, cmap="twilight")
axes[1].set_aspect("equal")
axes[1].set_title("Box-Muller $(z_1, z_2)$")

# %%
# The output is normal
# --------------------

z = box_muller_normals(100_000, rng=1)
x = np.linspace(-4, 4, 200)
axes[2].hist(z, bins=80, density=True, alpha=0.6, label="Box-Muller")
axes[2].plot(x, norm.pdf(x), "k", label="N(0, 1)")
axes[2].legend()
pvalue = kstest(z, "norm").pvalue
axes[2].set_title(f"KS p-value {pvalue:.2f}")
corr = np.corrcoef(z1, z2)[0, 1]
print(f"mean {z.mean():+.4f}, variance {z.var():.4f}, corr(z1, z2) {corr:+.4f}")
assert pvalue > 0.01
assert abs(z.mean()) < 4 / np.sqrt(z.size) and abs(z.var() - 1) < 4 * np.sqrt(2 / z.size)
assert abs(corr) < 4 / np.sqrt(z1.size)
fig.tight_layout()
plt.show()

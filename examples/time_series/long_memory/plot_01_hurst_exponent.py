r"""
Hurst exponent
==============

To size the Aswan reservoirs, Harold Edwin Hurst (1951) studied the
*range* of cumulative departures of the Nile's annual flow from its mean:
the storage a dam would need to deliver the average flow every year.
Divided by the standard deviation, the range of independent inflows grows
like :math:`n^{1/2}`; Hurst found :math:`n^{0.73}` for the Nile and many
other records. Wet years cluster with wet years. Mandelbrot later linked
the exponent :math:`H` to fractional Brownian motion: :math:`H > 1/2`
means persistence, :math:`H < 1/2` mean reversion, and in markets a
reliable :math:`H \ne 1/2` would make returns predictable.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series import hurst_exponent, simulate_fgn
from financialkit.time_series.visualizers import plot_rescaled_range

# %%
# Persistent, independent and anti-persistent noise
# -------------------------------------------------
# Fractional Gaussian noise with three exponents, cumulated into "flows".

rng = np.random.default_rng(1951)
series = {H: simulate_fgn(4096, H, rng=rng) for H in (0.3, 0.5, 0.75)}
fig, ax = plt.subplots(figsize=(8, 3.5))
for H, x in series.items():
    ax.plot(np.cumsum(x), lw=0.8, label=f"H = {H}")
ax.set_xlabel("year")
ax.set_ylabel("cumulative departure")
ax.legend()

# %%
# Rescaled-range analysis
# -----------------------
# The slope of :math:`\log(R/S)_n` on :math:`\log n` estimates :math:`H`.
# The dashed line is Anis and Lloyd's exact expectation for independent
# data; its slope exceeds 1/2 in short windows, which is why the corrected
# estimate measures the excess over it.

fig, ax = plt.subplots()
for H, x in series.items():
    res = hurst_exponent(x)
    plot_rescaled_range(res, ax=ax, label=f"true {H}:")
    print(f"true H = {H}: raw slope {res.hurst:.2f}, corrected {res.hurst_corrected:.2f}")
ax.legend(fontsize=8)
plt.show()

# %%
# The returns of a random-walk price have :math:`H \approx 1/2` once the
# small-sample bias is removed.

returns = 0.01 * np.random.default_rng(0).standard_normal(4096)
print(f"random-walk returns: corrected H = {hurst_exponent(returns).hurst_corrected:.2f}")

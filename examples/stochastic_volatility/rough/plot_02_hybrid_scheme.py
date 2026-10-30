r"""
Bennedsen, Lunde and Pakkanen's Hybrid Scheme
=============================================

Exact Cholesky simulation of a Volterra process costs :math:`O(n^3)` to
factorise and :math:`O(n^2)` per path. Approximating the integral
:math:`\int_0^t (t - s)^{\alpha} dW_s` by a Riemann sum is cheap but fails
near :math:`s = t`, where the kernel is singular: for
:math:`H = 0.07` it loses about half the variance. Bennedsen, Lunde and
Pakkanen (2017) integrated the kernel exactly over the last
:math:`\kappa` steps, as Wiener integrals jointly Gaussian with the
Brownian increments, and evaluated it at optimal points

.. math:: b_k = \left(\frac{k^{\alpha+1} - (k-1)^{\alpha+1}}{\alpha + 1}\right)^{1/\alpha}

elsewhere. The far part is a discrete convolution, done by FFT in
:math:`O(n \log n)` per path. This example compares the variance of the
three schemes with :math:`t^{2H}`, checks that hybrid and exact rough
Bergomi prices agree, and plots paths for three Hurst exponents.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility import RoughBergomiModel, hybrid_scheme_variance, rough_bergomi_mc_price, volterra_paths

H, T, n = 0.07, 1.0, 50
t = np.linspace(0.0, T, n + 1)
hyb = hybrid_scheme_variance(H, T, n)
riem = hybrid_scheme_variance(H, T, n, "riemann")
print(f"Var at T: exact 1, hybrid {hyb[-1]:.4f}, Riemann {riem[-1]:.4f}")
assert abs(hyb[-1] - 1.0) < 0.01 and riem[-1] < 0.6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(t, t ** (2 * H), "k", label="exact t^(2H)")
ax1.plot(t, hyb, "o", ms=3, label="hybrid, kappa = 1")
ax1.plot(t, riem, "s", ms=3, label="Riemann sum")
ax1.set_xlabel("t")
ax1.set_ylabel("Var of the Volterra process")
ax1.set_title(f"H = {H}, {n} steps")
ax1.legend()

# %%
# Prices of an at-the-money call under rough Bergomi, on independent
# random numbers: hybrid and Cholesky agree within their standard errors,
# while the Riemann sum, which understates the variance of volatility,
# does not. At 50 steps both exact and hybrid sampling are fast; the
# hybrid scheme's :math:`O(n \log n)` cost pays off at the thousands of
# steps that short maturities and path-dependent payoffs need.
model = RoughBergomiModel(H=H, eta=1.9, rho=-0.9, xi0=0.235**2)
results = {}
for seed, scheme in enumerate(("cholesky", "hybrid", "riemann")):
    start = time.perf_counter()
    res = rough_bergomi_mc_price(model, 1.0, 1.0, 0.5, 0.0, n_steps=n, n_paths=30_000, scheme=scheme, rng=seed)
    results[scheme] = res
    print(f"{scheme:8s}: {res.estimate:.5f} +/- {res.std_error:.5f}  ({time.perf_counter() - start:.2f} s)")
c, h = results["cholesky"], results["hybrid"]
assert abs(c.estimate - h.estimate) < 4 * np.hypot(c.std_error, h.std_error)

# %%
# The same Brownian increments (the hybrid scheme with one seed) drive
# Volterra paths for three Hurst exponents. Small :math:`H` gives the
# rough, rapidly varying paths that the model needs.
for Hk in (0.05, 0.25, 0.5):
    tt, X, _ = volterra_paths(Hk, T, 500, 1, rng=1)
    ax2.plot(tt, X[0], lw=0.8, label=f"H = {Hk}")
ax2.set_xlabel("t")
ax2.set_ylabel("Volterra process")
ax2.set_title("Paths by the hybrid scheme")
ax2.legend()
fig.tight_layout()

plt.show()

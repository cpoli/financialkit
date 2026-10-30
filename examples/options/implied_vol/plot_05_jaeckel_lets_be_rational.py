r"""
Jäckel's "Let's be rational" implied volatility
===============================================

Every implied-volatility solver has to cope with prices that span
hundreds of orders of magnitude: a far out-of-the-money option on a
short expiry is worth :math:`10^{-80}` of its strike or less. Peter
Jäckel (2015) normalised the Black price to
:math:`b(x, s) = c / \sqrt{FK}` with :math:`x = \ln(F/K)` and total
volatility :math:`s = \sigma\sqrt{T}`, evaluated it to full relative
precision everywhere, built an initial guess from four rational cubic
branches around the inflection point :math:`s_c = \sqrt{2|x|}`, and
showed that two Householder steps then reach machine precision, with no
bracket and no tolerance to choose.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_price, implied_volatility, normalised_black_call, normalised_implied_volatility

# %%
# The four branches of the initial guess
# --------------------------------------
# For one moneyness, :math:`b` is convex below the inflection point and
# concave above it. The tangent there meets 0 at :math:`s_l` and the
# upper bound :math:`e^{x/2}` at :math:`s_u`, which splits the price
# axis into four branches.

x = -1.0
s = np.linspace(1e-3, 6, 400)
b = normalised_black_call(x, s)
s_c = np.sqrt(2 * abs(x))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(s, b, label="$b(x, s)$, $x = -1$")
axes[0].axhline(np.exp(x / 2), color="k", lw=0.5, ls="--", label="$b_{max} = e^{x/2}$")
axes[0].axvline(s_c, color="C1", lw=0.8, label=r"inflection $s_c = \sqrt{2|x|}$")
axes[0].set_xlabel("total volatility s")
axes[0].set_ylabel("normalised call price")
axes[0].legend()

# %%
# Two iterations to machine precision
# -----------------------------------
# Round trips over a wide grid of out-of-the-money options: the rational
# guess alone is within a few percent, one Householder step gains about
# eight digits, and two land on the last bit.

X, S = np.meshgrid(-np.logspace(-6, 1.3, 60), np.logspace(-2.5, 0.7, 60))
B = normalised_black_call(X, S)
keep = B > 1e-300
X, S, B = X[keep], S[keep], B[keep]
for n in range(3):
    rel = np.abs(normalised_implied_volatility(B, X, n_iter=n) - S) / S
    print(f"{n} iterations: median relative error {np.median(rel):.1e}, max {rel.max():.1e}")
assert np.median(rel) < 4 * np.finfo(float).eps and rel.max() < 1e-12

# %%
# Against the bracketing solvers
# ------------------------------
# On a 100 x 100 surface of out-of-the-money options, Jäckel agrees with
# the safeguarded Newton solver to Newton's tolerance, with no tolerance
# of its own. It is just as accurate on a price of order 1e-75.

K = np.linspace(70.0, 140.0, 100)[:, None]
T = np.linspace(0.05, 3.0, 100)[None, :]
F = 100.0 * np.exp(0.02 * T)
prices = black_scholes_price(100.0, K, T, 0.02, 0.2 + 0.1 * np.log(K / 100.0) ** 2, option_type="call")
otm = np.where(K >= F, prices, prices - (100.0 - K * np.exp(-0.02 * T)))  # out-of-the-money puts below the forward
kind = np.where(K >= F, "call", "put")
timings = {}
for method in ("newton", "jaeckel"):
    t0 = time.perf_counter()
    vols = {
        k: implied_volatility(
            otm[kind == k], 100.0, np.broadcast_to(K, otm.shape)[kind == k], np.broadcast_to(T, otm.shape)[kind == k], 0.02, 0.0, k, method=method
        )
        for k in ("call", "put")
    }
    timings[method] = (time.perf_counter() - t0, vols)
diff = max(np.max(np.abs(timings["newton"][1][k] - timings["jaeckel"][1][k])) for k in ("call", "put"))
print(f"Newton {timings['newton'][0] * 1e3:.1f} ms, Jäckel {timings['jaeckel'][0] * 1e3:.1f} ms, max difference {diff:.1e}")
assert diff < 1e-9

tiny = black_scholes_price(100.0, 120.0, 0.25, 0.0, 0.02)
print(f"a price of {tiny:.2e} has implied volatility {implied_volatility(tiny, 100.0, 120.0, 0.25, 0.0, method='jaeckel'):.15f}")
assert abs(implied_volatility(tiny, 100.0, 120.0, 0.25, 0.0, method="jaeckel") - 0.02) < 1e-14

Xg, Sg = np.meshgrid(-np.logspace(-4, 1.3, 120), np.logspace(-2.5, 0.7, 120))
Bg = normalised_black_call(Xg, Sg)
err = np.where(Bg > 1e-300, np.abs(normalised_implied_volatility(np.maximum(Bg, 1e-300), Xg) - Sg) / Sg, np.nan)
mesh = axes[1].pcolormesh(np.abs(Xg), Sg, np.log10(np.maximum(err, 1e-17)), shading="auto", vmin=-17, vmax=-12)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("|x| = |ln(F/K)|")
axes[1].set_ylabel("total volatility s")
axes[1].set_title("log10 relative error after two iterations")
fig.colorbar(mesh, ax=axes[1])
fig.tight_layout()

plt.show()

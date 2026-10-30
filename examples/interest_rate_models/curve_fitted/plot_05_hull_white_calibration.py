r"""
Hull-White calibrated to caps and swaptions
===========================================

Hull-White fits today's curve by construction, which leaves the mean
reversion :math:`a` and volatility :math:`\sigma` to be chosen. In
practice they are fitted to the options the model will hedge: caps,
quoted as flat Black volatilities, and swaptions, quoted as Black
volatilities of the forward swap rate. Each quote becomes a price with
Black's formula, and :func:`scipy.optimize.least_squares` matches the
model's prices (caplets as puts on zero-coupon bonds, swaptions by
Jamshidian's decomposition), each error divided by its Black vega so the
fit is in volatility points.

:math:`\sigma` sets the level of the volatilities and :math:`a` how fast
they fall with expiry, so a cap strip pins down both.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, hull_white_calibrate, model_implied_vols

curve = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
truth = HullWhiteModel(curve, 0.08, 0.009)

# %%
# Round trip: quotes from a known model are fitted back exactly
# --------------------------------------------------------------

cap_maturities = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0])
cap_vols = model_implied_vols(truth, cap_maturities)
exact = hull_white_calibrate(curve, cap_vols, cap_maturities)
print(f"true a = 0.08, sigma = 0.009; fitted a = {exact.params['a']:.6f}, sigma = {exact.params['sigma']:.6f}")

# %%
# Noisy cap quotes
# ----------------
# Market quotes never sit exactly on one model's curve. Adding up to half
# a vol point of noise, the fit lands near the true parameters and the
# residuals are the noise itself.

rng = np.random.default_rng(0)
noisy = cap_vols + rng.uniform(-0.005, 0.005, cap_vols.size)
caps = hull_white_calibrate(curve, noisy, cap_maturities)
print(f"noisy caps: a = {caps.params['a']:.4f}, sigma = {caps.params['sigma']:.5f}, RMSE = {100 * caps.rmse:.3f} vol points")

# %%
# The same model fitted to a swaption column
# ------------------------------------------
# Five-year swaps, expiring in 1 to 10 years: the fitted parameters agree
# with the cap fit because the quotes came from the same model.

expiries = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
swaption_vols = model_implied_vols(truth, expiries, "swaption", 5.0)
swaptions = hull_white_calibrate(curve, swaption_vols, expiries, "swaption", 5.0)
print(f"swaptions:  a = {swaptions.params['a']:.6f}, sigma = {swaptions.params['sigma']:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(cap_maturities, 100 * noisy, "o", label="market (noisy)")
axes[0].plot(cap_maturities, 100 * caps.model_vols, "-", label=f"Hull-White fit (a = {caps.params['a']:.3f})")
axes[0].plot(cap_maturities, 100 * cap_vols, ":", color="gray", label="true model")
axes[0].set_xlabel("cap maturity (years)")
axes[0].set_ylabel("flat Black volatility (%)")
axes[0].set_title("ATM caps")
axes[0].legend()
axes[1].plot(expiries, 100 * swaption_vols, "o", label="market")
axes[1].plot(expiries, 100 * swaptions.model_vols, "-", label="Hull-White fit")
axes[1].set_xlabel("swaption expiry (years)")
axes[1].set_ylabel("Black volatility (%)")
axes[1].set_title("ATM swaptions into 5-year swaps")
axes[1].legend()
fig.tight_layout()

assert abs(exact.params["a"] - 0.08) < 1e-6 and abs(exact.params["sigma"] - 0.009) < 1e-8  # exact quotes, exact parameters
assert exact.rmse < 1e-8 and swaptions.rmse < 1e-8
assert abs(swaptions.params["a"] - 0.08) < 1e-6 and abs(swaptions.params["sigma"] - 0.009) < 1e-8
assert caps.rmse < 0.005  # the fit is no worse than the half-vol-point noise
assert abs(caps.params["sigma"] - 0.009) < 5e-4 and abs(caps.params["a"] - 0.08) < 0.02  # and stays near the truth

plt.show()

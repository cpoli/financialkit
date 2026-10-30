r"""
Piterbarg: OIS discounting of collateralized trades, with separate forecasting curves
=====================================================================================

When the LIBOR-OIS spread blew out in 2008, it showed that one LIBOR curve
could not both project LIBOR coupons and discount every cash flow.
Piterbarg (2010) showed that a trade under a collateral agreement grows
at the collateral rate, so its cash flows are discounted on the OIS
curve. LIBOR coupons come from a separate forecasting curve, which is
solved so that every par swap, discounted on OIS, is worth zero.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import MultiCurve, bootstrap_multi_curve, bootstrap_par_curve, swap_rate

# %%
# Two curves from two sets of quotes
# ----------------------------------
#
# Par OIS rates pay annually against compounded overnight. Par swaps pay
# a semiannual fixed rate against 3-month LIBOR, at a spread above OIS
# that narrows with maturity.

T = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])
ois_rates = np.array([3.00, 3.10, 3.20, 3.40, 3.50, 3.60, 3.70, 3.75]) / 100
swap_rates = ois_rates + np.array([30, 28, 26, 24, 22, 20, 19, 18]) / 1e4
curves = bootstrap_multi_curve(T, ois_rates, T, swap_rates, fixed_freq=2, float_freq=4)
assert np.allclose(swap_rate(curves.discount, T, freq=1), ois_rates, atol=1e-10)
assert np.allclose(curves.swap_rate(T), swap_rates, atol=1e-10)

# %%
# Before 2008 the swaps alone gave one curve, used for both jobs.

libor = bootstrap_par_curve(T, swap_rates, freq=2)
single = MultiCurve(libor, libor)
assert np.allclose(single.swap_rate(T), swap_rates, atol=1e-10)

t = np.arange(0.0, 19.76, 0.25)
ois_forwards = MultiCurve(curves.discount, curves.discount).forecast_rate(t, t + 0.25)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(t, 100 * curves.forecast_rate(t, t + 0.25), label="3M LIBOR forwards (dual curve)")
axes[0].plot(t, 100 * single.forecast_rate(t, t + 0.25), "--", label="3M LIBOR forwards (single curve)")
axes[0].plot(t, 100 * ois_forwards, label="3M OIS forwards")
axes[0].set_xlabel("start (years)")
axes[0].set_ylabel("forward rate (%)")
axes[0].legend(fontsize=8)
assert np.all(curves.forecast_rate(t, t + 0.25) > ois_forwards)

# %%
# Par swaps agree, off-market swaps do not
# ----------------------------------------
#
# Both curve sets reprice the par swaps they were built from. A swap
# struck years ago at another rate does not have a value of zero, and
# that value depends on the discount curve. Receiving a fixed rate above
# the market rate is worth more when the annuity is discounted at the
# lower OIS rates.

strikes = np.linspace(0.02, 0.06, 41)
diff = np.array([curves.swap_value(1e8, K, 10.0, pay_fixed=False).value - single.swap_value(1e8, K, 10.0, pay_fixed=False).value for K in strikes])
K10 = curves.swap_rate(10.0)
v_ois, v_libor = (c.swap_value(1e8, 0.05, 10.0, pay_fixed=False).value for c in (curves, single))
print(f"10y receiver struck at 5% on $100m: {v_ois:,.0f} discounted on OIS, {v_libor:,.0f} on LIBOR")
assert abs(np.interp(K10, strikes, diff)) < 2e3 and diff[-1] > 0 > diff[0]
axes[1].plot(100 * strikes, diff / 1e3)
axes[1].axhline(0.0, color="0.6", lw=0.8)
axes[1].axvline(100 * K10, color="0.6", lw=0.8, ls=":")
axes[1].set_xlabel("fixed rate received (%)")
axes[1].set_ylabel("OIS less LIBOR discounting ($k per $100m)")
axes[1].set_title("10-year receiver swap")
plt.tight_layout()
plt.show()

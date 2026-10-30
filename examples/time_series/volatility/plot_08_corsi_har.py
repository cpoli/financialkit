r"""
Corsi's HAR model of realized volatility
========================================

Realized variance is persistent far beyond what a short autoregression
can capture: its autocorrelations decay slowly, almost hyperbolically,
the signature of long memory. Fulvio Corsi (2009) obtained it with a
regression on three averages of past realized variance, over the last
day, week and month. The idea comes from the heterogeneous market
hypothesis of Müller et al. (1997): intraday traders, swing traders and
long-term investors each react to volatility at their own horizon. The
HAR model has four coefficients, is fitted by least squares, and became
the benchmark for volatility forecasting.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series import autocorrelation, fit_arma, fit_har, mincer_zarnowitz, realized_variance
from financialkit.time_series.utils import simulate_intraday_returns

# %%
# A volatility cascade
# --------------------
# Log variance is the sum of three AR(1) components with half-lives of
# about a day, a week and three months. Each day's variance is spread over
# 78 five-minute returns, and summing their squares gives realized variance.

rng = np.random.default_rng(2009)
n = 3000
phis, sds = (0.5, 0.87, 0.992), (0.35, 0.2, 0.06)
logv = np.log(1e-4) + sum(
    np.convolve(sd * rng.standard_normal(n + 500), phi ** np.arange(500), mode="full")[500 : n + 500] for phi, sd in zip(phis, sds, strict=True)
)
daily_var = np.exp(logv)
rv = realized_variance(simulate_intraday_returns(daily_var, 78, rng=rng))

lags = np.arange(1, 101)
acf = autocorrelation(rv, 100)[1:]

# %%
# HAR against an AR(1), in and out of sample
# ------------------------------------------
# Both models are fitted on the first 2000 days and forecast the next day's
# realized variance over the last 1000.

split = 2000
har = fit_har(rv[:split])
ar1 = fit_arma(rv[:split], 1)
print(f"HAR: const {har.coef[0]:.2e}, day {har.coef[1]:.3f}, week {har.coef[2]:.3f}, month {har.coef[3]:.3f}")
print(f"in-sample R^2: HAR {har.ols.r_squared:.3f}")
assert np.all(har.coef[1:] > 0)

f_har = har.predict(rv)[split:]
f_ar = ar1.mean + ar1.ar[0] * (rv[split - 1 : -1] - ar1.mean)
target = rv[split:]
r2 = {name: mincer_zarnowitz(target, f).r_squared for name, f in (("HAR", f_har), ("AR(1)", f_ar))}
mse = {name: np.mean((target - f) ** 2) for name, f in (("HAR", f_har), ("AR(1)", f_ar))}
for name in r2:
    print(f"out of sample {name:5s}: Mincer-Zarnowitz R^2 = {r2[name]:.3f}, MSE = {1e8 * mse[name]:.3f}e-8")
assert r2["HAR"] > r2["AR(1)"] and mse["HAR"] < mse["AR(1)"]

# %%
# The HAR's implied memory
# ------------------------
# The three averages make an AR(22) with step-shaped weights; its
# autocorrelations, computed by simulating the fitted model, decay far more
# slowly than the AR(1)'s geometric :math:`\phi^k`.

w = np.zeros(22)
w[0] += har.coef[1]
w[:5] += har.coef[2] / 5
w[:22] += har.coef[3] / 22
noise = rng.standard_normal(20_000 + 22)
sim = np.zeros_like(noise)
for t in range(22, sim.size):
    sim[t] = w @ sim[t - 22 : t][::-1] + noise[t]
har_acf = autocorrelation(sim[22:], 100)[1:]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].bar(np.arange(1, 23), w)
axes[0].set_title("HAR as an AR(22): lag weights")
axes[0].set_xlabel("lag (days)")
axes[1].semilogx(lags, acf, "o", ms=3, color="0.4", label="realized variance")
axes[1].semilogx(lags, har_acf, "C3", label="fitted HAR")
axes[1].semilogx(lags, ar1.ar[0] ** lags, "C0--", label="fitted AR(1)")
axes[1].set_title("autocorrelation")
axes[1].set_xlabel("lag (days)")
axes[1].legend(fontsize=8)
fig.tight_layout()
assert har_acf[49] > 5 * ar1.ar[0] ** 50
plt.show()

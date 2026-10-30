r"""
KPSS stationarity test
======================

Unit-root tests take the random walk as the null hypothesis and are weak
against persistent but stationary alternatives, so "cannot reject a unit
root" is a common and uninformative verdict. Kwiatkowski, Phillips,
Schmidt and Shin (1992) reversed the roles. They write a series as a
deterministic part, a random walk and stationary noise, and test whether
the random walk's variance is zero, using the partial sums of the
residuals. Large partial sums reject stationarity. Used together, the
Dickey-Fuller and KPSS tests can say "stationary", "unit root" or
"the data cannot tell".
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series import adf_test, kpss_test, simulate_arma

# %%
# Four series, two tests
# ----------------------

rng = np.random.default_rng(1992)
n = 500
series = {
    "white noise": rng.standard_normal(n),
    "random walk": np.cumsum(rng.standard_normal(n)),
    "AR(1), phi = 0.97": simulate_arma(ar=[0.97], n=n, rng=rng),
    "trend + noise": 0.02 * np.arange(n) + rng.standard_normal(n),
}
print(f"{'series':18s} {'ADF':>6s} {'unit root?':>11s} {'KPSS':>6s} {'stationary?':>12s}")
verdict = {}
for name, x in series.items():
    reg = "ct" if name.startswith("trend") else "c"
    adf = adf_test(x, regression=reg)
    kp = kpss_test(x, regression=reg)
    verdict[name] = (adf.rejects(), kp.rejects())
    print(f"{name:18s} {adf.statistic:6.2f} {'rejected' if adf.rejects() else 'kept':>11s} {kp.statistic:6.3f} {'rejected' if kp.rejects() else 'kept':>12s}")
assert verdict["white noise"] == (True, False)
assert verdict["random walk"] == (False, True)
assert verdict["trend + noise"] == (True, False)

# %%
# The partial-sum statistic
# -------------------------
# For a stationary series the cumulated residuals :math:`S_t` stay of order
# :math:`\sqrt T`; for a random walk they grow like :math:`T^{3/2}`, so
# :math:`\eta = T^{-2}\sum S_t^2/\hat\sigma^2_\infty` explodes.

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for name in ("white noise", "random walk"):
    x = series[name]
    axes[0].plot(np.cumsum(x - x.mean()), lw=0.8, label=name)
axes[0].set_title("partial sums $S_t$ of the demeaned series")
axes[0].set_xlabel("t")
axes[0].legend()

# %%
# Size and power by simulation
# ----------------------------
# Rejection rates at 5% for AR(1) series of increasing persistence (and a
# random walk at :math:`\phi = 1`). Stationarity is rejected more often as
# :math:`\phi` approaches one: in samples of 200 a very persistent AR(1) is
# hard to tell from a unit root.

phis = [0.0, 0.5, 0.8, 0.95, 1.0]
rates = []
for phi in phis:
    hits = [kpss_test(simulate_arma(ar=[phi], n=200, burn=0 if phi == 1 else 500, rng=rng)).rejects() for _ in range(200)]
    rates.append(np.mean(hits))
axes[1].plot(phis, rates, "o-")
axes[1].axhline(0.05, color="0.5", ls="--", label="nominal 5%")
axes[1].set_xlabel("AR coefficient $\\phi$")
axes[1].set_ylabel("KPSS rejection rate")
axes[1].legend()
fig.tight_layout()
print("rejection rates:", dict(zip(phis, rates, strict=True)))
assert rates[0] < 0.12 and rates[-1] > 0.6 and rates == sorted(rates)
plt.show()

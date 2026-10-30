r"""
Bayer, Friz and Gatheral's Rough Bergomi
========================================

Classical stochastic volatility models such as Heston produce an
at-the-money skew that flattens to a constant as the maturity shrinks,
while in equity markets it keeps growing roughly like
:math:`\tau^{-0.4}`. Bayer, Friz and Gatheral (2016) drove the variance
by a Volterra process with a singular power kernel,

.. math::

   v_t = \xi_0 \exp\left(\eta \sqrt{2H}\int_0^t (t - s)^{H - 1/2} dW_s - \tfrac12\eta^2 t^{2H}\right),

and showed that with :math:`H \approx 0.07` three parameters fit the
whole S&P 500 surface, and the skew explodes like :math:`\tau^{H - 1/2}`.
Here the model is simulated with their exact Cholesky method, using
their parameters :math:`H = 0.07`, :math:`\eta = 1.9`,
:math:`\rho = -0.9` and :math:`\xi_0 = 0.235^2`. The resulting skew term
structure is compared with Heston's.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import implied_volatility
from financialkit.stochastic_volatility import HestonModel, RoughBergomiModel, rough_bergomi_paths

S, r = 1.0, 0.0
model = RoughBergomiModel(H=0.07, eta=1.9, rho=-0.9, xi0=0.235**2)
paths = rough_bergomi_paths(model, S, 1.0, r, n_steps=128, n_paths=30_000, scheme="cholesky", rng=2016)


def smile(ST, tau, k):
    """Implied vols of out-of-the-money options on simulated terminal prices."""
    K = S * np.exp(k)
    otm = k >= 0
    out = np.empty(k.shape)
    calls = np.maximum(ST[:, None] - K[otm], 0.0).mean(axis=0)
    puts = np.maximum(K[~otm] - ST[:, None], 0.0).mean(axis=0)
    out[otm] = implied_volatility(calls, S, K[otm], tau, r, 0.0, "call")
    out[~otm] = implied_volatility(puts, S, K[~otm], tau, r, 0.0, "put")
    return out


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
z = np.linspace(-0.4, 0.2, 13)
for i in (16, 64, 128):
    tau = paths.times[i]
    ax1.plot(z, smile(paths.spot[:, i], tau, z * np.sqrt(tau)), label=f"T = {tau:.3g}")
ax1.set_xlabel("log-moneyness / sqrt(T)")
ax1.set_ylabel("implied volatility")
ax1.set_title("Rough Bergomi smiles")
ax1.legend()

# %%
# The at-the-money skew :math:`|\partial\sigma_{BS}/\partial k|` at each
# maturity, by a central difference on the same paths. On a log-log
# scale rough Bergomi's skew is a straight line with slope near
# :math:`H - 1/2 = -0.43`. Heston, fitted to have a similar level, has a
# skew that flattens at the short end.
heston = HestonModel(v0=0.235**2, kappa=1.5, theta=0.235**2, xi=1.0, rho=-0.9)
taus, rb_skew, h_skew = [], [], []
for i in (8, 16, 32, 64, 128):
    tau = paths.times[i]
    h = 0.1 * np.sqrt(tau)
    v = smile(paths.spot[:, i], tau, np.array([-h, h]))
    hv = heston.implied_vol(S, S * np.exp([-h, h]), tau, r)
    taus.append(tau)
    rb_skew.append((v[0] - v[1]) / (2 * h))
    h_skew.append((hv[0] - hv[1]) / (2 * h))
slope = np.polyfit(np.log(taus), np.log(rb_skew), 1)[0]
short = np.polyfit(np.log(taus[:3]), np.log(rb_skew[:3]), 1)[0]
h_short = np.polyfit(np.log(taus[:3]), np.log(h_skew[:3]), 1)[0]
print(f"skew ~ tau^{slope:.2f} under rough Bergomi (H - 1/2 = {model.H - 0.5:.2f})")
print(f"slope below 3 months: rough Bergomi {short:.2f}, Heston {h_short:.2f}")
assert abs(slope - (model.H - 0.5)) < 0.1 and short < -0.3 and h_short > -0.15
ax2.loglog(taus, rb_skew, "o-", label=f"rough Bergomi, slope {slope:.2f}")
ax2.loglog(taus, h_skew, "s-", label="Heston")
ax2.set_xlabel("maturity")
ax2.set_ylabel("ATM skew")
ax2.set_title("The skew term structure")
ax2.legend()
fig.tight_layout()

plt.show()

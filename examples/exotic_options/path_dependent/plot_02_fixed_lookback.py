r"""
Conze-Viswanathan fixed lookbacks
=================================

Antoine Conze and Viswanathan (1991) priced the fixed-strike lookback:
a call on the maximum, :math:`(\max_t S_t - K)^+`, and a put on the
minimum, :math:`(K - \min_t S_t)^+`. Once the running maximum has passed
the strike, the call has locked in :math:`\max - K` and what remains is
a call on any further rise; that is how a seasoned option is priced,
with the maximum observed so far as an input.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import fixed_lookback_price, lookback_option_mc
from financialkit.options import black_scholes_price

S, T, r, sigma = 100.0, 1.0, 0.05, 0.25

# %%
# New options across strikes
# --------------------------

K = np.linspace(70.0, 140.0, 50)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K, fixed_lookback_price(S, K, T, r, sigma), label="call on the maximum")
axes[0].plot(K, fixed_lookback_price(S, K, T, r, sigma, option_type="put"), label="put on the minimum")
axes[0].plot(K, black_scholes_price(S, K, T, r, sigma), "k--", label="vanilla call")
axes[0].plot(K, black_scholes_price(S, K, T, r, sigma, option_type="put"), "k:", label="vanilla put")
for k in (80.0, 100.0, 120.0):
    for option_type in ("call", "put"):
        res = lookback_option_mc(S, T, r, sigma, option_type=option_type, K=k, n_steps=10, n_paths=10_000, rng=int(k))
        axes[0].errorbar(k, res.estimate, yerr=2 * res.std_error, fmt="k.", capsize=3)
axes[0].set_xlabel("strike K")
axes[0].set_ylabel("value")
axes[0].legend()

# %%
# A seasoned call: the running maximum so far
# -------------------------------------------
# Halfway through, with the spot at 100 and a strike of 105, the value
# rises with the maximum already reached, linearly once it passes the
# strike.

Smax = np.linspace(100.0, 140.0, 100)
axes[1].plot(Smax, fixed_lookback_price(S, 105.0, 0.5, r, sigma, S_extreme=Smax), label="seasoned call on the maximum")
axes[1].plot(Smax, np.exp(-r * 0.5) * np.maximum(Smax - 105.0, 0.0), "--", label=r"locked in $e^{-r\tau}(S_{\max} - K)^+$")
axes[1].axvline(105.0, color="0.5", lw=1)
axes[1].set_xlabel(r"running maximum $S_{\max}$")
axes[1].set_ylabel("value")
axes[1].legend()
fig.tight_layout()
plt.show()

:orphan:

From Returns to Risk
====================

A risk number is only as good as the model of returns behind it. This
tutorial takes one return series through two domains: diagnosing it with
:mod:`financialkit.time_series` (fat tails, volatility clustering, a
GARCH fit) and turning it into risk measures with :mod:`financialkit.risk`
(value at risk, expected shortfall, backtests and extreme value theory).
The returns are synthetic, simulated from a GARCH(1,1) model, so the
right answers are known.

1. The stylized facts
---------------------

Twelve years of daily returns from a GARCH(1,1) process whose long-run
volatility is 16% a year:

.. code-block:: python

   import numpy as np
   import financialkit as fk

   true = fk.time_series.GARCH(omega=1e-6, alpha=0.12, beta=0.87, mu=0.0003)
   r = true.simulate(3000, rng=0).returns

   stats = fk.time_series.return_statistics(r)
   print(f"excess kurtosis {stats.excess_kurtosis:.2f}, Jarque-Bera p-value {stats.jarque_bera_pvalue:.1e}")
   # excess kurtosis 2.89, Jarque-Bera p-value 8.2e-228
   print("autocorrelation of returns, lags 1-3:        ", np.round(fk.time_series.autocorrelation(r, 3)[1:], 3))
   # autocorrelation of returns, lags 1-3:         [-0.06   0.014  0.021]
   print("autocorrelation of squared returns, lags 1-3:", np.round(fk.time_series.autocorrelation(r**2, 3)[1:], 3))
   # autocorrelation of squared returns, lags 1-3: [0.33  0.322 0.342]

These are the stylized facts of real markets. The returns are not normal:
their tails are fat. They are nearly uncorrelated, so past returns barely
predict future ones, but their *squares* are strongly correlated: calm
days follow calm days, and turbulent days cluster.

2. Fitting GARCH
----------------

Bollerslev's GARCH(1,1) models exactly that clustering: tomorrow's
variance is a weighted sum of a long-run level, today's squared return
and today's variance. Fitting it by maximum likelihood recovers the
parameters:

.. code-block:: python

   fit = fk.time_series.GARCH.fit(r)
   m = fit.model
   print(f"alpha {m.alpha:.3f} (true 0.12), beta {m.beta:.3f} (true 0.87)")
   # alpha 0.130 (true 0.12), beta 0.865 (true 0.87)
   print(f"persistence alpha + beta = {m.alpha + m.beta:.3f} (true 0.99)")
   # persistence alpha + beta = 0.995 (true 0.99)
   print(f"long-run volatility {100 * np.sqrt(252 * m.unconditional_variance):.1f}% a year (true 15.9%)")
   # long-run volatility 20.9% a year (true 15.9%)
   sigma = np.sqrt(fit.variance)
   print(f"daily volatility ranged from {100 * sigma.min():.2f}% to {100 * sigma.max():.2f}%")
   # daily volatility ranged from 0.30% to 3.23%

Alpha and beta are recovered to within about one standard error. The
long-run volatility is the weak spot: it is
:math:`\omega / (1 - \alpha - \beta)`, so with persistence this close to
one, a small error in alpha + beta moves it a lot. Volatility shocks die
out over weeks rather than days.

Is the model adequate? If it has captured the clustering, the
standardized residuals :math:`r_t / \sigma_t` should be white noise, in
their levels and in their squares. The Ljung-Box test agrees:

.. code-block:: python

   z = r / sigma
   print(f"Ljung-Box p-values: residuals {fk.time_series.ljung_box(z, 10).pvalue:.2f}, squared residuals {fk.time_series.ljung_box(z**2, 10).pvalue:.2f}")
   # Ljung-Box p-values: residuals 0.47, squared residuals 0.17

3. Two ways to compute value at risk
------------------------------------

The 99% one-day VaR is the loss exceeded on one day in a hundred. The
simplest method, historical simulation, takes the 1% quantile of the last
250 days. The GARCH method scales a normal quantile by each day's
conditional volatility:

.. code-block:: python

   window = 250
   test = r[window:]
   historical = fk.risk.rolling_historical_var(r, window, 0.99)
   garch = fk.risk.parametric_var(m.mu, sigma[window:], 0.99)
   print(f"average VaR: historical {100 * historical.mean():.2f}%, GARCH {100 * garch.mean():.2f}%")
   # average VaR: historical 2.32%, GARCH 2.12%

On average the two methods are close. The difference is in *when* each
one is high.

4. Backtesting: counting and timing the exceptions
--------------------------------------------------

Over 2,750 days a correct 99% VaR should be exceeded about 28 times.
Kupiec's test checks the count, and Christoffersen's adds a check that
exceptions do not cluster:

.. code-block:: python

   for name, var in (("historical", historical), ("GARCH", garch)):
       hits = fk.risk.var_exceptions(test, var)
       kupiec = fk.risk.kupiec_test(hits, 0.99)
       christoffersen = fk.risk.christoffersen_test(hits, 0.99)
       print(
           f"{name:10s}: {hits.sum()} exceptions (expected {kupiec.expected:.1f}), "
           f"Kupiec p = {kupiec.p_value:.3f}, Christoffersen p = {christoffersen.p_value:.3f}"
       )
   # historical: 44 exceptions (expected 27.5), Kupiec p = 0.004, Christoffersen p = 0.014
   # GARCH     : 21 exceptions (expected 27.5), Kupiec p = 0.194, Christoffersen p = 0.366

Historical VaR fails both tests. It reacts to a volatility burst only
after a year-long window has filled with bad days, so its exceptions
arrive too often and in clusters, exactly when risk matters. The GARCH
VaR follows volatility day by day and passes. (The GARCH parameters were
estimated on the whole sample, a small look-ahead that a production
backtest would remove by refitting on a rolling window.)

5. Beyond VaR: the tail itself
------------------------------

VaR says nothing about how bad the bad days are. Expected shortfall, the
average loss beyond the VaR, does, and unlike VaR it is coherent: it
never penalises diversification. For quantiles too extreme to estimate
from the data directly, extreme value theory fits a generalized Pareto
distribution to the losses beyond a high threshold:

.. code-block:: python

   hist = fk.risk.historical_var(r, 0.99)
   print(f"99% VaR {100 * hist.var:.2f}%, expected shortfall {100 * hist.es:.2f}%")
   # 99% VaR 2.75%, expected shortfall 3.40%

   pot = fk.risk.fit_pot(r, threshold=np.quantile(-r, 0.95))
   print(f"tail shape xi = {pot.xi:.3f} from {pot.n_exceed} exceedances")
   # tail shape xi = 0.035 from 150 exceedances
   print(f"99.9% VaR: peaks over threshold {100 * fk.risk.pot_var(pot, 0.999):.2f}%")
   # 99.9% VaR: peaks over threshold 4.41%
   print(f"           normal distribution  {100 * fk.risk.parametric_var(r.mean(), r.std(), 0.999):.2f}%")
   #            normal distribution  3.00%

The normal distribution, fitted to the same mean and volatility,
understates the one-in-a-thousand-day loss, which is the fat-tails lesson
of the first section restated as a risk number.

See Also
--------

- :doc:`/api/time_series`
- :doc:`/api/risk`
- :doc:`/history/time_series_breakthroughs`
- :doc:`/history/risk_breakthroughs`

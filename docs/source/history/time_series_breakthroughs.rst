Breakthroughs in Financial Time Series
======================================


.. include:: /_generated/nav/time_series.rst

.. epigraph::

   "The series looks like a wandering one, almost as if once a week the
   Demon of Chance drew a random number from a symmetrical population of
   fixed dispersion and added it to the current price to determine the
   next week's price."
   -- Maurice Kendall, 1953

A price series is a single realization of a random process that can never
be rerun, so everything known about it has to be inferred from its own
past. Financial time-series analysis began by borrowing the
autoregressions that statisticians had built for sunspots and rivers,
and then found that prices broke their assumptions one by one.
Prices wander like random walks, so their changes, not their levels, are
the object of study. The changes are nearly uncorrelated, which is the
efficient-markets hypothesis in statistical form, and much of the history
is a sequence of ever sharper tests of that claim. Yet the changes are not
independent: their size clusters, their tails are fat, and volatility
responds more to falls than to rises. Many series move together, through
shared trends, feedback and correlations that change over time, and some
switch between regimes. Two Nobel prizes, Engle's and
Granger's in 2003, went to modelling these features. This chronology
traces the ideas behind :mod:`financialkit.time_series`.

.. contents:: Timeline
   :local:
   :depth: 1

1927 -- Yule's Autoregression
-----------------------------

The standard tool for periodic data was Schuster's periodogram, which
treats a series as exact sine waves plus measurement error. Udny Yule
found it unconvincing for sunspot numbers, whose cycles vary in length
and height. He proposed instead a pendulum "bombarded by peas": a damped
oscillation that random shocks keep alive. For Wolfer's sunspot numbers
he fitted :math:`x_t = 1.34\,x_{t-1} - 0.65\,x_{t-2} + \varepsilon_t`,
whose complex characteristic roots give a pseudo-cycle of about ten
and a half years. The coefficients solve the *Yule-Walker equations*, a Toeplitz
system in the sample autocorrelations. With this paper the
autoregressive model, the backbone of time-series analysis, was born.

.. math::

   \rho_k = \sum_{i=1}^{p}\phi_i\,\rho_{k-i}, \quad k = 1, \dots, p

*Implementation:* :func:`financialkit.time_series.systems.linear_models.yule_walker`
(via :func:`scipy.linalg.solve_toeplitz`), with
:func:`~financialkit.time_series.systems.linear_models.ar_cycle_period`
for the pseudo-period and
:func:`~financialkit.time_series.systems.linear_models.simulate_arma`.

*References:* G. U. Yule, "On a Method of Investigating Periodicities in
Disturbed Series, with Special Reference to Wolfer's Sunspot Numbers,"
*Philosophical Transactions of the Royal Society A* 226 (1927) 267-298.

.. minigallery:: ../../examples/time_series/linear_models/plot_01_yule_autoregression.py

1951 -- Hurst Exponent
----------------------

Harold Edwin Hurst spent decades measuring the Nile to size the
reservoirs above Aswan. The storage a dam needs to release the average
flow every year is the range :math:`R` of cumulative departures from the
mean. Divided by the standard deviation :math:`S`, this rescaled range
grows like :math:`n^{1/2}` for independent inflows, but Hurst found
:math:`n^{0.73}` for the Nile and for hundreds of other natural records.
Wet years cluster with wet years over long horizons. Mandelbrot later
explained the exponent :math:`H` with fractional Brownian motion and
brought it to finance, where :math:`H \ne 1/2` would mean long-range
predictability. Anis and Lloyd's (1976) exact expectation of :math:`R/S`
for independent data corrects the estimate's small-sample bias.

.. math::

   E\Bigl[\frac{R}{S}\Bigr]_n \propto n^{H}

*Implementation:* :func:`financialkit.time_series.systems.long_memory.hurst_exponent`
returns a :class:`~financialkit.time_series.core.base.HurstResult` with
the raw and Anis-Lloyd-corrected exponents;
:func:`~financialkit.time_series.systems.long_memory.simulate_fgn`
generates exact fractional Gaussian noise by circulant embedding
(:mod:`scipy.fft`).

*References:* H. E. Hurst, "Long-Term Storage Capacity of Reservoirs,"
*Transactions of the American Society of Civil Engineers* 116 (1951)
770-799; A. A. Anis and E. H. Lloyd, "The Expected Value of the Adjusted
Rescaled Hurst Range of Independent Normal Summands," *Biometrika* 63(1)
(1976) 111-116; B. B. Mandelbrot and J. W. Van Ness, "Fractional Brownian
Motions, Fractional Noises and Applications," *SIAM Review* 10(4) (1968)
422-437.

.. minigallery:: ../../examples/time_series/long_memory/plot_01_hurst_exponent.py

1953 -- Kendall: the Random Walk of Prices
------------------------------------------

Maurice Kendall went looking for cycles in 22 British price series:
wheat, cotton and industrial share indices, sampled weekly. Instead he
found that the serial correlations of the price *changes* were close to
zero at every lag, while the levels were almost perfectly correlated. He
concluded that prices wander like a random walk. The finding surprised
economists who had expected cycles, but it established the empirical fact that efficient-markets theory would
explain a decade later: in a well-functioning market, past price changes
do not predict future ones.

.. math::

   \hat\rho_k = \frac{\sum_{t}(x_t - \bar x)(x_{t-k} - \bar x)}{\sum_t (x_t - \bar x)^2},
   \qquad |\hat\rho_k| \lesssim \frac{1.96}{\sqrt n}\ \text{under independence}

*Implementation:* :func:`financialkit.time_series.systems.efficiency.autocorrelation`
(by FFT) and :func:`~financialkit.time_series.systems.efficiency.bartlett_band`;
:func:`~financialkit.time_series.visualizers.plots.plot_acf` draws the
correlogram.

*References:* M. G. Kendall, "The Analysis of Economic Time-Series -- Part
I: Prices," *Journal of the Royal Statistical Society A* 116(1) (1953)
11-34.

.. minigallery:: ../../examples/time_series/efficiency/plot_01_kendall_autocorrelation.py

1960 -- Kalman Filter and the State-Space Model
-----------------------------------------------

Rudolf Kalman wrote a linear dynamic system as a hidden state that evolves
by :math:`x_t = F x_{t-1} + w_t` and is measured with noise as
:math:`y_t = H x_t + v_t`. His recursion updates the Gaussian mean and
covariance of the state with each measurement, weighting the surprise by a
*gain* that balances state noise against measurement noise, and it needs
no stored history. It guided the Apollo spacecraft before it reached
economics. The same pass gives the exact likelihood as a product of
independent one-step prediction errors (Schweppe 1965), which made the
state-space form the general way to estimate time-series models: ARMA,
unobserved components, time-varying betas and term-structure factors.
Muth's (1960) random walk plus noise, an efficient price seen through
microstructure noise, is the simplest case. Its steady-state filter is an
exponentially weighted moving average.

.. math::

   \hat x_{t|t} = \hat x_{t|t-1} + K_t\bigl(y_t - H\hat x_{t|t-1}\bigr), \qquad
   K_t = P_{t|t-1}H'\bigl(HP_{t|t-1}H' + R\bigr)^{-1}

*Implementation:* :func:`financialkit.time_series.systems.state_space.kalman_filter`
returns a :class:`~financialkit.time_series.core.base.KalmanResult` with
the prediction-error log-likelihood, missing-value handling and the
Rauch-Tung-Striebel :meth:`~financialkit.time_series.core.base.KalmanResult.smooth`;
:func:`~financialkit.time_series.systems.state_space.fit_local_level`
estimates the local-level model by maximum likelihood
(:func:`scipy.optimize.minimize`), and
:func:`~financialkit.time_series.systems.state_space.local_level_steady_state_gain`
solves its Riccati equation.

*References:* R. E. Kalman, "A New Approach to Linear Filtering and
Prediction Problems," *Journal of Basic Engineering* 82(1) (1960) 35-45;
J. F. Muth, "Optimal Properties of Exponentially Weighted Forecasts,"
*Journal of the American Statistical Association* 55(290) (1960) 299-306;
H. E. Rauch, F. Tung and C. T. Striebel, "Maximum Likelihood Estimates of
Linear Dynamic Systems," *AIAA Journal* 3(8) (1965) 1445-1450; J. Durbin
and S. J. Koopman, *Time Series Analysis by State Space Methods*, 2nd ed.
(Oxford, 2012).

.. minigallery:: ../../examples/time_series/state_space/plot_01_kalman_filter.py

1963 -- Mandelbrot: Fat Tails
-----------------------------

Benoit Mandelbrot plotted the frequency of large daily and monthly
changes in cotton prices on doubly logarithmic paper and found
straight lines. The tails decay like a power law, not like the Gaussian
that Bachelier's random walk assumed, so extreme moves are vastly more
common than the normal distribution predicts. He proposed the stable
Paretian laws, with index :math:`\alpha \approx 1.7`, which have infinite
variance. He also noted that large changes tend to be followed by large
changes. Later work settled on tail indices between 3 and 5 for most
assets, so variances are finite, but fat tails and volatility clustering
became the first two *stylized facts* of returns.

.. math::

   P(|X| > u) \sim C\,u^{-\alpha}, \qquad u \to \infty

*Implementation:* :func:`financialkit.time_series.systems.stylized_facts.tail_survival`
(the log-log tail), :func:`~financialkit.time_series.systems.stylized_facts.return_statistics`
(moments and the Jarque-Bera test via :func:`scipy.stats.jarque_bera`) and
:func:`~financialkit.time_series.systems.stylized_facts.log_returns`.

*References:* B. Mandelbrot, "The Variation of Certain Speculative
Prices," *Journal of Business* 36(4) (1963) 394-419; R. Cont, "Empirical
Properties of Asset Returns: Stylized Facts and Statistical Issues,"
*Quantitative Finance* 1(2) (2001) 223-236.

.. minigallery:: ../../examples/time_series/stylized_facts/plot_01_mandelbrot_fat_tails.py

1965 -- Fama's Runs Test
------------------------

Eugene Fama's thesis, published in full in the *Journal of Business*,
examined the daily prices of the 30 Dow Jones Industrials. Correlation coefficients can be
dominated by a few huge changes, which Mandelbrot had just shown to be
common. Fama therefore added a distribution-free test that uses only the
signs of the changes, counting the *runs* of consecutive changes with the
same sign. Trends would produce too few runs and reversals too many.
The counts were close to their expected values under independence. The
paper, and Fama's 1970 survey, made the efficient-markets hypothesis the
organizing idea of empirical finance.

.. math::

   m = \frac{N(N+1) - \sum_i n_i^2}{N}, \qquad
   \sigma_m^2 = \frac{\sum_i n_i^2\bigl[\sum_i n_i^2 + N(N+1)\bigr] - 2N\sum_i n_i^3 - N^3}{N^2(N-1)}

*Implementation:* :func:`financialkit.time_series.systems.efficiency.runs_test`
with Fama's three-sign (+, -, 0) moments, returning a
:class:`~financialkit.time_series.core.base.TestResult`.

*References:* E. F. Fama, "The Behavior of Stock-Market Prices," *Journal
of Business* 38(1) (1965) 34-105; A. Wald and J. Wolfowitz, "On a Test
Whether Two Samples Are from the Same Population," *Annals of
Mathematical Statistics* 11(2) (1940) 147-162.

.. minigallery:: ../../examples/time_series/efficiency/plot_02_fama_runs_test.py

1970 -- Box-Jenkins ARMA
------------------------

George Box and Gwilym Jenkins's book turned mixed autoregressive
moving-average models into a practical method with three stages. First,
*identify* the orders from the autocorrelation function, which cuts off
after lag :math:`q` for a moving average, and the partial autocorrelation
function, which cuts off after lag :math:`p` for an autoregression.
Second, *estimate* by maximum likelihood. Third, *check* the residuals for
leftover structure, and repeat until they are white noise. Then forecast,
with error bands from the model's moving-average representation. ARMA
models became the benchmark that every economic forecast had to beat.

.. math::

   \phi(B)(x_t - \mu) = \theta(B)\,\varepsilon_t, \qquad
   \operatorname{Var}(\hat x_{T+h} - x_{T+h}) = \sigma^2\sum_{j<h}\psi_j^2

*Implementation:* :func:`financialkit.time_series.systems.linear_models.fit_arma`
(conditional maximum likelihood by :func:`scipy.optimize.least_squares`,
kept stationary and invertible by Monahan's transform), returning an
:class:`~financialkit.time_series.core.base.ARMAResult` with
:meth:`~financialkit.time_series.core.base.ARMAResult.forecast`;
:func:`~financialkit.time_series.systems.linear_models.partial_autocorrelation`
(Durbin-Levinson), :func:`~financialkit.time_series.systems.linear_models.arma_acf`
and :func:`~financialkit.time_series.systems.linear_models.arma_psi_weights`.

*References:* G. E. P. Box and G. M. Jenkins, *Time Series Analysis:
Forecasting and Control* (San Francisco: Holden-Day, 1970); J. F. Monahan,
"A Note on Enforcing Stationarity in Autoregressive-Moving Average
Models," *Biometrika* 71(2) (1984) 403-404.

.. minigallery:: ../../examples/time_series/linear_models/plot_02_box_jenkins_arma.py

1974 -- Akaike Information Criterion
------------------------------------

Box and Jenkins chose orders by inspecting correlograms. Hirotugu Akaike
replaced the judgement call with a number. The maximized log-likelihood
is an optimistic estimate of how well a model will predict new data, and
Akaike showed that the optimism is about one unit per estimated
parameter. Minimizing :math:`-2\ell + 2k` therefore picks the model with
the smallest expected Kullback-Leibler distance from the truth. Gideon
Schwarz (1978) derived the heavier Bayesian penalty
:math:`k \ln n`, which selects the true order with probability tending
to one. AIC is better for prediction and BIC for identification.

.. math::

   \mathrm{AIC} = -2\ell(\hat\theta) + 2k, \qquad \mathrm{BIC} = -2\ell(\hat\theta) + k\ln n

*Implementation:* :func:`financialkit.time_series.systems.linear_models.select_arma_order`
fits every order on a common sample and returns an
:class:`~financialkit.time_series.core.base.OrderSelectionResult`;
every :class:`~financialkit.time_series.core.base.ARMAResult` and
:class:`~financialkit.time_series.core.base.VolatilityFit` reports
``aic`` and ``bic``.

*References:* H. Akaike, "A New Look at the Statistical Model
Identification," *IEEE Transactions on Automatic Control* 19(6) (1974)
716-723; G. Schwarz, "Estimating the Dimension of a Model," *Annals of
Statistics* 6(2) (1978) 461-464.

.. minigallery:: ../../examples/time_series/linear_models/plot_03_akaike_order_selection.py

1975 -- Hill Tail Index
-----------------------

If a tail is a power law, the logarithms of the largest observations,
measured from a high threshold, are exponentially distributed with mean
:math:`1/\alpha`. Bruce Hill turned this into a simple estimator that
uses only the :math:`k` largest order statistics and is the maximum
likelihood estimator for an exact Pareto tail. The *Hill plot* of
:math:`\hat\alpha_k` against :math:`k` shows the central difficulty of
tail estimation: small :math:`k` is noisy, and large :math:`k` reaches
into the body of the distribution and is biased. Applied to returns, it
gave tail indices between 3 and 5, so variances exist but fourth moments
barely do.

.. math::

   \hat\alpha_k = \Bigl[\frac1k\sum_{i=1}^{k}\ln X_{(i)} - \ln X_{(k+1)}\Bigr]^{-1},
   \qquad \operatorname{se}(\hat\alpha_k) \approx \frac{\alpha}{\sqrt k}

*Implementation:* :func:`financialkit.time_series.systems.stylized_facts.hill_estimator`
returns a :class:`~financialkit.time_series.core.base.TailIndexResult`
for either tail; :func:`~financialkit.time_series.visualizers.plots.plot_hill`
draws the Hill plot.

*References:* B. M. Hill, "A Simple General Approach to Inference About
the Tail of a Distribution," *Annals of Statistics* 3(5) (1975)
1163-1174.

.. minigallery:: ../../examples/time_series/stylized_facts/plot_02_hill_tail_index.py

1978 -- Ljung-Box Portmanteau Test
----------------------------------

Checking many autocorrelations one by one is bound to find a
"significant" one by chance. Box and Pierce (1970) pooled the first
:math:`h` into a single chi-squared statistic for diagnosing fitted ARMA
residuals. Its distribution is far from :math:`\chi^2` in samples of
realistic size, so it rejected too rarely. Greta Ljung and George Box
weighted each term by :math:`(n+2)/(n-k)`, the reciprocal of its exact
variance factor, and the corrected statistic matches its reference
closely. Applied to squared returns rather than returns, the same test
became a standard check for volatility clustering.

.. math::

   Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2}{n-k} \;\sim\; \chi^2_{h-p-q}

*Implementation:* :func:`financialkit.time_series.systems.efficiency.ljung_box`
and :func:`~financialkit.time_series.systems.efficiency.box_pierce`, with
``fitted_params`` for ARMA residuals.

*References:* G. M. Ljung and G. E. P. Box, "On a Measure of Lack of Fit
in Time Series Models," *Biometrika* 65(2) (1978) 297-303; G. E. P. Box
and D. A. Pierce, "Distribution of Residual Autocorrelations in
Autoregressive-Integrated Moving Average Time Series Models," *Journal of
the American Statistical Association* 65(332) (1970) 1509-1526.

.. minigallery:: ../../examples/time_series/efficiency/plot_03_ljung_box.py

1979 -- Dickey-Fuller Unit Roots
--------------------------------

Whether a series has a unit root decides whether shocks to it are
permanent, as for a stock price, or fade away, as for a stationary
interest rate. The obvious test, the t-ratio of :math:`\gamma` in
:math:`\Delta y_t = \gamma y_{t-1} + e_t`, does not have a t distribution
when :math:`\gamma = 0`: it is skewed to the left. David Dickey and Wayne
Fuller derived and tabulated its distribution by simulation. Said and
Dickey (1984) added lagged differences to absorb serial correlation, and
James MacKinnon's response surfaces give critical values for any sample
size. Nelson and Plosser's (1982) finding that most macroeconomic series
cannot reject a unit root changed empirical macroeconomics.

.. math::

   \Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{k}\delta_i\,\Delta y_{t-i} + e_t,
   \qquad H_0: \gamma = 0

*Implementation:* :func:`financialkit.time_series.systems.unit_roots.adf_test`
(AIC lag selection), returning a
:class:`~financialkit.time_series.core.base.UnitRootResult` with
:func:`~financialkit.time_series.systems.unit_roots.mackinnon_critical_values`;
:func:`~financialkit.time_series.utils.synthetic.simulate_dickey_fuller`
traces the null distribution.

*References:* D. A. Dickey and W. A. Fuller, "Distribution of the
Estimators for Autoregressive Time Series with a Unit Root," *Journal of
the American Statistical Association* 74(366) (1979) 427-431; S. E. Said
and D. A. Dickey, "Testing for Unit Roots in Autoregressive-Moving
Average Models of Unknown Order," *Biometrika* 71(3) (1984) 599-607; J.
G. MacKinnon, "Critical Values for Cointegration Tests," Queen's
Economics Department Working Paper 1227 (2010).

.. minigallery:: ../../examples/time_series/unit_roots/plot_01_dickey_fuller.py

1980 -- Parkinson's Range Volatility
------------------------------------

Close-to-close returns throw away almost everything that happened during
the day. Michael Parkinson, a physicist, applied the theory of the range
of Brownian motion. The squared log ratio of the high to the low, divided
by :math:`4\ln 2`, is an unbiased estimate of the period's variance and
is about five times more efficient than the squared return. A month of
daily ranges carries roughly as much information as five months of
closes. The idea led to the Garman-Klass and Rogers-Satchell estimators
and to range-based volatility models.

.. math::

   \hat\sigma^2_t = \frac{(\ln H_t - \ln L_t)^2}{4\ln 2}

*Implementation:* :func:`financialkit.time_series.systems.realized.parkinson_variance`
and :func:`~financialkit.time_series.systems.realized.close_to_close_variance`;
:func:`~financialkit.time_series.utils.synthetic.simulate_ohlc` builds
daily bars from exact GBM paths (:func:`financialkit.integrators.gbm_paths`).

*References:* M. Parkinson, "The Extreme Value Method for Estimating the
Variance of the Rate of Return," *Journal of Business* 53(1) (1980)
61-65.

.. minigallery:: ../../examples/time_series/volatility/plot_01_parkinson_range.py

1980 -- Sims's Vector Autoregression
------------------------------------

The large macroeconometric models of the 1970s identified their equations
by excluding variables and treating others as exogenous, restrictions that
Christopher Sims called "incredible". He proposed instead to regress every
variable on lags of all of them and to let the data speak. The fitted VAR
is summarized by its *impulse responses* to orthogonalized shocks, built
from a Cholesky factor of the residual covariance so that the ordering of
the variables fixes which shocks can move which variables within the
period, and by the *variance decomposition* of each forecast error into
the shares due to each shock. VARs became the standard tool for monetary
policy analysis, and Sims shared the 2011 Nobel prize.

.. math::

   y_t = c + \sum_{i=1}^{p}A_i\,y_{t-i} + u_t, \qquad
   \frac{\partial y_{t+s}}{\partial \varepsilon_t'} = \Phi_s P, \quad PP' = \Sigma

*Implementation:* :func:`financialkit.time_series.systems.vector_autoregression.fit_var`
(equation-by-equation least squares by :func:`numpy.linalg.lstsq`) returns a
:class:`~financialkit.time_series.core.base.VectorAutoregressionResult` with
:meth:`~financialkit.time_series.core.base.VectorAutoregressionResult.impulse_response`,
:meth:`~financialkit.time_series.core.base.VectorAutoregressionResult.fevd`,
:meth:`~financialkit.time_series.core.base.VectorAutoregressionResult.forecast`, the
companion-matrix stability check and AIC/BIC;
:func:`~financialkit.time_series.systems.vector_autoregression.simulate_var`.

*References:* C. A. Sims, "Macroeconomics and Reality," *Econometrica*
48(1) (1980) 1-48; H. Lütkepohl, *New Introduction to Multiple Time Series
Analysis* (Springer, 2005), Chs. 2-3.

.. minigallery:: ../../examples/time_series/linear_models/plot_04_sims_var.py

1982 -- Engle's ARCH
--------------------

Robert Engle was modelling UK inflation, whose forecast errors were small
in some periods and large in others. He let the conditional variance
depend on past squared errors. The model's returns are uncorrelated but
not independent, cluster in volatility and have fat tails even with
normal innovations, which reproduces Mandelbrot's two stylized facts at
once. Engle also gave a Lagrange-multiplier test that regresses squared
residuals on their own lags. ARCH opened the econometrics of risk and
earned him the 2003 Nobel prize.

.. math::

   r_t = \sigma_t z_t, \qquad \sigma_t^2 = \omega + \alpha\,r_{t-1}^2, \qquad
   \mathrm{LM} = T R^2 \sim \chi^2_q

*Implementation:* :class:`financialkit.time_series.systems.garch.ARCH`
(a :class:`~financialkit.time_series.core.base.VolatilityModel` fitted by
maximum likelihood) and :func:`~financialkit.time_series.systems.garch.arch_lm_test`.

*References:* R. F. Engle, "Autoregressive Conditional Heteroscedasticity
with Estimates of the Variance of United Kingdom Inflation,"
*Econometrica* 50(4) (1982) 987-1007.

.. minigallery:: ../../examples/time_series/volatility/plot_02_engle_arch.py

1986 -- Bollerslev's GARCH
--------------------------

Volatility decays slowly, so pure ARCH needed long lag lists with
awkward constraints. Tim Bollerslev, Engle's student, added the lagged
conditional variance itself, which gives an ARCH(:math:`\infty`) with
geometrically declining weights in just three parameters. GARCH(1,1)
fitted to daily returns almost anywhere gives a small :math:`\alpha`
and a persistence :math:`\alpha + \beta` just below one. It is still the
default volatility model of practitioners, and its forecasts mean-revert
to a long-run variance at a rate set by the persistence.

.. math::

   \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2, \qquad
   E_T[\sigma^2_{T+h}] = \bar\sigma^2 + (\alpha+\beta)^{h-1}(\sigma^2_{T+1} - \bar\sigma^2)

*Implementation:* :class:`financialkit.time_series.systems.garch.GARCH`
(variance filtering by :func:`scipy.signal.lfilter`, estimation by
:func:`scipy.optimize.minimize` with standard errors from the numerical
Hessian), with :meth:`~financialkit.time_series.systems.garch.GARCH.forecast`
and :func:`~financialkit.time_series.visualizers.plots.plot_conditional_volatility`.

*References:* T. Bollerslev, "Generalized Autoregressive Conditional
Heteroskedasticity," *Journal of Econometrics* 31(3) (1986) 307-327.

.. minigallery:: ../../examples/time_series/volatility/plot_03_bollerslev_garch.py

1987 -- Engle-Granger Cointegration
-----------------------------------

Granger and Newbold (1974) had shown that regressions between
independent random walks routinely produce high :math:`R^2` and
"significant" coefficients. Yet economic theory ties some nonstationary
series together, such as consumption and income, spot and futures
prices, or a stock's two share classes. Clive Granger called such series
*cointegrated*: each wanders, but a linear combination does not. Engle
and Granger proved that cointegrated series have an error-correction
representation and proposed a two-step test. The first step regresses
the levels, and the second tests the residuals for a unit root, using
critical values that account for the estimated coefficients.

.. math::

   y_t = a + b\,x_t + u_t, \qquad u_t \sim I(0) \iff y, x \text{ cointegrated}

*Implementation:* :func:`financialkit.time_series.systems.unit_roots.engle_granger`
returns a :class:`~financialkit.time_series.core.base.CointegrationResult`
with MacKinnon's critical values for two to four variables;
:func:`~financialkit.time_series.utils.synthetic.simulate_cointegrated_pair`
builds test pairs.

*References:* R. F. Engle and C. W. J. Granger, "Co-Integration and Error
Correction: Representation, Estimation, and Testing," *Econometrica*
55(2) (1987) 251-276; C. W. J. Granger and P. Newbold, "Spurious
Regressions in Econometrics," *Journal of Econometrics* 2(2) (1974)
111-120.

.. minigallery:: ../../examples/time_series/unit_roots/plot_02_engle_granger.py

1988 -- Lo-MacKinlay Variance Ratio
-----------------------------------

If log prices are a random walk, the variance of :math:`q`-period returns
grows in proportion to :math:`q`. Andrew Lo and Craig MacKinlay built a
test on the ratio of the two variances. It uses overlapping observations,
corrects for bias, and has a standard error that stays valid under
volatility clustering. On weekly U.S. stock returns from 1962 to 1985 it
rejected the random walk decisively, chiefly through the positive
autocorrelation of small-stock portfolios. The paper's title, "Stock
Market Prices Do Not Follow Random Walks," reopened the debate that
Kendall and Fama seemed to have settled.

.. math::

   VR(q) = \frac{\hat\sigma^2_c(q)}{\hat\sigma^2_a} \approx 1 + 2\sum_{k=1}^{q-1}\Bigl(1 - \frac kq\Bigr)\rho_k,
   \qquad z^*(q) = \frac{\sqrt T\,(VR(q) - 1)}{\sqrt{\hat\theta(q)}}

*Implementation:* :func:`financialkit.time_series.systems.efficiency.variance_ratio`
returns a :class:`~financialkit.time_series.core.base.VarianceRatioResult`
with both the homoskedastic and the heteroskedasticity-consistent
statistics; :func:`~financialkit.time_series.visualizers.plots.plot_variance_ratio`.

*References:* A. W. Lo and A. C. MacKinlay, "Stock Market Prices Do Not
Follow Random Walks: Evidence from a Simple Specification Test," *Review
of Financial Studies* 1(1) (1988) 41-66; J. Y. Campbell, A. W. Lo and A.
C. MacKinlay, *The Econometrics of Financial Markets* (Princeton, 1997),
Ch. 2.

.. minigallery:: ../../examples/time_series/efficiency/plot_04_lo_mackinlay.py

1988 -- Johansen Cointegration Test
-----------------------------------

The Engle-Granger test finds at most one cointegrating relation, and which
one depends on the variable put on the left of the regression. Søren
Johansen estimated the whole cointegrating space by maximum likelihood. In
the error-correction form of a VAR, the matrix :math:`\Pi = \alpha\beta'`
on the lagged levels has rank equal to the number of cointegrating
relations. After partialling out the short-run dynamics, the likelihood is
maximized by a generalized eigenproblem whose eigenvalues are squared
canonical correlations between the changes and the lagged levels. The
trace and maximum-eigenvalue statistics test the rank one step at a time.
Their nonstandard distributions depend only on the number of
nonstationary directions and the deterministic terms.

.. math::

   \Delta y_t = \mu + \alpha\beta' y_{t-1} + \sum_{i=1}^{l}\Gamma_i\,\Delta y_{t-i} + u_t, \qquad
   \lambda_{\text{trace}}(r) = -T\sum_{i=r+1}^{k}\ln(1 - \hat\lambda_i)

*Implementation:* :func:`financialkit.time_series.systems.unit_roots.johansen_test`
(the eigenproblem by :func:`scipy.linalg.eigh`) returns a
:class:`~financialkit.time_series.core.base.JohansenResult` with both
statistics, MacKinnon-Haug-Michelis critical values for two to five
series, :meth:`~financialkit.time_series.core.base.JohansenResult.rank`
and the normalized
:meth:`~financialkit.time_series.core.base.JohansenResult.cointegrating_vectors`.

*References:* S. Johansen, "Statistical Analysis of Cointegration
Vectors," *Journal of Economic Dynamics and Control* 12(2-3) (1988)
231-254; S. Johansen, "Estimation and Hypothesis Testing of Cointegration
Vectors in Gaussian Vector Autoregressive Models," *Econometrica* 59(6)
(1991) 1551-1580; J. G. MacKinnon, A. A. Haug and L. Michelis, "Numerical
Distribution Functions of Likelihood Ratio Tests for Cointegration,"
*Journal of Applied Econometrics* 14(5) (1999) 563-577.

.. minigallery:: ../../examples/time_series/unit_roots/plot_03_johansen.py

1989 -- Hamilton's Regime-Switching Model
-----------------------------------------

Recessions are not just small expansions: the economy seems to move
between distinct states. James Hamilton let the mean growth rate of U.S.
GNP switch between two values according to a hidden Markov chain, and gave
the nonlinear filter that turns each observation into updated
probabilities of the states, together with the likelihood. The estimated
recession probabilities matched the NBER's dating of business cycles
without using it. Kim (1994) added the smoother, and Hamilton (1990) the
EM algorithm. With switching means and variances, the model separates calm
bull markets from turbulent bear markets in returns, and its regimes
became a standard ingredient of asset allocation and term-structure
models.

.. math::

   \xi_{t|t} = \frac{(P'\xi_{t-1|t-1}) \odot \eta_t}{\mathbf 1'\bigl[(P'\xi_{t-1|t-1}) \odot \eta_t\bigr]},
   \qquad \eta_{t,j} = \phi\bigl(x_t;\,\mu_j, \sigma_j^2\bigr)

*Implementation:* :func:`financialkit.time_series.systems.regime_switching.hamilton_filter`,
:func:`~financialkit.time_series.systems.regime_switching.kim_smoother` and
the EM fit :func:`~financialkit.time_series.systems.regime_switching.fit_markov_switching`,
returning a :class:`~financialkit.time_series.core.base.RegimeSwitchingResult`
with expected durations and ergodic probabilities;
:func:`~financialkit.time_series.systems.regime_switching.simulate_markov_switching`.

*References:* J. D. Hamilton, "A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle," *Econometrica* 57(2)
(1989) 357-384; J. D. Hamilton, "Analysis of Time Series Subject to
Changes in Regime," *Journal of Econometrics* 45(1-2) (1990) 39-70; C.-J.
Kim, "Dynamic Linear Models with Markov-Switching," *Journal of
Econometrics* 60(1-2) (1994) 1-22.

.. minigallery:: ../../examples/time_series/state_space/plot_02_hamilton_regime_switching.py

1991 -- Nelson's EGARCH
-----------------------

GARCH responds to the size of a shock but not to its sign, and its
coefficients need positivity constraints. Daniel Nelson modelled the
logarithm of the variance instead. The variance is then positive by
construction, and a term in the signed standardized shock lets bad news
raise volatility more than good news of the same size. This is the
*leverage effect* that Black (1976) had observed in stock returns. Nelson
also worked out the model's stationarity and moments, and found a
strongly negative asymmetry in daily returns on the U.S. stock market.

.. math::

   \ln\sigma_t^2 = \omega + \beta\ln\sigma_{t-1}^2 + \alpha\bigl(|z_{t-1}| - E|z|\bigr) + \gamma\,z_{t-1}

*Implementation:* :class:`financialkit.time_series.systems.garch.EGARCH`,
with the closed-form unconditional variance for normal innovations;
:func:`~financialkit.time_series.visualizers.plots.plot_news_impact`
compares news impact curves.

*References:* D. B. Nelson, "Conditional Heteroskedasticity in Asset
Returns: A New Approach," *Econometrica* 59(2) (1991) 347-370; F. Black,
"Studies of Stock Price Volatility Changes," *Proceedings of the American
Statistical Association, Business and Economic Statistics Section*
(1976) 177-181.

.. minigallery:: ../../examples/time_series/volatility/plot_04_nelson_egarch.py

1992 -- KPSS Stationarity Test
------------------------------

Dickey-Fuller tests keep the unit root unless the evidence against it is
strong, and they have little power against persistent stationary
alternatives, so "cannot reject a unit root" is often all they say. Denis
Kwiatkowski, Peter Phillips, Peter Schmidt and Yongcheol Shin reversed the
null hypothesis. They write a series as a deterministic trend, a random
walk and stationary noise, and derive the Lagrange-multiplier test that
the random walk's variance is zero. The statistic is the scaled sum of
squared partial sums of the residuals, which stays bounded for a
stationary series and explodes under a unit root. Run together, the two
tests can say "stationary", "unit root" or "the data cannot tell".

.. math::

   \eta = \frac{1}{T^2\hat\sigma^2_\infty}\sum_{t=1}^{T}S_t^2, \qquad
   S_t = \sum_{s=1}^{t}\hat e_s

*Implementation:* :func:`financialkit.time_series.systems.unit_roots.kpss_test`
(level or trend stationarity, with a Bartlett-window long-run variance)
returns a :class:`~financialkit.time_series.core.base.StationarityResult`
with the paper's critical values and an interpolated p-value.

*References:* D. Kwiatkowski, P. C. B. Phillips, P. Schmidt and Y. Shin,
"Testing the Null Hypothesis of Stationarity against the Alternative of a
Unit Root," *Journal of Econometrics* 54(1-3) (1992) 159-178; W. K. Newey
and K. D. West, "A Simple, Positive Semi-Definite, Heteroskedasticity and
Autocorrelation Consistent Covariance Matrix," *Econometrica* 55(3) (1987)
703-708.

.. minigallery:: ../../examples/time_series/unit_roots/plot_04_kpss.py

1993 -- Glosten-Jagannathan-Runkle
----------------------------------

Do investors demand a higher expected return when volatility is high?
Estimates of this risk-return trade-off had come out positive, negative
or insignificant depending on the volatility model. Lawrence Glosten,
Ravi Jagannathan and David Runkle showed that the answer hinged on the
asymmetric response of volatility to news. They added a single term to
GARCH for negative shocks. On U.S. monthly excess returns nearly all of
the response came from bad news. The GJR model keeps GARCH's analytic
forecasts and became, with EGARCH, the standard asymmetric volatility
model.

.. math::

   \sigma_t^2 = \omega + \bigl(\alpha + \gamma\,\mathbf 1\{\varepsilon_{t-1} < 0\}\bigr)\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2

*Implementation:* :class:`financialkit.time_series.systems.garch.GJRGARCH`,
with persistence :math:`\alpha + \gamma/2 + \beta` and analytic
:meth:`~financialkit.time_series.systems.garch.GJRGARCH.forecast`.

*References:* L. R. Glosten, R. Jagannathan and D. E. Runkle, "On the
Relation between the Expected Value and the Volatility of the Nominal
Excess Return on Stocks," *Journal of Finance* 48(5) (1993) 1779-1801.

.. minigallery:: ../../examples/time_series/volatility/plot_05_gjr_garch.py

1994 -- RiskMetrics EWMA
------------------------

J.P. Morgan's chairman wanted a single number for the firm's risk at
4:15 every afternoon. The methodology behind it was published in October
1994, free of charge, as RiskMetrics. Its volatility model was an
exponentially weighted moving average of squared returns, with one decay
factor for every asset: :math:`\lambda = 0.94` for daily data and
:math:`0.97` for monthly. It is a GARCH(1,1) without a constant and with
unit persistence. It needs no estimation and adapts quickly, and it
became the industry's default and the template for Value-at-Risk
reporting.

.. math::

   \sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1 - \lambda)\,r_{t-1}^2

*Implementation:* :class:`financialkit.time_series.systems.garch.EWMA`
(``lam=0.94`` by default; :meth:`~financialkit.time_series.core.base.VolatilityModel.fit`
estimates :math:`\lambda` by maximum likelihood instead).

*References:* J.P. Morgan/Reuters, *RiskMetrics -- Technical Document*,
4th ed. (New York, 1996), Ch. 5.

.. minigallery:: ../../examples/time_series/volatility/plot_06_riskmetrics_ewma.py

1998 -- Andersen-Bollerslev Realized Volatility
-----------------------------------------------

Regressions of squared daily returns on GARCH forecasts gave
:math:`R^2` of a few percent, and critics concluded that volatility
models were useless out of sample. Torben Andersen and Tim Bollerslev
showed that the squared return is to blame. It is an unbiased but
extremely noisy measure of the day's variance, so even a perfect
forecast explains little of it. Summing squared intraday returns gives
*realized variance*, which converges to the day's integrated variance.
Against it the same forecasts explain far more. Realized measures went
on to underpin variance swaps and a new generation of volatility
forecasting models.

.. math::

   RV_t = \sum_{i=1}^{m} r_{t,i}^2 \;\xrightarrow{\;m\to\infty\;}\; \int_{t-1}^{t}\sigma_s^2\,ds

*Implementation:* :func:`financialkit.time_series.systems.realized.realized_variance`
and :func:`~financialkit.time_series.systems.realized.mincer_zarnowitz`;
:func:`~financialkit.time_series.utils.synthetic.simulate_intraday_returns`
spreads each day's variance over intraday returns.

*References:* T. G. Andersen and T. Bollerslev, "Answering the Skeptics:
Yes, Standard Volatility Models Do Provide Accurate Forecasts,"
*International Economic Review* 39(4) (1998) 885-905; J. Mincer and V.
Zarnowitz, "The Evaluation of Economic Forecasts," in *Economic Forecasts
and Expectations* (New York: NBER, 1969), 3-46.

.. minigallery:: ../../examples/time_series/volatility/plot_07_realized_volatility.py

2002 -- Engle's Dynamic Conditional Correlation
-----------------------------------------------

Diversification fails when it is needed most, because correlations rise
in crises. Multivariate GARCH models could capture this, but with a free
covariance matrix their parameters grow with the fourth power of the
number of assets, and positive definiteness is hard to guarantee. Robert
Engle split the problem in two. Each asset gets its own univariate GARCH,
and the correlation matrix of the standardized returns follows a scalar
GARCH-like recursion with two parameters, its intercept fixed at the
sample correlation. The two-step estimate scales to hundreds of assets,
and DCC became the workhorse for time-varying correlations in risk
management and portfolio construction.

.. math::

   Q_t = (1 - a - b)\bar Q + a\,z_{t-1}z_{t-1}' + b\,Q_{t-1}, \qquad
   R_t = \operatorname{diag}(Q_t)^{-1/2}\,Q_t\,\operatorname{diag}(Q_t)^{-1/2}

*Implementation:* :func:`financialkit.time_series.systems.dcc.fit_dcc`
(GARCH fits per series, then the correlation likelihood by
:func:`scipy.optimize.minimize`) returns a
:class:`~financialkit.time_series.core.base.DCCFit` with the conditional
correlations and covariances;
:func:`~financialkit.time_series.systems.dcc.dcc_correlations` (the
recursion by :func:`scipy.signal.lfilter`),
:func:`~financialkit.time_series.systems.dcc.dcc_loglikelihood` and
:func:`~financialkit.time_series.systems.dcc.simulate_dcc`.

*References:* R. Engle, "Dynamic Conditional Correlation: A Simple Class
of Multivariate Generalized Autoregressive Conditional Heteroskedasticity
Models," *Journal of Business & Economic Statistics* 20(3) (2002)
339-350; T. Bollerslev, "Modelling the Coherence in Short-Run Nominal
Exchange Rates: A Multivariate Generalized ARCH Model," *Review of
Economics and Statistics* 72(3) (1990) 498-505.

.. minigallery:: ../../examples/time_series/volatility/plot_09_engle_dcc.py

2009 -- Corsi's HAR Model of Realized Volatility
------------------------------------------------

Realized variance turned volatility into an observed series, and that
series has long memory: its autocorrelations decay slowly, almost
hyperbolically, far beyond what a GARCH or a short autoregression
produces. Fractionally integrated models fit it but are awkward to
estimate. Fulvio Corsi used the heterogeneous market hypothesis of Müller
and coauthors (1997), in which traders with daily, weekly and monthly
horizons each react to volatility at their own scale. He regressed
tomorrow's realized variance on its averages over the last day, week and
month. The model is an AR(22) with step-shaped weights and only four
coefficients, it is fitted by least squares, and it reproduces the
apparent long memory. It became the benchmark that new volatility
forecasts must beat.

.. math::

   RV_{t+1} = \beta_0 + \beta_d\,RV_t + \beta_w\,\tfrac15\sum_{j=0}^{4}RV_{t-j}
   + \beta_m\,\tfrac1{22}\sum_{j=0}^{21}RV_{t-j} + e_{t+1}

*Implementation:* :func:`financialkit.time_series.systems.realized.fit_har`
returns a :class:`~financialkit.time_series.core.base.HARResult` with
in- and out-of-sample
:meth:`~financialkit.time_series.core.base.HARResult.predict` and
:meth:`~financialkit.time_series.core.base.HARResult.forecast`;
:func:`~financialkit.time_series.systems.realized.har_regressors` builds
the averages for any set of horizons.

*References:* F. Corsi, "A Simple Approximate Long-Memory Model of
Realized Volatility," *Journal of Financial Econometrics* 7(2) (2009)
174-196; U. A. Müller, M. M. Dacorogna, R. D. Davé, R. B. Olsen, O. V.
Pictet and J. E. von Weizsäcker, "Volatilities of Different Time
Resolutions -- Analyzing the Dynamics of Market Components," *Journal of
Empirical Finance* 4(2-3) (1997) 213-239.

.. minigallery:: ../../examples/time_series/volatility/plot_08_corsi_har.py

See Also
--------

- :doc:`/api/time_series`
- :doc:`/history/options_breakthroughs`
- :doc:`/history/interest_rate_models_breakthroughs`

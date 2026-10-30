# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Every domain re-exports its visualizers (ROADMAP C10): `options`,
  `pde_methods`, `stochastic_volatility`, `risk`, `portfolio`,
  `asset_pricing`, `credit` and `market_microstructure` now list their
  `plot_*` functions in `__all__`, as the other six domains already did,
  so they appear in the API reference.
- `financialkit.datasets`: real data bundled as small CSVs and loaded as
  numpy arrays, with no network access at run time. `load_shiller`
  (Shiller's monthly S&P price, dividends, earnings, CPI and long rate
  since 1871), `load_market_daily` (the CRSP market's daily excess and
  T-bill returns since 1926), `load_fama_french_factors` (monthly
  Mkt-RF, SMB, HML, RMW, CMA, momentum and RF), `load_fama_french_25`
  (the 25 size by book-to-market portfolios since 1963) and
  `load_treasury_yields` (month-end 3-month to 10-year Treasury yields
  since 1981). Each returns a frozen dataclass with `datetime64` dates
  and decimal returns. `scripts/fetch_datasets.py` writes the CSVs from
  the original sources, using only the standard library; the author runs
  it by hand.
- `asset_pricing` breakthroughs from 1982 to 2016, each with a history
  entry and gallery example. Hansen's GMM (`gmm`, `GMMResult`, with
  `newey_west`) and Hansen and Singleton's Euler-equation estimate
  (`hansen_singleton`). The Campbell-Shiller log-linear present value
  (`log_linear_return`, `campbell_shiller_rho`,
  `campbell_shiller_decomposition`, `PresentValueDecomposition`).
  Campbell-Cochrane habit formation (`CampbellCochrane`) and Bansal-Yaron
  long-run risk (`BansalYaron`). The Fama-French five factors
  (`fama_french_five`, from `two_by_three_portfolios`), and
  Harvey-Liu-Zhu multiple-testing hurdles (`adjusted_p_values`,
  `t_hurdle`).
- `credit` counterparty risk and portfolio-loss additions.
  `swap_exposure` simulates a swap's mark-to-market on any one-factor
  short-rate model's paths (`HullWhiteModel` fitted to a curve) by
  Longstaff-Schwartz regression, `regression_mark_to_market`, or by
  closed-form revaluation, and returns an `ExposureProfile` (EPE, ENE,
  PFE). `pathwise_discount` discounts along the paths. On top of it:
  `unilateral_cva` and the swaption-strip `swap_cva_swaptions` (Brigo
  and Masetti), `bilateral_cva` with `BilateralCVAResult` (Gregory), and
  `funding_valuation_adjustment` with `FVAResult` (Burgard and Kjaer).
  `asb_loss_distribution` and `utils.loss_recursion` give the
  Andersen-Sidenius-Basu recursion for heterogeneous pools.
  `base_correlation`, `base_correlation_spread`, `compound_correlations`
  and `BaseCorrelationResult` handle base correlation.
  `visualizers.plot_exposure_profile`. New history entries and gallery
  examples: *Andersen, Sidenius and Basu: the loss recursion* (2003),
  *Base correlation* (2004), *Brigo and Masetti: unilateral CVA from
  exposure profiles* (2005), *Gregory: bilateral CVA and DVA* (2009) and
  *Burgard and Kjaer: funding valuation adjustment* (2011).
- `time_series`: the Kalman filter and RTS smoother (`kalman_filter`,
  `KalmanResult.smooth`, `fit_local_level`); Sims's VAR (`fit_var`,
  `simulate_var`, and `VectorAutoregressionResult` with impulse
  responses, FEVD and forecasts);
  Johansen's cointegration rank test (`johansen_test`); Hamilton's
  Markov-switching model (`hamilton_filter`, `kim_smoother`,
  `fit_markov_switching`, `simulate_markov_switching`); the KPSS
  stationarity test (`kpss_test`); Engle's DCC-GARCH (`fit_dcc`,
  `dcc_correlations`, `dcc_loglikelihood`, `simulate_dcc`); and Corsi's
  HAR model (`fit_har`, `har_regressors`). Seven new history entries
  (1960, 1980, 1988, 1989, 1992, 2002, 2009), each with its own gallery
  example, including a new `state_space` gallery section.
- `risk`: Rouvinez's exact delta-gamma VaR (`delta_gamma_var`,
  `delta_gamma_cdf`, `delta_gamma_cumulants`), Bangia-Diebold-Schuermann-
  Stroughair liquidity-adjusted VaR (`liquidity_adjusted_var`,
  `exogenous_liquidity_cost`), Acerbi-Szekely expected-shortfall
  backtests (`acerbi_szekely_z1`, `acerbi_szekely_z2`,
  `acerbi_szekely_test`, `ESBacktestResult`) and the FRTB 97.5% ES with
  liquidity horizons (`frtb_expected_shortfall`, `frtb_es_cascade`,
  `LIQUIDITY_HORIZONS`, `FRTBResult`, `visualizers.plot_frtb_cascade`).
  Four new history entries (1997, 1999, 2014, 2016) and gallery examples.
- `market_microstructure` order flow and transient impact:
  `simulate_hawkes`, `hawkes_intensity`, `hawkes_loglik`,
  `hawkes_compensator`, `fit_hawkes` and `HawkesResult` (exponential-kernel
  Hawkes process); `simulate_cst`, `cst_mid_up_probability` and
  `OrderBookResult` (the Cont-Stoikov-Talreja order book);
  `order_flow_imbalance` and `fit_ofi_impact`; `obizhaeva_wang`,
  `transient_impact_cost` and `TransientImpactResult`. New history
  entries and gallery examples *Hawkes: self-exciting order flow* (1971),
  *Cont, Stoikov and Talreja: a Markov limit order book* (2010),
  *Obizhaeva and Wang: transient impact and a refilling book* (2013) and
  *Cont, Kukanov and Stoikov: order-flow imbalance* (2014).
- `portfolio`: Rosenberg's (1974) fundamental factor risk model
  (`fundamental_factor_covariance`, `FactorRiskResult`), Goldfarb and
  Iyengar's (2003) robust portfolio selection (`robust_portfolio`,
  `RobustPortfolioResult`) and Gârleanu and Pedersen's (2013) dynamic
  trading with transaction costs (`garleanu_pedersen`,
  `DynamicTradingPolicy`). Three new history entries and gallery
  examples.
- `options`: Bjerksund and Stensland's (1993) American approximation
  (`bjerksund_stensland`), Haug, Haug and Lewis's (2003) prices with
  discrete cash dividends (`discrete_dividend_price`), Castagna and
  Mercurio's (2007) vanna-volga FX smile (`fx_pillar_strikes`,
  `vanna_volga_price`, `vanna_volga_smile`), Jäckel's (2015) "Let's be
  rational" implied volatility (`implied_volatility(method="jaeckel")`,
  `normalised_black_call`, `normalised_implied_volatility`), and the
  normal implied volatility for negative futures prices
  (`bachelier_implied_volatility`). Five new history entries (1993,
  2003, 2007, 2015, 2020), each with a gallery example.
- `interest_rate_models`: Hull and White's (1994) trinomial tree
  (`hull_white_tree`, a `TrinomialRateTree`), callable bonds on it
  (`callable_bond_price`, a `CallableBondResult`) and the option-adjusted
  spread (`option_adjusted_spread`); mortgage pass-throughs valued over
  short-rate paths with Richard and Roll's (1989) prepayment model
  (`richard_roll_cpr`, `mbs_price`, an `MBSResult`); and Lyashenko and
  Mercurio's (2019) forward market model for backward-looking rates
  (`fmm_caplet_price`, `simulate_fmm_rate`, `fmm_volatility_decay`).
  Three new history entries and gallery examples, including a new `mbs`
  gallery section.
- `fixed_income` Treasury bond futures: `conversion_factor` (the CBOT
  price at a 6% notional yield, maturity rounded to quarters or months),
  `cheapest_to_deliver` returning a `DeliveryBasis` (delivery costs and
  the cheapest bond) and `treasury_futures_price` (the cheapest bond's
  forward price per conversion factor). A new gallery example and
  history entry (1977) show the cheapest to deliver switching at the
  notional yield.
- `fixed_income` mortgage pass-throughs: `psa_cpr`,
  `single_monthly_mortality` and `mbs_cash_flows`, returning
  `MBSCashFlows` (interest, scheduled principal, prepayment, weighted
  average life, `as_cash_flows()` for pricing), and
  `plot_mbs_cash_flows`, with a gallery example and history entry (1985)
  on the PSA benchmark.
- `fixed_income.MultiCurve`, `bootstrap_forecast_curve` and
  `bootstrap_multi_curve`: an OIS discount curve and a separate LIBOR
  forecasting curve stripped from par OIS and par swap rates, with
  forward index rates, swap rates and swap values on the pair
  (Piterbarg, 2010). A gallery example and history entry (2010) show
  that par swaps agree on either curve set while off-market swaps do
  not.
- `fixed_income.compounded_in_arrears` (an overnight rate such as SOFR
  compounded over a period, with the ARRC lookback, observation-shift
  and lockout conventions, returning a `CompoundedRate`) and
  `overnight_forwards` (the overnight rates a curve implies), with a
  gallery example and history entry (2018).

- `stochastic_volatility.SSVISurface` and `ssvi_surface`: Gatheral and
  Jacquier's arbitrage-free SSVI surface, its no-arbitrage conditions and
  its raw SVI slices, fitted with the conditions built in. Rough
  volatility in `stochastic_volatility`: `RoughBergomiModel`,
  `rough_bergomi_paths` and `rough_bergomi_mc_price`, the Volterra
  kernels `volterra_covariance`, `volterra_paths` (exact Cholesky, the
  hybrid scheme and the Riemann sum) and `hybrid_scheme_variance`,
  `fbm_paths` (Davies-Harte), `volatility_roughness`, and
  `visualizers.plot_roughness`. New history entries and gallery examples
  *Gatheral and Jacquier's arbitrage-free SSVI surfaces* (2014), *Bayer,
  Friz and Gatheral's rough Bergomi* (2016), *Bennedsen, Lunde and
  Pakkanen's hybrid scheme* (2017) and *Gatheral, Jaisson and Rosenbaum:
  volatility is rough* (2018).
- `pde_methods.CraigSneydADI`, `pde_methods.HestonADI` (with
  `heston_spot_grid`, `heston_variance_grid` and `HestonPDEResult`) and
  `pde_methods.adi_march`, the Douglas, Craig-Sneyd, modified
  Craig-Sneyd and Hundsdorfer-Verwer splittings for two-factor PDEs
  with a mixed derivative, plus `difference_matrices` and
  `visualizers.plot_heston_surface`. New history entries and gallery
  examples *Craig-Sneyd ADI with mixed derivatives* (1988) and *In 't
  Hout-Foulon: the Heston PDE by ADI* (2010).
- `monte_carlo.lmm_swaption_greeks` and `LMMGreeks`: Giles and
  Glasserman's adjoint Greeks (2006). The deltas and vegas of a
  swaption book to every forward rate of a LIBOR market model come from
  one backward sweep, with forward-mode tangents for comparison. New
  history entry and gallery example *Giles-Glasserman adjoint Greeks*.
- `benchmarks/bench_njit.py` and a *Benchmarks* docs page: every `@njit`
  workload (trees, PSOR, Brennan-Schwartz, Longstaff-Schwartz, the SDE
  integrators) timed with and without numba, with a check that both give
  the same numbers.
- `options.bump_greek`: the Greek of any pricer in any named input, or in
  an attribute of a model argument such as `"model.v0"`, by central
  differences. Pairs of names give second and cross derivatives, and
  Monte Carlo pricers that take `rng` are bumped on common random
  numbers, so Heston vega in `v0` or a Monte Carlo barrier delta is one
  call. `finite_difference_greeks` is now built on it.
- Five cross-domain tutorials in `docs/source/tutorials/`, each linked
  from the galleries of the domains it uses: *One option, five pricers*
  (the Black-Scholes formula, trees, Monte Carlo, Crank-Nicolson and
  Fourier pricing agree, then check each other on an American put), *The
  yield curve, end to end* (bootstrapping, Nelson-Siegel, swaps,
  Hull-White and caps), *Where the smile comes from* (implied volatility,
  Heston, the Breeden-Litzenberger density, jumps and SABR), *From returns
  to risk* (stylized facts, GARCH, VaR backtests, expected shortfall and
  extreme value theory) and *Credit risk, from one firm to a portfolio*
  (Merton, KMV, hazard rates and CDS, Vasicek, Li's copula and the Basel
  formula).
- `notebooks/quickstart.ipynb`, a ten-minute tour of six domains, with a
  Colab badge in the README and the docs. It also ships in the JupyterLite
  site, so the "Try it online" badge opens on it.
- `financialkit/tests/test_history_conventions.py`: every history page has
  at least 15 dated entries in chronological order, each with an
  *Implementation:* line and a minigallery, and every gallery example
  exists and is linked from exactly one entry.
- `financialkit/options/tests/test_reference_values.py`: seven published
  values from Haug, 2nd ed., Sections 1.1.1-1.1.3 and Chapter 2 (Black-Scholes,
  Merton, Black-76, delta, gamma, theta and rho), each held to half a unit
  in the last printed digit.
- `stochastic_volatility.fourier_calibrate`: fits any
  `CharacteristicFunctionModel` (Heston, Bates, Merton, variance gamma,
  Kou) to option prices or implied volatilities across strikes and
  maturities by vega-weighted `scipy.optimize.least_squares`, returning a
  `FourierCalibration` (fitted model, parameters, residuals, RMSE in vol
  points), with default bounds per model in `CALIBRATION_BOUNDS`, fixed
  parameters, and `plot_calibration`. A new gallery example under the
  Heston breakthrough, *Calibrating Heston to a smile*, refits a synthetic
  surface exactly and with a quarter of a vol point of noise.
- `interest_rate_models.hull_white_calibrate` fits Hull-White's `a` and
  `sigma` to Black cap or swaption volatilities (priced in the model by
  `model_cap_floor_price` and `jamshidian_swaption`), and
  `g2pp_calibrate` fits G2++ to caps, both by vega-weighted
  `scipy.optimize.least_squares` returning a `ShortRateCalibration`
  (fitted model, parameters, market and model vols and prices, RMSE).
  `model_implied_vols` quotes a model's caps or swaptions as Black
  volatilities. A new gallery example under the Hull-White breakthrough,
  *Hull-White calibrated to caps and swaptions*, recovers a known model
  exactly and from noisy cap quotes.
- `interest_rate_models.g2pp_swaption` prices European swaptions in G2++
  by Brigo and Mercurio's one-dimensional integral (Theorem 4.2.3), with
  Gauss-Hermite quadrature over the narrower factor. `g2pp_calibrate`
  takes `kind="swaption"` and `tenors`, and `model_implied_vols` quotes
  G2++ swaptions. A new gallery example under the G2++ breakthrough,
  *G2++ calibrated to a swaption grid*, recovers all five parameters
  where Hull-White misses by half a vol point.
- `exotic_options.longstaff_schwartz`, `barrier_option_mc`,
  `arithmetic_asian_mc` and `lookback_option_mc` take `paths=` (spot paths
  from any model, on the grid `t_i = iT/n`) and `discount=` (discount
  factors `D(0, t_i)`, one row for all paths or one row per path), so
  they price under Heston, local volatility or jumps. Without them the
  pricers simulate GBM as before, with the same prices to the last bit.
  `sigma` is now optional: it is needed only for GBM paths, the Brownian
  bridge of continuous monitoring and the Asian geometric control, which
  by default is used on GBM paths only. A new gallery example under the
  Longstaff-Schwartz breakthrough, *Longstaff-Schwartz on Heston paths*,
  prices an American put on Andersen QE paths.
- `stochastic_volatility.SVISurface`, an implied-volatility surface of raw
  SVI slices in log-forward-moneyness, joined linearly in total variance,
  and `svi_surface`, which fits one from quotes. It checks Gatheral and
  Jacquier's static arbitrage (`butterfly_arbitrage` through
  `durrleman_g` plus Lee's wing bound, `calendar_arbitrage`,
  `is_arbitrage_free`), prices at its implied vols, and builds a Dupire
  local volatility from them (`SVISurface.local_vol`, which feeds its call
  prices to `dupire_local_vol`). A test solves Dupire's forward equation
  on that local volatility and gets back the surface's vols on a Heston
  smile to 0.1 vol points.
- Coupon schedules and business days in `time_value`: `coupon_schedule`
  (start, end, frequency, short or long front or back stub, the
  end-of-month rule) returns a `Schedule` with unadjusted and adjusted
  dates and SIA quasi-coupon periods; `adjust_date` (following, modified
  following, preceding, modified preceding), `is_business_day` and
  `settlement_date` (T + n) work on Monday-Friday plus a user-supplied
  holiday array, through `numpy.busday_offset`; `add_months` and
  `is_end_of_month` do the month arithmetic. `fixed_income.bond_cash_flows`,
  `bond_price`, `clean_price` and `bond_accrued_interest` take
  `schedule=`, `settlement=` and `day_count=`, and price, accrue and
  discount odd first and last coupons the SIA way; `swap_annuity`,
  `swap_rate`, `interest_rate_swap_value` and `currency_swap_value` take
  `schedule=`, `valuation_date=` and `day_count=` and pay on the adjusted
  dates. Existing calls are unchanged.

### Changed

- Seven gallery examples run on real data instead of samples from the
  model they illustrate. Mandelbrot's fat tails (excess kurtosis 16 and
  a five-sigma day about once a year), the Hill estimator (tail index
  near 3 on both sides) and GARCH(1,1) (persistence 0.98 since 2000) use
  the daily CRSP market. Shiller's excess volatility uses his 1871-1979
  data (price 4.1 times as volatile as p*). The equity premium puzzle
  measures an 8.6% real premium for 1927-2025. Fama-French tests the 25
  size-value portfolios: GRS rejects the CAPM but not the three factors
  on 1963-1991. Yield-curve PCA uses monthly Treasury changes (three
  components explain 98.7%).
- The five tutorials are executed `.. plot::` blocks, and a test checks
  every printed result in them and in the quickstart notebook against the
  output written under it, so a wrong number fails the tests.
- CLAUDE.md sets the bundled-data policy. Shiller's stock-market data,
  FRED Treasury yields and Ken French's factors will ship as small CSVs
  under `financialkit/datasets/`. They are written by
  `scripts/fetch_datasets.py`, which only the author runs, offline from
  CI. Examples where synthetic data would be circular will use them.
- The Euler-Maruyama and Milstein kernels and the SDE drift and diffusion
  callbacks in `monte_carlo`, `interest_rate_models` and
  `stochastic_volatility` are no longer `@njit`-compiled. Each step is
  vectorised over paths, so with numba they were no faster (0.8-0.9x) and
  cost up to a second of compilation on the first call. Callbacks can now
  be any callable, including closures and bound methods; existing `@njit`
  callbacks keep working, and `integrators.njit` stays exported. numba now
  compiles only the `pde_methods` solvers.
- `options.implied_volatility` now inverts every price at once by a
  safeguarded Newton iteration on the log price (Manaster-Koehler start,
  bisection fallback), about 50 times faster than the per-element
  `brentq` loop on a 100 × 100 surface and agreeing with it to
  `DEFAULT_ATOL`. The loop is kept as `method="brentq"`.
- `credit` prices off any `fixed_income` discount curve. The rate `r`
  of `cds_legs`, `cds_par_spread`, `cds_upfront`, `upfront_to_spread`,
  `bootstrap_hazard_curve`, `tranche_spread`, `jarrow_turnbull_zero`,
  `jlt_risky_zero`, `risky_zero_coupon`, `duffie_singleton_cir_zero`,
  `merton_model`, `merton_calibrate` and `kmv_iterate` may now be a
  `YieldCurve`, so a CDS can be bootstrapped on a bootstrapped Treasury
  curve. A number still gives the flat-rate result, to within a few ulps.
  The structural models use the zero rate to the debt's maturity, which
  is exact for deterministic rates. Leland's perpetual-debt model keeps a
  flat rate by construction.
- CI's no-numba job fails below 100% line coverage
  (`--cov-fail-under=100`) and prints any missed lines. Real tests cover
  every line that was missed before: input validation, bracket expansion
  in the root finders, reprs, and the fallbacks for a singular Hessian or
  an infeasible program. The unused `VolatilityModel._vector` is removed.
- The `stochastic_volatility` gallery examples end in `assert`s of the
  result each one illustrates (Fang and Oosterlee's, Kou's and Andersen's
  published prices, Fourier methods agreeing with each other, smile skew
  and curvature moving the way each parameter says, simulations within
  three standard errors of the closed forms), so a wrong number fails the
  docs build. Every gallery example in the package now asserts its claim.
- The `time_series` gallery examples end in `assert`s of the result each
  one illustrates (tests at their size, estimates within three standard
  errors of the truth, fat tails and leverage asymmetry), so a wrong number
  fails the docs build.
- The `credit` gallery examples end in `assert`s of the result each one
  illustrates (the CreditMetrics BBB bond's 107.09 and 2.99, Hull's
  Example 24.3, the Basel Committee's corporate risk weights, Leland's 75%
  leverage, bootstrapped curves repricing their quotes, simulations within
  four standard errors of the closed forms), so a wrong number fails the
  docs build.
- The `portfolio` gallery examples end in `assert`s of the result each
  one illustrates (He and Litterman's equilibrium returns, the numerical
  frontier on Merton's hyperbola, Tobin separation, equal risk
  contributions, the most-diversified portfolio's equal correlations,
  simulations within three standard errors of the closed forms), so a
  wrong number fails the docs build.
- The `asset_pricing` gallery examples end in `assert`s of the result
  each one illustrates (the security market line, GRS size and power,
  Lucas's Euler equation, Mehra and Prescott's 0.35% premium, the
  Hansen-Jagannathan bound, Fama-French and Carhart alphas,
  Harrison-Kreps state prices, Modigliani-Miller, Shiller's bound), so a
  wrong number fails the docs build.
- The test suite treats every warning as an error
  (`filterwarnings = ["error"]` in `pyproject.toml`), with numba and
  without it. Nothing needed allow-listing.
- The minimum scipy is now 1.11 (was 1.10). scipy 1.10's
  non-central chi-squared and binomial functions warn on overflow in
  ranges the CEV and credit-portfolio models reach.
- CI tests on Windows, and on Python 3.10 with every dependency at the
  minimum version `pyproject.toml` allows.
- The docs build runs without numba and fails when a gallery example takes
  more than 5 s (`docs/check_gallery_times.py`). A weekly workflow,
  `linkcheck.yml`, checks the docs' external links.
- `options`, `exotic_options` and `interest_rate_models` validate their
  inputs through one module, `financialkit/_validation.py`, so a bad input
  raises the same `ValueError` everywhere, naming the argument and its
  first bad entry (`"sigma must be non-negative, got -0.1"`,
  `"rho must lie in (-1, 1), got 1"`,
  `"kind must be 'cap' or 'floor', got 'collar'"`). Messages that
  listed several arguments at once now name the one that failed.

### Fixed

- `credit.expected_tranche_loss` and `one_factor_loss_distribution` on a
  finite pool no longer raise `OverflowError` at high correlation. A
  conditional default probability near 1e-307 made scipy's binomial
  overflow; probabilities below 1e-300 are now treated as 0.
- `RateTree.bond_option` raised `KeyError` for an `option_type` other than
  `"call"` or `"put"`; it now raises `ValueError` like every other pricer.
- `hull_white_price` lost its correction to Black-Scholes when `xi**2 * T`
  was just above `1e-4`: the closed-form moments of the average variance
  cancel catastrophically there, which halved the correction at
  `xi = 0.01`. Below `xi**2 * T = 0.1` the moments now come from their
  twelve-term Taylor series.
- `black_derman_toy_tree` brackets each step's median rate exactly, within
  `sigma sqrt(dt) i` of the forward in log terms, instead of widening a
  guess in a loop. A non-positive forward rate, which the lognormal tree
  cannot fit, now raises `ValueError`. Before, it looped forever.
- `monte_carlo.randomized_qmc_estimate(..., sequence="halton")` no
  longer returns one scrambling repeated `n_replications` times, with a
  zero standard error, on scipy older than 1.15.
- `options.implied_volatility` no longer loops forever at `T = 0`: it
  raises `ValueError` for any `T <= 0`, caps its root bracket at
  `sigma = 1e4`, and raises a clear `ValueError` when a price is too close
  to a no-arbitrage bound for its volatility to be resolved.
- The closed-form pricers and Greeks of `options`, `exotic_options` and
  `interest_rate_models` return their limits at `T = 0`, `sigma = 0`,
  `K = 0` and negative rates, or raise `ValueError`, instead of `nan` or a
  `RuntimeWarning`. With no volatility or no time left, barriers,
  lookbacks, Asians, one-touches, compounds and Stulz options are worth
  the payoff of the riskless path `S e^{bt}`;
  `black_scholes_greeks` gives the limits (at the money: delta
  `phi e^{-qT}/2`, gamma `+inf`); `options.black_d1_d2` returns `+-inf`
  or 0 when `sigma sqrt(T) = 0`; `bivariate_normal_cdf` accepts infinite
  limits; `compound_option_price` accepts `T1 = 0` and
  `zero_coupon_bond_option` `T = 0`. Negative spots, strikes, maturities
  and volatilities raise `ValueError` (Bachelier still accepts negative
  spots and strikes), and so does Black-76 on the negative forwards that
  negative rates produce in caps and swaptions, instead of returning
  `nan` or a negative price.
- `exotic_options.gap_option_price` returned minus the gap put's value;
  the put now pays `K2 - S_T`, and gap put-call parity is tested.

### Deprecated

- Public names that meant two different things are split, each docstring
  pointing to its counterpart; the old names still work but emit a
  `DeprecationWarning`:
  `fixed_income.accrued_interest` (year fractions) is now
  `fixed_income.bond_accrued_interest`, leaving
  `time_value.accrued_interest` (dates and a day count);
  `exotic_options.realized_variance` (annualized, from prices) is now
  `exotic_options.annualized_realized_variance`, leaving
  `time_series.realized_variance` (a sum of squared returns); and
  `fixed_income.plot_cash_flows` is now
  `fixed_income.plot_discounted_cash_flows`, which draws its bars with
  `time_value.plot_cash_flows`.
- The API review before 1.0 (ROADMAP C10) gives each parameter and
  result field one spelling, recorded in CONTRIBUTING.md under "Public
  API conventions". The old spellings still work and emit a
  `DeprecationWarning`. Parameters: `n_sims` is now `n_paths`
  (`credit.simulate_migrations`, `creditmetrics_portfolio`,
  `simulate_one_factor_defaults`, `time_series.simulate_dickey_fuller`);
  `n_exercise` is now `n_steps` (`exotic_options.longstaff_schwartz`);
  `nobs` is now `n_obs` (`time_series.mackinnon_critical_values`,
  `bartlett_band`); `rtol` is now `tol` (`credit.kmv_iterate`);
  `ci_level` is now `confidence` (`risk.monte_carlo_var`); and
  `frequency` is now `freq` in the nine `credit` CDS and tranche
  functions. Result fields: `nobs` is now `n_obs` in twelve
  `time_series` results; `pvalue` is now `p_value` (`TestResult`,
  `VarianceRatioResult`, `StationarityResult`), with
  `VarianceRatioResult.p_value_robust` and
  `ReturnStatistics.jarque_bera_p_value`; `success` is now `converged`
  (`stochastic_volatility.FourierCalibration`,
  `interest_rate_models.ShortRateCalibration`); and
  `pde_methods.LCPSolution.iterations` is now `n_iter`. The old field
  names are readable but are not accepted by the dataclass constructors.

## [0.1.0] - 2026-10-10

The first release.

### Added

- Repository scaffold, structurally cloned from mathematicskit:
  `pyproject.toml` (numpy, scipy and matplotlib, with numba in the `fast`
  extra), CI, docs and release workflows, pre-commit, the Sphinx site
  with per-domain hubs, galleries, history pages and JupyterLite, and the
  root docs. The README and the docs index state that the package is
  educational and research software, not investment advice.
- `financialkit.constants`: `TRADING_DAYS_PER_YEAR`,
  `CALENDAR_DAYS_PER_YEAR`, the `DayCount` enum and the default
  tolerances `DEFAULT_ATOL`, `DEFAULT_RTOL` and `DEFAULT_MAX_ITER`.
- `financialkit._jit`: an optional numba `njit` that falls back to a
  no-op decorator, so the package runs under Pyodide.
- `financialkit.integrators`, the shared SDE module: `euler_maruyama` and
  `milstein`, exact `gbm_paths` and `ou_paths`, `brownian_increments`,
  `correlated_increments`, `antithetic_increments`, `sobol_increments`
  and `brownian_bridge`, and `strong_convergence` and
  `weak_convergence` (`ConvergenceResult`).
- 14 domain subpackages. Each has its own tests, visualizers, gallery
  and a history page of at least 15 dated breakthroughs, each linked to
  its own gallery example (239 examples in all):
  - `time_value`: compounding and equivalent rates, present value and
    NPV, annuities, perpetuities and life annuities, IRR (with
    `MultipleIRRWarning`) and MIRR, amortization and the APR, sinking
    funds, and day counts and accrued interest for every `DayCount`.
  - `fixed_income`: bond prices, yields and STRIPS; Macaulay, modified,
    Fisher-Weil, effective and key rate durations and convexity;
    Redington immunization; bootstrapped, McCulloch spline,
    Nelson-Siegel and Svensson curves; yield-curve PCA; and interest
    rate and currency swaps.
  - `options`: Bachelier, Black-Scholes-Merton, Black-76 and
    Garman-Kohlhagen; parity and no-arbitrage bounds; the Greeks and
    delta hedging; implied volatility, the smile and the
    Breeden-Litzenberger density; CRR, Rendleman-Bartter, Leisen-Reimer
    and trinomial trees; and Barone-Adesi-Whaley.
  - `interest_rate_models`: Merton, Vasicek, CIR and affine short rates;
    Ho-Lee, Hull-White, G2++ and Black-Derman-Toy fitted to the curve;
    Jamshidian bond options and swaptions; Black caps, floors and
    swaptions; and HJM.
  - `monte_carlo`: crude Monte Carlo with confidence intervals;
    stratification, importance sampling, antithetic, control and
    moment-matched variates and Latin hypercubes; Halton and Sobol QMC
    and discrepancies; pathwise and likelihood-ratio Greeks; and
    multilevel Monte Carlo.
  - `exotic_options`: single, double and discrete barriers, binaries
    and one-touches, geometric and arithmetic Asians, lookbacks,
    variance swaps, compound, forward-start and chooser options,
    Margrabe, Stulz and quanto options, and Longstaff-Schwartz.
  - `pde_methods`: explicit, implicit and Crank-Nicolson schemes with
    Rannacher smoothing, the Thomas algorithm, CFL and von Neumann
    stability, Richardson extrapolation, Tavella-Randall grids, the
    heat-equation transform, PSOR, Brennan-Schwartz, the penalty method
    and Peaceman-Rachford ADI.
  - `stochastic_volatility`: CEV and Dupire local volatility,
    Hull-White and Heston, Merton, Kou, Bates and variance gamma,
    Carr-Madan FFT, Lewis and COS pricing, SABR, SVI and Lee's moment
    formula, and full-truncation and QE simulation.
  - `time_series`: stylized facts and the Hill estimator, random-walk
    tests, Yule-Walker and ARMA, Dickey-Fuller and Engle-Granger, the
    Hurst exponent, ARCH, GARCH, EGARCH, GJR and EWMA, and range and
    realized volatility.
  - `risk`: historical, parametric, Cornish-Fisher, Monte Carlo, hybrid
    and filtered VaR, expected shortfall and coherence, Euler
    allocation, extreme value theory, VaR backtests, copulas and tail
    dependence, safety first and drawdowns.
  - `portfolio`: Markowitz, Tobin and Merton, Kelly, CPPI, single-index
    and Ledoit-Wolf covariances, Black-Litterman and resampling, CVaR
    optimization, maximum diversification, risk parity and HRP, and
    performance ratios.
  - `asset_pricing`: dividend discount and Gordon growth,
    Modigliani-Miller and Shiller's variance bound, the CAPM, Jensen's
    alpha, the zero-beta CAPM and the GRS test, Fama-MacBeth, the APT
    and factor models, Lucas's tree, the consumption CAPM and the
    equity premium puzzle, and the Hansen-Jagannathan bound.
  - `credit`: Altman's Z-score, Merton, Black-Cox and Leland, the KMV
    distance to default, hazard curves, CDS pricing and bootstrapping,
    CreditMetrics and generator matrices, Vasicek's large pool,
    CreditRisk+, Li's copula, the Basel IRB formula and CDO tranches.
  - `market_microstructure`: quoted, effective and realized spreads,
    the Roll and Corwin-Schultz estimators, Lee-Ready classification,
    Garman, Ho-Stoll and Avellaneda-Stoikov dealers, Kyle and
    Glosten-Milgrom, Hasbrouck's VAR, PIN and VPIN, square-root impact,
    Bertsimas-Lo and Almgren-Chriss execution, and Amihud illiquidity.

[Unreleased]: https://github.com/cpoli/financialkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cpoli/financialkit/releases/tag/v0.1.0

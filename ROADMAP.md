# Roadmap

## Where financialkit stands

- **0.1.0** shipped on 2026-10-10 with 14 domains, 236 breakthroughs,
  239 gallery examples, 868 tests at 99% line coverage, 5 tutorials and
  the quickstart notebook.
- **0.2.0** shipped on 2026-10-30 with the work listed under "Done"
  below.
- **The record** of every entry is the history pages in
  `docs/source/history/`. The old per-domain build plans are in git
  history (`git show 681a826:ROADMAP.md`).

Counts today:

| Domain | Breakthroughs | Gallery examples | Latest entry |
|---|---:|---:|---:|
| `time_value` | 16 | 16 | 1998 |
| `fixed_income` | 20 | 21 | 2018 |
| `options` | 21 | 23 | 2020 |
| `interest_rate_models` | 19 | 21 | 2019 |
| `monte_carlo` | 19 | 19 | 2008 |
| `exotic_options` | 17 | 17 | 2001 |
| `pde_methods` | 18 | 18 | 2010 |
| `stochastic_volatility` | 21 | 22 | 2018 |
| `time_series` | 26 | 26 | 2009 |
| `risk` | 21 | 21 | 2016 |
| `portfolio` | 21 | 21 | 2016 |
| `asset_pricing` | 24 | 24 | 2016 |
| `credit` | 22 | 22 | 2011 |
| `market_microstructure` | 20 | 20 | 2014 |

## The 0.1.0 audit

The audit found six weaknesses and a few smaller problems. **All are
resolved**, with the two leftovers noted below.

1. **The examples proved nothing.** No gallery script asserted its
   claim, and the tutorials held pasted output that nothing re-ran.
   - Resolved by B2 (2026-10-21): every gallery script asserts its claim.
   - Resolved by B6 (2026-10-27): the tutorials and the notebook are
     checked line by line.
2. **The domains didn't compose.** `credit` took a flat `r` instead of
   a `CurveLike`, and the Longstaff-Schwartz and exotic Monte Carlo
   pricers simulated their own GBM.
   - Resolved by C1 and C2 (2026-10-21). D11 built XVA on them.
3. **No calibration** beyond SABR and one SVI slice.
   - Resolved by C3-C5 (2026-10-21): Heston, Bates, variance gamma,
     Hull-White and Dupire.
   - G2++ to swaptions followed on 2026-10-27.
4. **No dates.** Bonds and swaps used year fractions and a regular
   `freq`, with no schedules, stubs, rolls or settlement lags.
   - Resolved by C8 (2026-10-21). D1 (2026-10-22) added SOFR in arrears.
5. **Synthetic data only.** The empirical lessons ran on data drawn
   from the model being shown, which is circular.
   - Resolved by E1 and E2 (2026-10-27): `financialkit.datasets`
     bundles Shiller, French and FRED data, and seven examples use it.
   - Left over: the Fama-French five-factor example is still synthetic.
6. **The history pages stopped early** and skipped classics such as
   GMM, regime switching, Johansen, DCC-GARCH and the Kalman filter.
   - Resolved by D1-D3, D7, D9 and D10 (2026-10-22).
   - Left over: `time_value` still ends before 2000.

**Smaller problems**, all resolved on 2026-10-21:

- `implied_volatility` looped forever at `T = 0` and solved one element
  at a time (A2, C9).
- Pricers returned `nan` silently on invalid input (A3, C7).
- `accrued_interest` and `realized_variance` each had two clashing
  implementations (A4).
- Warnings didn't fail the tests (B3).
- Nothing checked that every example is linked from one breakthrough (B1).

## How to work from this file

- **One item per session**, from a prompt such as "Do ROADMAP item C10".
  Each item says what to change and when it is done.
- **Order:** an item marked **After** waits for the items it names.
  Anything else can be done in any order.
- **An item is done when:**
  - `ruff check .`, `ruff format --check .`, `mypy financialkit`,
    `MPLBACKEND=Agg pytest -q`, the same with `NUMBA_DISABLE_JIT=1`, the
    doctests and `cd docs && make html` all pass;
  - CHANGELOG.md has a line for it under `## [Unreleased]`;
  - it is moved from "Open work" to "Done" here, with the date.
- **CLAUDE.md applies to everything:**
  - no new dependencies and nothing on the network;
  - a domain imports only from earlier domains;
  - every model cites its source;
  - every new breakthrough has a dated history entry, its own gallery
    example and a `systems/` implementation;
  - a history page never falls below 15 entries;
  - the public API stays backward compatible: a renamed name keeps the
    old one as a deprecated alias.
- **Sessions may run in parallel.** CHANGELOG.md, this file, the table
  above and `docs/source/conf.py` are shared. Re-read them just before
  editing, and change only your item's lines.

## Open work

At a glance (all can start now except G3):

| Item | What | Who |
|---|---|---|
| A1 | Commit or revert the pending working tree | author decides |
| B5 | Test the dependency floors and Windows | |
| B7 | Textbook reference tests, one domain per session | |
| B8 | Property tests for the pricers | |
| F1 | Exercises, one domain per session | |
| F2 | Two tutorials: two-curve swaps, counterparty risk | |
| F3 | A calibration tutorial | |
| G2-G3 | conda-forge, 1.0 | author |

Sections D (curriculum gaps) and E (data) have nothing open.
`time_value`, `exotic_options` and `monte_carlo` have no pressing
curriculum gaps.

### A. Hygiene and bugs

**A1. Commit or revert the pending working tree.**

- **Problem:** `README.md`, `docs/source/conf.py`,
  `docs/source/index.rst` and `pyproject.toml` hold uncommitted SEO work
  (sitemap, OpenGraph, a "Related toolkits" rubric). They also put
  shields.io badges back after commits `df90ed0`, `c210460` and
  `6ffbd6f` moved them to badgen.
- **To do:**
  - decide which badge host wins;
  - check that `sphinx-sitemap` and `sphinxext-opengraph` are in the
    `docs` extra;
  - commit or drop the work.
- **Done when** `git status` is clean and the docs build passes.

### B. Quality gates

**B5. Test the supported dependency range and Windows.**

- **Problem:** `pyproject.toml` promises numpy 1.24, scipy 1.10 and
  matplotlib 3.6, but CI installs the latest versions.
- **To do:**
  - add a Python 3.10 job with those floors pinned;
  - add a `windows-latest` row;
  - fix what fails, or raise the floors.
- **Done when** both jobs pass.

**B7. Textbook reference tests, one domain per session.**

- **Problem:** only 29 of 104 test files cite Hull or Haug.
- **To do:** for the domain named in the prompt, add
  `tests/test_reference_values.py` with at least five published numbers
  from Hull, Haug, Glasserman, Brigo-Mercurio, Tsay,
  McNeil-Frey-Embrechts or the original paper. Cite the edition and the
  table or example number.
- **Done when** the domain has five such tests and each fails if the
  model is perturbed.
- **Domains done:** `options`.

**B8. Property tests for pricers.**

- **Problem:** `hypothesis` is a test dependency but only 3 files use it.
- **To do:** add property tests for the no-arbitrage invariants:
  - a call price increases in `S` and `T`, and is convex and decreasing
    in `K`;
  - put-call parity holds for every European pricer (closed form, trees,
    PDE, FFT, COS);
  - an American price is at least the European one.
- **Done when** the properties run on every European pricer.

### C. Architecture and composability

Nothing open: C1-C10 are done.

### D. Curriculum gaps

Nothing open. New entries join their domain's history page in
chronological order, each with its own gallery example and a `systems/`
implementation.

### E. Data

Nothing open: E1 and E2 are done.

### F. Course material

**F1. Exercises, one domain per session.**

- **To do:** for the domain named in the prompt, add
  `docs/source/exercises/<domain>.rst` with five to ten problems:
  - each one built on a gallery example (change a parameter, predict,
    check with financialkit);
  - solutions in collapsed `sphinx-design` dropdowns that run in the
    docs build and are checked as in B6.
- **Done when** the page builds and its solutions are tested.

**F2. Two tutorials on the new material.** Its prerequisites (D1, D3,
D11 and B6) are done.

- *One swap, two curves:* OIS discounting, SOFR compounding and the
  forward market model.
- *Counterparty risk, end to end:* Hull-White exposure, CVA, DVA and FVA.
- **Done when** both are linked from the galleries they use and run in CI.

**F3. A calibration tutorial.** Its prerequisites (C2, C3 and C5) are
done.

- *From quotes to a model:* an SVI surface, local volatility from it,
  Heston fitted to the same quotes, and the exotic prices the two models
  disagree on.

### G. Releases (the author)

- **G2. conda-forge recipe.** A `recipe.yaml` with only `version` in
  `context`, and `name: financialkit` and the PyPI source URL written
  out.
- **G3. 1.0** freezes the public API (see CONTRIBUTING.md). **After**
  all of B (C10 is done).

### H. Scope decisions

- **Out of scope**, because the dependencies are fixed:
  - deep hedging and other neural-network methods, which need a
    deep-learning framework;
  - market-data fetching;
  - wrappers around QuantLib.
- **Candidate domains**, not scheduled. CLAUDE.md forbids adding a
  domain without the author's decision, and each would need 15 or more
  breakthroughs:
  - `commodities`: Gibson-Schwartz (1990), Schwartz's one-, two- and
    three-factor models (1997), Schwartz-Smith (2000), seasonality,
    storage and spark-spread options;
  - `inflation`: Jarrow-Yildirim (2003), zero-coupon and year-on-year
    inflation swaps, inflation caps;
  - `real_options`: McDonald-Siegel (1986), Dixit-Pindyck (1994),
    Trigeorgis's expansion and abandonment options, Myers's
    underinvestment (1977).

## Done

### 0.2.0 (2026-10-30)

- G1 (2026-10-30): released 0.2.0 to PyPI and GitHub, with Zenodo
  version DOI 10.5281/zenodo.23795067 in `CITATION.cff` and the README.
- C10 (2026-10-27): the API review before 1.0, written into
  CONTRIBUTING.md as "Public API conventions". Functions and classes
  each keep their job (a model with shared parameters is a class;
  pricers, estimators and tests are functions that take the model).
  One spelling per parameter: `n_sims`, `n_exercise`, `nobs`, `rtol`,
  `ci_level` and `frequency` became `n_paths`, `n_steps`, `n_obs`,
  `tol`, `confidence` and `freq` in 18 functions; PDE grids keep
  `n_space` and `n_time`. One name per result field: `p_value`, `n_obs`,
  `n_iter` and `converged` in 15 result classes. The old spellings are
  deprecated aliases (`renamed_parameter` and `deprecated_field` in
  `financialkit._deprecation`). All 14 domains now export their
  `plot_*` functions.
- E2 (2026-10-27): `financialkit.datasets` loads Shiller's monthly
  data, French's daily market, monthly factors and 25 size-value
  portfolios, and month-end FRED Treasury yields (about 800 KB of CSV)
  as frozen dataclasses of numpy arrays. `scripts/fetch_datasets.py`
  writes the CSVs with the standard library only, including a minimal
  reader for Shiller's binary `.xls`. Ported to real data: Mandelbrot's
  fat tails, Hill, GARCH, Shiller's excess volatility (ratio 4.1 on
  1871-1979), the equity premium (8.6% real, 1927-2025; the consumption
  moments are still Mehra-Prescott's), Fama-French on the 25 portfolios
  and yield-curve PCA. The Fama-French five-factor example is still
  synthetic.
- G2++ to swaptions (2026-10-27), the remainder of audit point 3:
  `interest_rate_models.g2pp_swaption` implements Brigo and Mercurio's
  one-dimensional swaption integral (2nd ed., Theorem 4.2.3), and
  `g2pp_calibrate` fits swaptions through it. The price agrees with
  payer-receiver parity to 1e-13, with Monte Carlo to within 4 standard
  errors and with Jamshidian under Hull-White when the second factor is
  silenced. A 4 x 3 swaption grid from a known model gives back all five
  parameters to 1e-4. New example
  `short_rate/plot_08_g2pp_swaption_calibration.py`.
- B6 (2026-10-27): the five tutorials are `.. plot::` blocks that the
  docs build runs, and `financialkit/tests/test_course_material.py` runs
  them too and compares each print with the `# ` lines under it. The
  quickstart notebook's prints carry the same lines and its code cells
  run in the same test. All 73 printed lines on the pages and 5 in the
  notebook were still right. Scaling `black_scholes_price` by 1.0001
  fails three tutorials and the notebook; the other two tutorials don't
  use it.
- E1 (2026-10-27): the bundled-data policy is in CLAUDE.md. Shiller's
  stock-market data, FRED constant-maturity Treasury yields and Ken
  French's factors ship as small CSVs under `financialkit/datasets/`.
  French's library states no licence, so only the trimmed factor series
  ship, with credit to French and CRSP. `scripts/fetch_datasets.py` is
  the only code that downloads. The author runs it by hand, offline from
  CI, and commits the CSVs. A README records each file's source, terms
  and retrieval date.
- D11 (2026-10-22): five `credit` breakthroughs, so the page has 22
  entries and ends in 2011. `swap_exposure` values a swap on
  `HullWhiteModel` paths by Longstaff-Schwartz regression of the
  realized cash flows on the short rate and the fixed coupon, and
  discounts along each path. The paths reprice the fitted curve to
  within 4 standard errors. Revalued exactly on the same paths, the
  discounted positive exposure at each payment date matches a Jamshidian
  payer swaption to within 4 standard errors. Over 12 seeds the
  regression matches the same swaptions to within 0.5 standard errors
  of the mean, so it is unbiased. On the payment dates `unilateral_cva`
  agrees with the Brigo-Masetti swaption strip. The full exposure grid
  gives about 10% more, because between payments the exposure still
  includes the coming coupon. With first-to-default weighting,
  `bilateral_cva` makes one side's DVA equal to the other's CVA to
  rounding. `funding_valuation_adjustment` lending at the own credit
  spread gives an FBA equal to the DVA to about lambda dt / 2. The ASB
  recursion matches enumeration of every default pattern exactly and
  Monte Carlo within 4 standard errors. `base_correlation` recovers an
  input skew to 1e-8, and `compound_correlations` finds two roots for a
  3-7% quote and none at three times that quote. The work also found
  and fixed an `OverflowError` in finite-pool `expected_tranche_loss`
  and `one_factor_loss_distribution` at high correlation (scipy's
  binomial at p near 1e-307).
- D10 (2026-10-22): seven `asset_pricing` breakthroughs, each with its
  own gallery example. 1982, Hansen: `gmm` (identity, two-step or
  iterated weighting by `least_squares`, Newey-West `S`, sandwich
  errors and the J test). It gives OLS with White's errors exactly, and
  in the example J follows chi-squared(9) with 7.2% size at 5% over 400
  samples. 1982, Hansen-Singleton: `hansen_singleton` recovers a Lucas
  tree's `(beta, gamma)` within 3 standard errors. A 6% unexplained
  premium forces beta 0.61 and gamma 40. 1988, Campbell-Shiller:
  `log_linear_return`, `campbell_shiller_rho` and
  `campbell_shiller_decomposition`. The linearisation error is 0.5% of
  the return's volatility, and the long-run coefficients differ by one.
  1999, Campbell-Cochrane: `CampbellCochrane` solves the
  price-consumption ratio as one linear system on a surplus grid. It
  reproduces their S-bar 0.057 and S_max 0.094 and a constant 0.94%
  riskless rate, and gives a 6.6% premium with a Sharpe ratio of 0.50 at
  gamma = 2. The Euler equations hold in simulation. 2004, Bansal-Yaron:
  `BansalYaron` solves the log-linear model as fixed points in the
  Campbell-Shiller constants. Simulated Euler equations hold for
  psi = 1.5 and 0.5, the log premium is 4.2% a year and the equity
  volatility 19%, and the premium collapses for psi < 1. 2015,
  Fama-French: `fama_french_five` from `two_by_three_portfolios`. In the
  example GRS rejects three factors (p = 2e-5) but not five, and HML is
  spanned by the other four. 2016, Harvey-Liu-Zhu: `adjusted_p_values`
  (Bonferroni, Holm, BH, BHY) and `t_hurdle`, which reproduces their
  3.78 Bonferroni hurdle for 316 factors. The page now has 24 entries
  and ends in 2016.
- D7 (2026-10-22): seven `time_series` breakthroughs, each with its own
  gallery example. 1960, Kalman: `kalman_filter` (prediction-error
  likelihood, NaN observations skipped) and the RTS smoother match direct
  Gaussian conditioning to 1e-9, and `fit_local_level` recovers the noise
  levels of an efficient price behind microstructure noise. 1980, Sims:
  `fit_var` is equation-by-equation OLS, with Cholesky impulse responses,
  FEVD, forecasts and a stability check; 92% of the true responses fall
  inside 90% bootstrap bands in the example. 1988, Johansen:
  `johansen_test` (trace and max-eigenvalue, no deterministic terms or an
  unrestricted constant, two to five series). Its MacKinnon-Haug-Michelis
  asymptotic critical values agree with a 4000-draw simulation at
  T = 1000 for every row, and the size tests hold at 5%. 1989, Hamilton:
  `hamilton_filter` and `kim_smoother` equal brute-force enumeration of
  all regime paths to 1e-10; `fit_markov_switching` (EM) classifies 97%
  of days. 1992, KPSS: `kpss_test` with the paper's Table 1 and the
  nominal size. 2002, Engle: `fit_dcc` (GARCH per series, then the
  correlation likelihood) and `simulate_dcc`. The DCC correlation's mean
  error is 0.016 against 0.109 for a 60-day rolling window. 2009, Corsi:
  `fit_har` beats an AR(1) out of sample on a three-component volatility
  cascade. The page has 26 entries.
- D8 (2026-10-22): four `risk` breakthroughs. `delta_gamma_var`
  inverts the characteristic function of the quadratic P&L (Rouvinez
  1997) by Gil-Pelaez with QUADPACK's Fourier integrator and matches
  the chi-squared closed form to 1e-9. On a short-option book hedged
  to 90% of its delta, the 99% VaR is 70 delta-normal, 690 by
  Cornish-Fisher, 645 exact and 617 by full revaluation.
  `liquidity_adjusted_var` adds the BDSS spread cost; on a simulated
  market whose spreads widen in sell-offs, market VaR has 100
  exceptions where 22.5 are expected, against 27 for LVaR.
  `acerbi_szekely_z1`/`z2`/`_test` reproduce the -0.70 5% critical
  value of Z2, and catch fat tails behind a correct VaR that Kupiec
  misses (75% power for Z1 against Kupiec's 7%).
  `frtb_expected_shortfall` cascades ten-day 97.5% ES over the five
  liquidity horizons. New examples `var/plot_07`, `var/plot_08`,
  `backtesting/plot_04` and `expected_shortfall/plot_04`; the page has
  21 entries.
- D12 (2026-10-22): `market_microstructure.systems.order_flow` holds
  the exponential Hawkes process (simulated by its cluster
  representation, fitted by maximum likelihood with Ozaki's recursion,
  checked by Ogata's compensator residuals), the Cont-Stoikov-Talreja
  book (exact event-by-event simulation, and the probability that the
  mid moves up next from the queues at a one-tick spread, solved as a
  sparse linear system rather than by the paper's Laplace transforms)
  and Cont-Kukanov-Stoikov order-flow imbalance. `obizhaeva_wang` gives
  the exact discrete optimal schedule (end blocks
  `X / (2 + (N - 1)(1 - e^{-rho T/N}))`), which tends to the paper's
  continuous solution. The simulated book's price moves match the
  predicted probability within four standard errors, and OFI explains
  63% of 200 interval price changes against 20% for trade imbalance. Its
  slope falls faster than the paper's 1/depth when the book deepens,
  and the example says so. The page has 20 entries.
- D9 (2026-10-22): three `portfolio` entries, each with a gallery
  example. `fundamental_factor_covariance` is Rosenberg's (1974)
  fundamental factor model: weighted cross-sectional regressions per
  period, fixed or time-varying exposures, and a `FactorRiskResult` that
  splits a portfolio's variance into factor and specific parts. With 100
  stocks and 120 months its minimum-variance portfolio forecasts its own
  risk to within 4%, where the sample covariance's forecasts 6% and
  delivers 37%. `robust_portfolio` is Goldfarb and Iyengar's (2003)
  worst-case mean-variance problem with a box around the means and
  F-norm balls around the factor loadings, solved by SLSQP on split
  long/short weights; the tests build the worst-case inputs and check
  that they attain the reported moments. `garleanu_pedersen` returns
  Gârleanu and Pedersen's (2013) closed-form policy, which matches value
  iteration on the Bellman equation to 1e-8; in the example it beats
  trading any fixed fraction of the way to Markowitz (0.99 against 0.81
  bp a day). The page has 21 entries.
- D2 (2026-10-22): five `options` entries, each with a gallery example.
  `bjerksund_stensland` (1993) reproduces Haug's 5.2704 and stays below
  a 3000-step tree. `discrete_dividend_price` is Haug-Haug-Lewis (2003)
  by nested Gauss-Legendre quadrature split at the kinks: exact for a
  dividend at expiry, and within Monte Carlo error for three dividends.
  `vanna_volga_price` and `vanna_volga_smile` (Castagna-Mercurio 2007)
  reproduce the pillar quotes. `implied_volatility(method="jaeckel")`
  runs Jäckel's "Let's be rational" (2015) on the whole array, with a
  median round-trip error of 0 and a worst of 3e-15 after two
  iterations on out-of-the-money options. `bachelier_implied_volatility`
  covers the 2020 negative oil futures.
- D3 (2026-10-22): three `interest_rate_models` breakthroughs, each with
  its own example. 1989, Richard-Roll: `richard_roll_cpr` (refinancing
  S-curve, seasoning, seasonality, burnout) and `mbs_price`, a monthly
  pass-through simulated along any one-factor short-rate model with its
  10-year yield as the refinancing rate; with a constant CPR it matches
  the present value on the curve within four standard errors, and on
  Hull-White paths the price is concave around today's rates.
  1994, Hull-White trinomial tree: `hull_white_tree` reproduces Hull's
  three-step example (Sec. 32.7: alpha 3.824/5.205/6.252%, Q and the
  edge probabilities), reprices the curve to 1e-12, and a bond callable
  once converges to Jamshidian's European price (2.5e-5 at 640 steps);
  `callable_bond_price` and `option_adjusted_spread` value Bermudan
  callables and invert for the OAS. The example is
  `curve_fitted/plot_06_callable_bond_oas.py`, since C4 took `plot_05`.
  2019, Lyashenko-Mercurio: `fmm_caplet_price` (Black or Bachelier with
  variance `sigma^2 (T_{j-1} + tau/3)`, or the forward-looking caplet,
  which equals `cap_floor_price`'s), `simulate_fmm_rate` with exact
  variance steps, and `fmm_volatility_decay`; simulation matches the
  caplets within four standard errors, and compounded Hull-White short
  rates recover the curve forward.
- D1 (2026-10-22): four `fixed_income` breakthroughs, each with its own
  gallery example. 1977, Treasury bond futures: `conversion_factor`
  (quarterly or monthly rounding), `cheapest_to_deliver` and
  `treasury_futures_price` reproduce Hull's 1.4623, 1.2199, the
  2.69/1.87/2.12 delivery costs and the 71.79 futures price (9th ed.,
  Sec. 6.2). 1985, PSA: `psa_cpr`, `single_monthly_mortality` and
  `mbs_cash_flows` reproduce Fabozzi's $400m pass-through to the dollar
  in month 1 (2,500,000 interest, 267,535 scheduled principal, 267,470
  prepaid). 2010, Piterbarg: `bootstrap_multi_curve` strips OIS with
  `bootstrap_par_curve` and the forecasting curve node by node
  (`bootstrap_forecast_curve`), returning a `MultiCurve` that reprices
  both input sets to 1e-10 and reduces to the par yield with one curve.
  2018, SOFR: `compounded_in_arrears` with ARRC lookback, observation
  shift and lockout on a weekend-plus-holidays calendar, and
  `overnight_forwards`, whose compounded rate equals the curve's simple
  forward to 1e-12. D3's Richard-Roll entry can build on
  `mbs_cash_flows`.

- D6 (2026-10-22): `stochastic_volatility.SSVISurface` is Gatheral and
  Jacquier's SSVI with power-law or Heston-like curvature, checks their
  calendar (Theorem 4.1) and butterfly (Theorem 4.2) conditions, and
  turns its slices into raw SVI; `ssvi_surface` fits it with the
  no-arbitrage bounds built into the parameters, and the fits pass
  Durrleman's test on every slice. `systems/rough.py` holds the
  fractional kernels, here and not in `integrators`: the
  Bayer-Friz-Gatheral joint covariance for exact Cholesky sampling, the
  hybrid scheme (kappa = 1, FFT convolution) and the Riemann sum, plus
  `RoughBergomiModel`, `rough_bergomi_paths` and
  `rough_bergomi_mc_price`. Davies-Harte `fbm_paths` and
  `volatility_roughness` cover the 2018 entry. Hybrid variance is within
  1% of t^(2H) at H = 0.07 (the Riemann sum loses half of it), and the
  rough Bergomi skew scales as tau^-0.39 against H - 1/2 = -0.43. The
  2018 paper is by Gatheral, Jaisson and Rosenbaum; this item's old
  title said "Jaillet".
- D5 (2026-10-22): `pde_methods.utils.splitting.adi_march` steps a
  two-factor operator split by the Douglas, Craig-Sneyd, modified
  Craig-Sneyd or Hundsdorfer-Verwer scheme. `CraigSneydADI` prices
  two-asset payoffs with it, and on Margrabe's exchange option it
  measures order 1.0 in time for Douglas and for Peaceman-Rachford's
  explicit mixed term against 2.0 for Craig-Sneyd. `HestonADI` solves
  the Heston PDE on in 't Hout and Foulon's non-uniform grids. Against
  Heston's formula its error is about 1.5 cents on a 50 x 25 grid for
  their cases A-D and under half a cent on 100 x 50; on Andersen's
  Feller-violating Case I it is 0.4 cents on 200 x 100. New 1988 and
  2010 history entries and examples `multi_factor/plot_02_craig_sneyd.py`
  and `multi_factor/plot_03_heston_adi.py`; the page has 18 entries.
- D4 (2026-10-22): `monte_carlo.lmm_swaption_greeks` simulates Giles
  and Glasserman's one-factor LIBOR market model (Euler steps in log
  rates under the spot measure) and returns a payer-swaption book's
  price with its delta and vega to every forward rate in an `LMMGreeks`,
  by the adjoint sweep or by forward-mode tangents. The two agree to
  rounding and match central differences on common random numbers to
  1e-7. On their test case (80 quarterly rates, 15 swaptions) the
  forward mode is about 50 times slower than the adjoint, and the gap
  grows linearly in the number of rates. New 2006 history entry and
  example `greeks/plot_03_adjoint_greeks.py`; the page has 19 entries.
- B9 (2026-10-22): `docs.yml` builds without numba and then runs
  `docs/check_gallery_times.py`, which reads `sg_execution_times.rst` and
  fails listing every example over 5 s. A no-numba build with a
  `time.sleep(6)` example failed on exactly that example; the other 242
  pass, the slowest at 2.25 s.
- B11 (2026-10-22): `.github/workflows/linkcheck.yml` runs
  `make linkcheck` every Monday without executing the gallery. A local
  run is green: 69 external links work, and the only two failures were
  false alarms, now skipped in `conf.py` (the JupyterLite `lite/` path,
  which exists only in the html build, and GitHub's JavaScript heading
  anchors). No publisher refused the checker, so `linkcheck_ignore`
  names none yet.
- B10 (2026-10-21): `benchmarks/bench_njit.py` times the CRR and
  trinomial trees, PSOR, Brennan-Schwartz, Longstaff-Schwartz and the
  Euler and Milstein SDE kernels with numba and in a child process where
  `import numba` fails, checks that both give the same numbers, and
  writes the table on `docs/source/benchmarks.rst`. PSOR is about 35x
  faster with numba and Brennan-Schwartz about 10x. The trees and
  Longstaff-Schwartz have no `@njit` kernel, and the SDE kernels, which
  are vectorised over paths, are no faster compiled (0.8-0.9x).
- C8 (2026-10-21): `time_value.coupon_schedule` generates dates with
  short or long front or back stubs and the end-of-month rule;
  `adjust_date`, `is_business_day` and `settlement_date` roll on a weekend
  calendar plus user-supplied holidays. Bonds (`bond_cash_flows`,
  `bond_price`, `clean_price`, `bond_accrued_interest`) and all three swap
  functions accept a schedule. A bond with a short first coupon reproduces
  the SIA formula and Excel's `ODDFPRICE` value of 113.597717 and its
  `ODDFYIELD` of 7.72% on 30/360. A long first coupon matches the SIA
  formula, and ISDA's 1998 ICMA examples (0.5, 0.41096, 0.91576) and
  Hull's Treasury and corporate accruals (3.3696, 3.3889) hold.
- C9 (2026-10-21): `implied_volatility` solves the whole array at once
  by a safeguarded Newton iteration on the log price, started at
  Manaster and Koehler's inflection point and bisecting whenever a step
  leaves its per-element bracket. A 100 × 100 surface takes about 3 ms
  against 155 ms for the `brentq` loop, which stays as
  `method="brentq"`, and the two agree to `DEFAULT_ATOL`; the benchmark
  is in `options/tests/test_implied_vol.py`. Newton is the more
  accurate of the two on deep out-of-the-money prices. The `method`
  argument is where D2's Jäckel solver plugs in.
- C7 (2026-10-21): `financialkit/_validation.py` holds `positive`,
  `nonnegative`, `finite`, `in_interval`, `one_of` and `option_type`. Every
  sign, interval and choice check in `options`, `exotic_options` and
  `interest_rate_models` goes through it, so each raises
  `"<name> must be positive, got <first bad value>"` and the like;
  `financialkit/tests/test_validation.py` checks 17 pricers across the
  three domains against the exact message. Relational checks (`T1 < T2`,
  `L < U`) and those whose message points to another function stay as
  they were.
- C6 (2026-10-21): `options.bump_greek(pricer, wrt, *args, **kwargs)` takes
  a central finite difference of any pricer in any named input, or in an
  attribute of a model argument (`"model.v0"`, bumped on a copy). A pair
  of names gives gamma, volga or vanna. A pricer that takes `rng` gets
  one integer seed for every evaluation, so Monte Carlo bumps share their
  random numbers. `finite_difference_greeks` is now built on it and gives
  the same numbers. Tests compare it with `black_scholes_greeks` to 1e-6
  and with `pathwise_greeks` on the same normals to within 0.05 standard
  errors. Heston vega in `v0` (Fourier and QE Monte Carlo) and the
  closed-form and Monte Carlo barrier deltas are one call each.
- C5 (2026-10-21): `stochastic_volatility.SVISurface` holds one raw SVI
  slice per expiry in log-forward-moneyness, linear in total variance
  between expiries and flat in implied vol outside them, which keeps a
  calendar-free set of slices calendar-free. `svi_surface` fits it from
  quotes. Butterfly arbitrage is Durrleman's `g` on a grid plus Lee's
  bound `b(1 + |rho|) <= 2`, flagged on Vogt's slice (Gatheral-Jacquier,
  Example 3.1); calendar arbitrage is a negative `w` increment.
  `SVISurface.local_vol` feeds the surface's call prices to
  `dupire_local_vol`. In the test, Crank-Nicolson on Dupire's forward
  equation with that local vol reprices the Heston-generated input
  quotes (3 expiries, strikes 70-130) to within 0.1 vol points. D6's
  SSVI was not scheduled, so the slices are raw SVI.
  `pde_methods.black_scholes_operator` now documents and types an
  array `sigma` (one local vol per interior node).
- C2 (2026-10-21): `longstaff_schwartz`, `barrier_option_mc`,
  `arithmetic_asian_mc` and `lookback_option_mc` take `paths=` and
  `discount=` through one helper, `exotic_options.utils.numerics.spot_paths`,
  which simulates GBM when no paths are given and otherwise checks that
  the paths start at `S` and the discount factors start at 1. Tests
  check that GBM paths passed in reproduce the default prices to the last
  bit (estimate, standard error and exercise boundary), and that discount
  factors replace the constant rate. `sigma` became optional, since only
  GBM, the bridge and the Asian geometric control need it. With supplied
  paths the Asian control is off unless asked for, and Longstaff-Schwartz
  treats the paths as independent for its standard error. The example
  *Longstaff-Schwartz on Heston paths* prices an American put on QE paths
  above the closed-form European put and below the Black-Scholes American
  put. The regression uses the spot only, so the price on a two-factor
  model is a lower bound.
- C4 (2026-10-21): `interest_rate_models.hull_white_calibrate` fits
  `(a, sigma)` to Black cap or swaption vols and `g2pp_calibrate` fits
  G2++ to caps, with `model_implied_vols` and a `ShortRateCalibration`
  result. Round-trip tests recover Hull-White from caps and from
  swaptions, and G2++ from a 20-maturity cap strip, to 1e-6. G2++ is not
  fitted to swaptions: `jamshidian_swaption` needs one factor, so that
  would need Brigo-Mercurio's two-factor swaption integral. New example
  `curve_fitted/plot_05_hull_white_calibration.py`.
- C3 (2026-10-21): `stochastic_volatility.fourier_calibrate` fits any
  characteristic-function model to prices or implied vols across strikes
  and maturities by vega-weighted `least_squares`, returning a
  `FourierCalibration` with the fitted model, parameters, residuals and
  RMSE in vol points. From a distant start, Heston, Bates, variance gamma
  and Kou recover the parameters that generated their prices to 1e-6
  (the fit reaches about 1e-9) in under 0.2 s each. A parameter vector
  the model rejects is penalised at a hundred vol points rather than
  crashing the fit. The second Heston example, *Calibrating Heston to a
  smile*, also fits noisy quotes to their 0.2 vol point noise level.
- C1 (2026-10-21): every `credit` function that discounts takes `r` as
  a number or a `fixed_income` `YieldCurve`, through one helper,
  `credit.utils.discount_factors`. Recovery of face value integrates
  the curve with `quad_vec`. Merton, its calibration and KMV use the zero
  rate to `T`, which is exact for deterministic rates. Tests pin the
  0.1.0 numbers for both a float and a `FlatCurve`, check the CDS legs
  on a bootstrapped par curve against quadrature, and check that an
  upward-sloping curve lowers the par spread of a rising hazard curve and
  raises that of a falling one. Leland (perpetual debt, stationary)
  keeps a flat `r` by design.
- B4 (2026-10-21): CI's no-numba job runs with `--cov-fail-under=100`.
  The premise was stale: 14 subpackages besides `fixed_income` and
  `interest_rate_models` were below 100%, about 207 lines in all. All of
  them are now covered by real tests: published or closed-form values,
  parity at extreme strikes, curves repriced, and validation errors
  matched by message. Two of these tests found bugs, both now fixed. The
  Hull-White moments cancelled catastrophically just above
  `xi^2 T = 1e-4`. Black-Derman-Toy looped forever on a non-positive
  forward rate and now uses an exact bracket. The only exclusion is the
  numba-present branch of `_jit.py`, which can't run in that job.
- B2.stochastic_volatility (2026-10-21): all 17 gallery examples assert
  what they illustrate (Fang and Oosterlee's VG values to 1e-8, Kou's
  9.14732, Andersen's 13.0847, Lewis matching Heston's two integrals,
  COS converging exponentially except on the kinked density, the
  Hull-White series breaking down at xi^2 T = 1.44, Bates's extra skew
  washing out by three years, QE unbiased from four steps a year,
  simulations within three standard errors of the closed forms); the
  slowest takes 0.9 s in the docs build. This was the last domain, so B2
  is done: every gallery example in the package asserts its claim.
- B2.asset_pricing (2026-10-21): all 17 gallery examples assert what
  they illustrate (zero SML and zero-beta pricing errors, GRS's 5% size
  within three binomial standard errors and its rejection of the CAPM but
  not Fama-French, Lucas's log premium of gamma sigma^2 and the Euler
  equation within three standard errors, Mehra and Prescott's 0.35%, only
  gamma = 25 inside the Hansen-Jagannathan bound, Harrison-Kreps's
  arbitrage, Modigliani-Miller's constant WACC, Shiller's violated bound);
  the slowest takes 0.7 s in the docs build.
- B2.risk (2026-10-21): all 17 gallery examples assert what they
  illustrate (Kupiec's power on one year and four, the Basel zones and
  binomial probabilities, Christoffersen rejecting the constant VaR,
  Kendall's tau unmoved by the marginals, VaR failing subadditivity only
  between 92% and 96%, Euler contributions summing to the closed forms,
  POT VaR within 15% of the truth out to 99.99%, filtered historical
  simulation passing conditional coverage where plain historical fails);
  the slowest takes 1.2 s in the docs build.
- B2.portfolio (2026-10-21): all 18 gallery examples assert what they
  illustrate (He and Litterman's equilibrium returns, SLSQP on Merton's
  hyperbola, Tobin separation, equal and budgeted risk contributions,
  CPPI's closed form and gap risk, simulations within three standard
  errors); the slowest takes 2.5 s, with numba or without.
- A3 (2026-10-21): every closed-form pricer and Greek in `options`,
  `exotic_options` and `interest_rate_models` returns its limit or raises
  `ValueError` at `T = 0`, `sigma = 0`, `K = 0`, negative rates and
  negative `S`; `tests/test_degenerate_inputs.py` in each domain checks
  every pricer at every limit, the riskless limits against the payoff of
  the path `S e^{bt}`. It also found and fixed the sign of the gap put.
- B2.exotic_options (2026-10-21): all 17 gallery examples assert what
  they illustrate (Haug's printed values, in-out and compound parity, Monte
  Carlo within 4 standard errors of each closed form and of
  Longstaff-Schwartz's Table 1, the BGK shift against discrete monitoring);
  the slowest takes 0.4 s in the docs build without numba.
- B2.pde_methods (2026-10-21): all 16 gallery examples assert what they
  illustrate (prices against Black-Scholes, Margrabe, the method of images
  and a CRR tree; observed orders of 1 and 2; the explicit scheme stable
  below its limit and blown up above it; Rannacher's gamma within 1e-3);
  the slowest takes 1.8 s in the docs build without numba.
- B2.interest_rate_models (2026-10-21): all 16 gallery examples assert
  what they illustrate (Monte Carlo within 3 standard errors of the closed
  form, cap-floor and swaption parity, Hull's printed values, the HJM
  drift condition); each still runs in under 2 s without numba.
- B7.options (2026-10-21): `options/tests/test_reference_values.py` pins
  seven Haug, 2nd ed., values; B7 stays open for the other domains.
- A4 (2026-10-21): `fixed_income.bond_accrued_interest`,
  `exotic_options.annualized_realized_variance` and
  `fixed_income.plot_discounted_cash_flows` (built on
  `time_value.plot_cash_flows`) replace the clashing names, which remain
  as deprecated aliases; each docstring names its counterpart.
- A2 (2026-10-21): `implied_volatility` raises for `T <= 0`, caps its
  bracket and has regression tests for `T = 0`, the bounds and deep
  out-of-the-money options.
- Five tutorials, the quickstart notebook in Colab and JupyterLite, and
  the `outreach/` drafts.
- B1 (2026-10-21): `financialkit/tests/test_history_conventions.py` checks
  every history page's dated, chronological entries, their
  *Implementation:* lines and minigalleries, and that every gallery
  example exists and is linked from exactly one entry.
- B3 (2026-10-21): `filterwarnings = ["error"]` in `pyproject.toml`; the
  suite and the doctests were already warning-free with numba, with
  `NUMBA_DISABLE_JIT=1` and with numba not installed, so nothing is
  allow-listed.

### 0.1.0 (2026-10-10)

- Scaffold cloned from mathematicskit: CI, docs, gallery, history pages,
  JupyterLite, release workflow, `constants`, `_jit` and `integrators`.
- All 14 domains, built one per session in the order above.
- GitHub repository and Pages, PyPI trusted publishing, the Zenodo
  version DOI and badge, the Codecov token.

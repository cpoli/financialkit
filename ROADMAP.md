# Roadmap

## Where financialkit stands

0.1.0 (2026-10-10) shipped 14 domains: 236 breakthroughs and 239 gallery
examples, 868 tests at 99% line coverage, 5 tutorials and the quickstart
notebook. All the domains are built. The per-domain build plans that used
to be here are in git history (`git show 681a826:ROADMAP.md`). The history
pages in `docs/source/history/` are the record of every entry.

| Domain | Breakthroughs | Gallery examples | Latest entry |
|---|---:|---:|---:|
| `time_value` | 16 | 16 | 1998 |
| `fixed_income` | 16 | 17 | 1994 |
| `options` | 16 | 18 | 1996 |
| `interest_rate_models` | 16 | 16 | 2001 |
| `monte_carlo` | 18 | 18 | 2008 |
| `exotic_options` | 17 | 17 | 2001 |
| `pde_methods` | 16 | 16 | 2002 |
| `stochastic_volatility` | 17 | 17 | 2010 |
| `time_series` | 19 | 19 | 1998 |
| `risk` | 17 | 17 | 2005 |
| `portfolio` | 18 | 18 | 2016 |
| `asset_pricing` | 17 | 17 | 1997 |
| `credit` | 17 | 17 | 2009 |
| `market_microstructure` | 16 | 16 | 2012 |

### What the 0.1.0 audit found

The breadth is real, but six weaknesses matter more than more breakthroughs:

1. **The examples prove nothing.** None of the 239 gallery scripts ends
   in an `assert`, and the tutorials are static `code-block`s with
   hand-pasted output that nothing re-runs. The docs build only catches
   crashes, so a wrong number goes straight to the site.
2. **The domains don't compose.** `credit` (CDS legs, hazard bootstrap,
   risky bonds) takes a flat `r` instead of a `CurveLike`. The
   Longstaff-Schwartz and exotic Monte Carlo pricers simulate their own
   GBM from `sigma`, so an American or a barrier under Heston, local
   volatility or Hull-White is impossible. The planned XVA work needs both.
3. **No calibration.** Only SABR and a single SVI slice can be fitted.
   Nothing calibrates Heston, Bates or variance gamma to a smile, Hull-White
   or G2++ to caps and swaptions, or Dupire to an implied-volatility
   surface, although calibration is how all these models are actually used.
4. **No dates.** Bonds and swaps are built from year fractions and a
   regular `freq`. There are no schedules, stubs, business-day rolls or
   settlement lags, and SOFR compounding in arrears (planned) needs them.
5. **Synthetic data only.** Every empirical lesson (stylized facts,
   GARCH, Fama-French, Shiller's excess volatility, the equity premium)
   runs on data drawn from the model being shown, which is circular.
6. **History pages stop early and skip classics.** Six domains end before
   2000. Missing pre-2008 cornerstones include Hansen's GMM, Hamilton's
   regime switching, Johansen cointegration, DCC-GARCH, the Kalman filter,
   Bjerksund-Stensland, callable bonds and OAS, MBS prepayment and Barra
   factor risk models.

Smaller problems: `options.implied_volatility` loops forever when `T = 0`
(its bracket-doubling `while` has no cap) and solves one element at a time
with `brentq`. Pricers return `nan` silently on invalid input such as
`S < 0` (there is no shared validation module, just 252 ad-hoc
`ValueError`s). Two public names have two implementations with different
signatures (`accrued_interest`, `realized_variance`). Warnings don't fail
the tests. Nothing tests the convention that every example is linked from
exactly one breakthrough.

## How to work from this file

Each item below is meant to be done on its own, in one session, from a
prompt such as "Do ROADMAP item C2". An item names what to change and when
it is done. Where one item should come after another it says **After**.
Nothing else is ordered, but section A is small and should go first.

Every item is done when:

- `ruff check .`, `ruff format --check .`, `mypy financialkit`,
  `MPLBACKEND=Agg pytest -q`, the same with `NUMBA_DISABLE_JIT=1`, the
  doctests and `cd docs && make html` are green;
- the change has a line under `## [Unreleased]` in CHANGELOG.md;
- the item is moved from "Open work" to "Done" here, with the date.

CLAUDE.md applies to everything: no new dependencies, nothing on the
network, a domain imports only from earlier domains, every model cites its
source, and every new breakthrough has a dated history entry, its own
gallery example and a `systems/` implementation. A history page must never
fall below 15 entries. Keep the public API backward compatible: when you
rename something, keep the old name as a deprecated alias.

Sessions may run in parallel. CHANGELOG.md, this file, the table above and
`docs/source/conf.py` are shared: re-read them just before editing and
change only your item's lines.

## Open work

### A. Hygiene and bugs

**A1. Commit or revert the pending working tree.** `README.md`,
`docs/source/conf.py`, `docs/source/index.rst` and `pyproject.toml` hold
uncommitted SEO work (sitemap, OpenGraph, a "Related toolkits" rubric) and
put shields.io badges back after commits `df90ed0`, `c210460` and `6ffbd6f`
moved them to badgen. Decide which badge host wins, check that
`sphinx-sitemap` and `sphinxext-opengraph` are in the `docs` extra, and
commit or drop. *Done when* `git status` is clean and the docs build passes.

**A2. Fix `implied_volatility`.** In
`financialkit/options/systems/implied_vol.py`, `while f(hi) < 0: hi *= 2`
never ends for `T = 0` (and in general has no cap). Return intrinsic-based
results or raise a clear `ValueError` for `T <= 0`, cap the bracket, and
add regression tests for `T = 0`, prices at the bounds and deep
out-of-the-money options. *Done when* every case returns or raises within
a second.

**A3. Degenerate inputs in closed-form pricers.** Check every closed-form
pricer and Greek in `options`, `exotic_options` and `interest_rate_models`
at `T = 0`, `sigma = 0`, `K = 0`, negative rates and, where it is invalid,
negative `S`. Each must return the correct limit or raise `ValueError`,
never `nan` or a `RuntimeWarning` (today `black_scholes_greeks` at `T = 0`
warns and `black_scholes_price(-100, ...)` returns `nan`). **After** B3,
which will surface these. *Done when* a parametrized test covers each
pricer at each limit.

**A4. One implementation per name.** `time_value.accrued_interest` (dates
and a day count) and `fixed_income.accrued_interest` (year fractions)
differ in signature, and so do `exotic_options.realized_variance` and
`time_series.realized_variance`, and the two `plot_cash_flows`. Make the
later domain call the earlier one, or rename one (keep a deprecated
alias), and say in each docstring which is which. *Done when* no public
name means two different things.

### B. Quality gates

**B1. Test the history conventions.** Add
`financialkit/tests/test_history_conventions.py`: every history page has at
least 15 dated entries in chronological order, each entry has an
`*Implementation:*` line and a `.. minigallery::`, every
`examples/**/plot_*.py` is linked from exactly one entry, and every linked
file exists. Copy engineeringkit's test if it fits. *Done when* the test
passes and deleting a link makes it fail.

**B2. Gallery examples check their claims, one domain per session.** For
the domain named in the prompt, end each `examples/<domain>/**/plot_*.py`
with an `assert` of the result it illustrates (parity holds, the tree
converges to the formula, the backtest rejects, within a stated
tolerance), not just that it ran. *Done when* every example of that domain
asserts, and the docs build still fits the 5-second budget.

**B3. Make warnings errors.** Add `filterwarnings = ["error"]` to
`[tool.pytest.ini_options]` and fix the root causes rather than filtering
them. Allow-list only third-party deprecations you cannot fix, each with a
comment. *Done when* the suite passes with and without numba.

**B4. Coverage gate.** Bring `fixed_income` (96%) and
`interest_rate_models` (97%) to 100% with real tests, then add
`--cov-fail-under=100` to the no-numba CI job, the one that measures
coverage. *Done when* CI enforces it.

**B5. Test the supported dependency range and Windows.** `pyproject.toml`
promises numpy 1.24, scipy 1.10 and matplotlib 3.6, but CI installs the
latest. Add a Python 3.10 job with those floors pinned and a
`windows-latest` row. Fix what fails or raise the floors. *Done when* both
jobs are green.

**B6. Run the tutorials and the quickstart notebook in CI.** The five
tutorials are `code-block`s with pasted output that nothing executes.
Convert them to executed blocks (`.. plot::`, or a test that extracts each
block and compares the printed lines with the `# ` comments, as in
engineeringkit's `test_course_material.py`), and execute
`notebooks/quickstart.ipynb` in CI without new runtime dependencies (a test
that runs its code cells is enough). *Done when* changing a model's output
fails the tests.

**B7. Textbook reference tests, one domain per session.** Today 29 of 104
test files cite Hull or Haug. For the domain named in the prompt, add
`tests/test_reference_values.py` with at least five published numbers
(Hull, Haug, Glasserman, Brigo-Mercurio, Tsay, McNeil-Frey-Embrechts or
the original paper), citing edition and table or example number. *Done
when* the domain has five such tests and each fails if the model is
perturbed.

**B8. Property tests for pricers.** `hypothesis` is a test dependency but
is used in only 3 files. Add property tests for the no-arbitrage
invariants: call price increasing in `S` and `T` and convex and decreasing
in `K`, put-call parity for every European pricer (closed form, trees,
PDE, FFT, COS), and American price at least the European. *Done when* the
properties run on every European pricer.

**B9. Enforce the 5-second budget per gallery example.** Fail the docs
build when an entry of `sg_execution_times.rst` exceeds 5 s in the
no-numba docs job. *Done when* CI fails on a deliberately slow example.

**B10. Benchmarks for the `@njit` kernels.** Time the trees, PSOR,
Longstaff-Schwartz and the SDE integrators with and without numba, in a
script under `benchmarks/` and a table in the docs, so "same results,
slower" has numbers. *Done when* the table is in the docs.

**B11. Check external links.** Add a weekly workflow running
`make -C docs linkcheck`, with `linkcheck_ignore` for publishers that
refuse robots. *Done when* the first run is green.

### C. Architecture and composability

**C1. Credit takes a discount curve.** Make `cds_legs`, `cds_par_spread`,
`cds_upfront`, `bootstrap_hazard_curve`, `risky_zero_coupon` and the other
`credit` functions that take `r` accept a `fixed_income` `CurveLike`
(`as_curve` already turns a float into a flat curve, so existing calls keep
working). Test that a flat curve reproduces today's numbers and that a
bootstrapped curve changes the par spread as expected. *Done when* no
`credit` function needs a flat rate.

**C2. Price on any paths.** Give `longstaff_schwartz`, `barrier_option_mc`,
`arithmetic_asian_mc` and `lookback_option_mc` a `paths=` (and discount
factors) entry point, keeping the GBM arguments as the default. Add a
gallery example pricing an American put on Heston paths
(`stochastic_volatility.heston_paths`) and a test that GBM paths
reproduce today's prices. Mind the build order: `stochastic_volatility`
is after `exotic_options`, so the example combines them and the library
does not import across. *Done when* the four pricers accept paths.

**C3. Calibrate Fourier models to a smile.** Add a calibration function
in `stochastic_volatility` that fits a `CharacteristicFunctionModel`
(Heston, Bates, variance gamma, Kou) to option prices or implied vols
across strikes and maturities with `scipy.optimize.least_squares`,
vega-weighted, with a result dataclass (parameters, residuals, RMSE in
vol points). Test that it recovers the parameters from prices it
generated. Add a gallery example as a second example under the Heston
breakthrough. *Done when* the round-trip test passes.

**C4. Calibrate short-rate models to caps and swaptions.** Add a
calibration of `HullWhiteModel` (`a`, `sigma`) and `G2PlusPlusModel` to
Black cap or swaption vols, using `model_cap_floor_price` and
`jamshidian_swaption`. Add a round-trip test and a gallery example under
the Hull-White breakthrough. *Done when* the round-trip test passes.

**C5. An implied-volatility surface.** Add a surface object in
`stochastic_volatility` built from SVI slices, with Gatheral's calendar
and butterfly arbitrage checks (`durrleman_g` exists), and feed it to
`dupire_local_vol` so local volatility can be built from implied vols, not
only from a price function. **After** D2's SSVI if both are scheduled.
*Done when* local volatility from the surface reprices the input options
within a stated tolerance in a test.

**C6. Greeks for any pricer.** Generalize `finite_difference_greeks`
(today fixed to `S, K, T, r, sigma, q`) to bump any named parameter of any
pricer or model, with central differences and common random numbers for
Monte Carlo pricers. Test it against `black_scholes_greeks` and against
`pathwise_greeks`. *Done when* Heston vega (bumping `v0`) and a barrier
delta are one call each.

**C7. Shared input validation.** Add `financialkit/_validation.py`
(`positive`, `nonnegative`, `finite`, `in_interval`, `option_type`) and use
it in the closed-form pricers first, replacing the duplicated checks.
**After** A3. *Done when* `options`, `exotic_options` and
`interest_rate_models` validate through it and the error messages are
uniform.

**C8. Schedules and business days.** Add to `time_value` a coupon schedule
generator (start, end, frequency, short or long front or back stub,
end-of-month rule), business-day adjustments (following, modified
following, preceding) on a weekend calendar plus a user-supplied holiday
array (no bundled holiday data), and settlement lags. Let
`fixed_income.bond_cash_flows` and the swaps accept a schedule. Test
against ISDA and SIA worked examples. *Done when* a bond with a short
first coupon prices and accrues correctly.

**C9. Vectorized implied volatility.** Replace the per-element `brentq`
loop with a vectorized solver (Newton or Halley on a rational initial
guess, falling back to bisection), so a 100 × 100 surface inverts in
milliseconds. Jäckel's method (D2) can then become the default. **After**
A2. *Done when* a benchmark test shows the speed-up and the results match
`brentq` to `DEFAULT_ATOL`.

**C10. API review before 1.0.** Review every `__all__`: function vs. class
for the same kind of model (`heston_price` vs. `HestonModel`, `GARCH` vs.
`fit_arma`), `tol`/`max_iter`/`n_paths`/`n_steps`/`rng` spelled the same
everywhere (`n_exercise`, `n_space`, `n_time`), consistent result field
names, and whether the visualizers are exported (`fixed_income` and
`exotic_options` export `plot_*`, `options` and `credit` don't). Write the
decisions into CONTRIBUTING.md and deprecate what changes. **After** the C
items above. *Done when* the review is written and applied.

### D. Curriculum gaps, one domain per item

Each entry joins its domain's history page in chronological order, with
its own gallery example at the path in brackets and a `systems/`
implementation. An item may be split across sessions; tick entries as
they land.

**D1. `fixed_income`.** **After** C8 for SOFR.
- [ ] 1977 -- Treasury bond futures: conversion factor and cheapest to deliver [`futures/plot_01_cheapest_to_deliver.py`]
- [ ] 1985 -- The PSA prepayment benchmark and MBS cash flows [`mbs/plot_01_psa_prepayment.py`]
- [ ] 2010 -- Piterbarg: OIS discounting of collateralized trades, with separate forecasting curves [`curves/plot_07_ois_multi_curve.py`]
- [ ] 2018 -- SOFR: compounding in arrears [`swaps/plot_03_sofr_compounded_in_arrears.py`]

**D2. `options`.** **After** C9 for Jäckel.
- [ ] 1993 -- Bjerksund-Stensland American approximation [`trees/plot_07_bjerksund_stensland.py`]
- [ ] 2003 -- Haug-Haug-Lewis: discrete cash dividends [`closed_form/plot_08_discrete_dividends.py`]
- [ ] 2007 -- Castagna-Mercurio vanna-volga FX smile [`implied_vol/plot_06_vanna_volga.py`]
- [ ] 2015 -- Jäckel's "Let's be rational" implied volatility [`implied_vol/plot_05_jaeckel_lets_be_rational.py`]
- [ ] 2020 -- Negative oil futures: Bachelier returns to the exchange [`closed_form/plot_07_bachelier_negative_prices.py`]

**D3. `interest_rate_models`.**
- [ ] 1989 -- Richard-Roll: MBS prepayment under a short-rate model [`mbs/plot_01_richard_roll_prepayment.py`]
- [ ] 1994 -- Hull-White trinomial tree: callable bonds and the option-adjusted spread [`curve_fitted/plot_05_callable_bond_oas.py`]
- [ ] 2019 -- Lyashenko-Mercurio forward market model for backward-looking rates [`options/plot_05_lyashenko_mercurio_fmm.py`]

**D4. `monte_carlo`.**
- [ ] 2006 -- Giles-Glasserman adjoint Greeks ("smoking adjoints") [`greeks/plot_03_adjoint_greeks.py`]

**D5. `pde_methods`.**
- [ ] 1988 -- Craig-Sneyd ADI with mixed derivatives [`multi_factor/plot_02_craig_sneyd.py`]
- [ ] 2010 -- In 't Hout-Foulon: the Heston PDE by ADI [`multi_factor/plot_03_heston_adi.py`]

**D6. `stochastic_volatility`.** Rough Bergomi needs a fractional kernel
simulator, written here, not in `integrators` unless it is general.
- [ ] 2014 -- Gatheral-Jacquier arbitrage-free SVI surfaces (SSVI) [`smile/plot_04_ssvi.py`]
- [ ] 2016 -- Bayer-Friz-Gatheral rough Bergomi [`rough/plot_01_rough_bergomi.py`]
- [ ] 2017 -- Bennedsen-Lunde-Pakkanen hybrid scheme [`rough/plot_02_hybrid_scheme.py`]
- [ ] 2018 -- Gatheral-Jaillet-Rosenbaum: volatility is rough [`rough/plot_03_volatility_is_rough.py`]

**D7. `time_series`.** The largest gap: the page stops in 1998 and has no
multivariate or state-space models.
- [ ] 1960 -- Kalman filter and the state-space model [`state_space/plot_01_kalman_filter.py`]
- [ ] 1980 -- Sims's vector autoregression and impulse responses [`linear_models/plot_04_sims_var.py`]
- [ ] 1988 -- Johansen cointegration test [`unit_roots/plot_03_johansen.py`]
- [ ] 1989 -- Hamilton's regime-switching model [`state_space/plot_02_hamilton_regime_switching.py`]
- [ ] 1992 -- KPSS stationarity test [`unit_roots/plot_04_kpss.py`]
- [ ] 2002 -- Engle's dynamic conditional correlation (DCC-GARCH) [`volatility/plot_09_engle_dcc.py`]
- [ ] 2009 -- Corsi's HAR model of realized volatility [`volatility/plot_08_corsi_har.py`]

**D8. `risk`.**
- [ ] 1997 -- Rouvinez: delta-gamma VaR [`var/plot_07_delta_gamma_var.py`]
- [ ] 1999 -- Bangia-Diebold-Schuermann-Stroughair liquidity-adjusted VaR [`var/plot_08_liquidity_adjusted_var.py`]
- [ ] 2014 -- Acerbi-Szekely expected-shortfall backtests [`backtesting/plot_04_acerbi_szekely.py`]
- [ ] 2016 -- FRTB: 97.5% expected shortfall with liquidity horizons [`expected_shortfall/plot_04_frtb_expected_shortfall.py`]

**D9. `portfolio`.**
- [ ] 1974 -- Rosenberg's fundamental factor risk model (Barra) [`covariance/plot_03_rosenberg_factor_model.py`]
- [ ] 2003 -- Goldfarb-Iyengar robust portfolio selection [`bayesian/plot_03_robust_optimization.py`]
- [ ] 2013 -- Gârleanu-Pedersen: trading with transaction costs [`growth/plot_04_garleanu_pedersen.py`]

**D10. `asset_pricing`.** The page stops in 1997 and lacks GMM, the
estimator the whole field uses.
- [ ] 1982 -- Hansen's generalized method of moments [`sdf/plot_03_hansen_gmm.py`]
- [ ] 1982 -- Hansen-Singleton: the Euler equation estimated [`consumption/plot_04_hansen_singleton.py`]
- [ ] 1988 -- Campbell-Shiller log-linear present value [`valuation/plot_05_campbell_shiller.py`]
- [ ] 1999 -- Campbell-Cochrane habit formation [`consumption/plot_05_campbell_cochrane_habit.py`]
- [ ] 2004 -- Bansal-Yaron long-run risk [`consumption/plot_06_bansal_yaron.py`]
- [ ] 2015 -- Fama-French five factors [`factors/plot_05_fama_french_five.py`]
- [ ] 2016 -- Harvey-Liu-Zhu: the factor zoo and multiple testing [`factors/plot_06_harvey_liu_zhu.py`]

**D11. `credit`.** **After** C1 and C2: exposure is Longstaff-Schwartz
regression on Hull-White paths, discounted on a curve.
- [ ] 2003 -- Andersen-Sidenius-Basu loss recursion [`portfolio_loss/plot_06_andersen_sidenius_basu.py`]
- [ ] 2004 -- Base correlation [`portfolio_loss/plot_07_base_correlation.py`]
- [ ] 2005 -- Brigo-Masetti: unilateral CVA from exposure profiles [`xva/plot_01_brigo_masetti_cva.py`]
- [ ] 2009 -- Gregory: bilateral CVA and DVA [`xva/plot_02_gregory_bilateral_cva.py`]
- [ ] 2011 -- Burgard-Kjaer: funding valuation adjustment [`xva/plot_03_burgard_kjaer_fva.py`]

**D12. `market_microstructure`.**
- [ ] 1971 -- Hawkes self-exciting order flow [`order_flow/plot_01_hawkes.py`]
- [ ] 2010 -- Cont-Stoikov-Talreja limit order book [`order_flow/plot_02_cont_stoikov_talreja.py`]
- [ ] 2013 -- Obizhaeva-Wang transient impact [`execution/plot_04_obizhaeva_wang.py`]
- [ ] 2014 -- Cont-Kukanov-Stoikov order-flow imbalance [`order_flow/plot_03_order_flow_imbalance.py`]

`time_value`, `exotic_options` and the rest of `monte_carlo` have no
pressing gaps.

### E. Data

**E1. Decide on bundled datasets (the author).** CLAUDE.md allows small
CSVs under `financialkit/datasets/`, "generated by a committed script",
but a script that downloads breaks the no-network rule, and the
candidate sources have different terms: Shiller's stock-market data
(public), FRED Treasury yields (public domain for most series), Ken
French's factor library (no stated licence). Decide which, if any, ship,
and how the script is run (by the author, once, offline from CI).
*Done when* the decision is written into CLAUDE.md.

**E2. Build `financialkit.datasets` and port the empirical examples.**
**After** E1. Add loaders that return numpy arrays (no pandas), the
generating script, package data and a test of each loader. Port the
examples where synthetic data is circular: Mandelbrot's fat tails, the
Hill estimator, GARCH, Shiller's excess volatility, the equity premium
puzzle, Fama-French and yield-curve PCA. *Done when* those examples run
on real data offline.

### F. Course material

**F1. Exercises, one domain per session.** For the domain named in the
prompt, add `docs/source/exercises/<domain>.rst`: five to ten problems,
each built on a gallery example (change a parameter, predict, check with
financialkit), with solutions in collapsed `sphinx-design` dropdowns that
run in the docs build and are checked as in B6. *Done when* the page
builds and its solutions are tested.

**F2. Two tutorials on the new material.** *One swap, two curves* (OIS
discounting, SOFR compounding, the forward market model) and
*Counterparty risk, end to end* (Hull-White exposure, CVA, DVA, FVA).
**After** D1, D3, D11 and B6. *Done when* both are linked from the
galleries they use and executed in CI.

**F3. A calibration tutorial.** *From quotes to a model*: an SVI surface,
local volatility from it, Heston fitted to the same quotes, and the
exotic prices the two disagree on. **After** C2, C3 and C5.

### G. Releases (the author)

**G1. Release 0.2.0** (the tutorials and the quickstart notebook under
`[Unreleased]` in CHANGELOG.md), then put its Zenodo version DOI in
`CITATION.cff` and the README badge. Releasing A2 with it is worth it.

**G2. conda-forge recipe.** A `recipe.yaml` with only `version` in
`context`, and `name: financialkit` and the PyPI source URL written out.

**G3. 1.0.** Freezes the public API (see CONTRIBUTING.md). **After** all
of B, C10 and G1.

### H. Scope decisions

- **Out of scope** (the dependencies are fixed): deep hedging and other
  neural-network methods, which need a deep-learning framework; market
  data fetching; wrappers around QuantLib.
- **Candidate domains**, not scheduled. CLAUDE.md forbids adding domains
  without the author's decision, each would need 15+ breakthroughs, and
  none should start before section D, since every existing domain still
  has gaps:
  - `commodities`: Gibson-Schwartz (1990), Schwartz's one-, two- and
    three-factor models (1997), Schwartz-Smith (2000), seasonality,
    storage and spark-spread options;
  - `inflation`: Jarrow-Yildirim (2003), zero-coupon and year-on-year
    inflation swaps, inflation caps;
  - `real_options`: McDonald-Siegel (1986), Dixit-Pindyck (1994),
    Trigeorgis's expansion and abandonment options, Myers's
    underinvestment (1977).

## Done

### Unreleased

- Five tutorials, the quickstart notebook in Colab and JupyterLite, and
  the `outreach/` drafts.

### 0.1.0 (2026-10-10)

- Scaffold cloned from mathematicskit: CI, docs, gallery, history pages,
  JupyterLite, release workflow, `constants`, `_jit` and `integrators`.
- All 14 domains, built one per session in the order above.
- GitHub repository and Pages, PyPI trusted publishing, the Zenodo
  version DOI and badge, the Codecov token.

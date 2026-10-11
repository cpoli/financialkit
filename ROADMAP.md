# Roadmap

financialkit is built in phases, each green (ruff, pytest, doctests,
mypy, docs) before the next. This file is the build checklist and the
plan of every domain's history page. Each breakthrough is implemented in
the domain's `systems/` modules and has its own gallery example, named in
brackets as `<topic>/plot_NN_<slug>.py`.

## Release TODOs (for the author, after the first GitHub Release)

- [x] Create the GitHub repository `cpoli/financialkit`, push, and enable
      Pages from `gh-pages` (the release workflow deploys there).
- [x] PyPI trusted publisher (see the header of `.github/workflows/release.yml`).
- [x] Enable Zenodo for the repository, then add the **version DOI** of
      the release (never the concept DOI) to `CITATION.cff` as `doi:`.
- [x] Add the Zenodo badge to the README in the
      `https://zenodo.org/badge/DOI/<version-doi>.svg` form (update it each release).
- [x] Add a Codecov token (repository secret `CODECOV_TOKEN`).

## Phase checklist

Each domain is built in its own session, from this file plus
`financialkit-spec.md`.

- [x] 1. Scaffold: infrastructure, `constants`, `_jit`, `integrators` (SDE schemes), root docs, git
  - [x] Infrastructure copied from mathematicskit and renamed; `pyproject.toml`, `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` edited
  - [x] `CLAUDE.md`
  - [x] `README.md`, `docs/source/index.rst`, `CHANGELOG.md`, `docs/make_readme_*figure*.py` (finance text and figures, with the disclaimer)
  - [x] `conf.py` `SUBPACKAGES`/`_CATEGORY_ORDER` and the docs index pages (finance categories; domains are added as they land)
  - [x] New logo (candlestick icon)
  - [x] `financialkit/__init__.py`, `constants.py`, `integrators/` with tests
  - [x] `git init` (author cpoli374 <cpoli374@gmail.com>) and first commit
- [x] 2. `time_value` (reference domain)
- [x] 3. `fixed_income`
- [x] 3. `options`
- [x] 3. `interest_rate_models`
- [x] 3. `monte_carlo`
- [x] 3. `exotic_options`
- [x] 3. `pde_methods`
- [x] 3. `stochastic_volatility`
- [x] 3. `time_series`
- [x] 3. `risk`
- [x] 3. `portfolio`
- [x] 3. `asset_pricing`
- [x] 3. `credit`
- [x] 3. `market_microstructure`
- [x] 4. Tutorials, quickstart notebook, JupyterLite contents, `outreach/`
  - [x] Five tutorials in `docs/source/tutorials/`, linked from the galleries they use: one option, five pricers; the yield curve, end to end; where the smile comes from; from returns to risk; credit risk, from one firm to a portfolio
  - [x] `notebooks/quickstart.ipynb` (six domains), and the Colab badge and quickstart links restored in `README.md` and `docs/source/index.rst`
  - [x] The quickstart notebook in the JupyterLite site (`jupyterlite_contents` in `conf.py`); the gallery notebooks were already there
  - [x] `outreach/` drafts (gitignored, as in mathematicskit)

## Planned breakthroughs

### time_value

1. c. 1202 -- Fibonacci's *Liber Abaci*: present value [`present_value/plot_01_fibonacci_present_value.py`]
2. 1494 -- Pacioli's rule of 72 [`compounding/plot_01_pacioli_rule_of_72.py`]
3. 1613 -- Witt's compound-interest tables [`compounding/plot_02_witt_compound_interest_tables.py`]
4. 1671 -- De Witt's life annuities [`annuities/plot_01_de_witt_life_annuity.py`]
5. 1683 -- Bernoulli and continuous compounding [`compounding/plot_03_bernoulli_continuous_compounding.py`]
6. 1751 -- Consols and the perpetuity [`annuities/plot_02_consol_perpetuity.py`]
7. 1772 -- Price's sinking fund [`annuities/plot_03_price_sinking_fund.py`]
8. 1896 -- Fisher's real and nominal rates [`present_value/plot_02_fisher_real_rate.py`]
9. 1907 -- Fisher's net present value [`present_value/plot_03_fisher_net_present_value.py`]
10. 1930 -- The internal rate of return [`irr/plot_01_internal_rate_of_return.py`]
11. 1933 -- The fully amortizing mortgage [`amortization/plot_01_amortizing_mortgage.py`]
12. 1955 -- Lorie-Savage: multiple IRRs [`irr/plot_02_lorie_savage_multiple_irr.py`]
13. 1959 -- Baldwin's modified IRR [`irr/plot_03_baldwin_mirr.py`]
14. 1968 -- Truth in Lending: APR and the effective rate [`amortization/plot_02_truth_in_lending_apr.py`]
15. 1993 -- SIA 30/360 and accrued interest [`day_counts/plot_01_thirty_360_accrued_interest.py`]
16. 1998 -- ISDA's actual/actual and the day-count zoo [`day_counts/plot_02_isda_actual_actual.py`]

### fixed_income

1. 1938 -- Macaulay duration [`duration/plot_01_macaulay_duration.py`]
2. 1939 -- Hicks: duration as an elasticity (modified duration) [`duration/plot_02_hicks_modified_duration.py`]
3. 1940 -- Lutz: forward rates [`curves/plot_01_lutz_forward_rates.py`]
4. 1952 -- Redington immunization [`immunization/plot_01_redington_immunization.py`]
5. 1971 -- Fisher-Weil duration [`duration/plot_03_fisher_weil_duration.py`]
6. 1971 -- McCulloch's spline discount function [`curves/plot_02_mcculloch_spline.py`]
7. 1972 -- *Inside the Yield Book*: yield to maturity [`bonds/plot_01_yield_to_maturity.py`]
8. 1976 -- Carleton-Cooper bootstrapping [`curves/plot_03_bootstrapping.py`]
9. 1981 -- The IBM-World Bank swap [`swaps/plot_01_ibm_world_bank_currency_swap.py`, `swaps/plot_02_interest_rate_swap.py`]
10. 1983 -- Bierwag-Kaufman-Toevs: convexity [`duration/plot_04_convexity.py`]
11. 1984 -- Fong-Vasicek M-squared [`immunization/plot_02_fong_vasicek_m_squared.py`]
12. 1985 -- STRIPS: a bond as a portfolio of zeros [`bonds/plot_02_strips_zero_coupon.py`]
13. 1987 -- Nelson-Siegel [`curves/plot_04_nelson_siegel.py`]
14. 1991 -- Litterman-Scheinkman: level, slope, curvature [`curves/plot_05_litterman_scheinkman_pca.py`]
15. 1992 -- Ho's key rate durations [`duration/plot_05_ho_key_rate_durations.py`]
16. 1994 -- Svensson [`curves/plot_06_svensson.py`]

### options

1. 1900 -- Bachelier [`closed_form/plot_01_bachelier.py`]
2. 1969 -- Stoll's put-call parity [`closed_form/plot_02_stoll_put_call_parity.py`]
3. 1973 -- Black-Scholes [`closed_form/plot_03_black_scholes.py`]
4. 1973 -- Merton: dividends and no-arbitrage bounds [`closed_form/plot_04_merton_bounds_and_dividends.py`]
5. 1973 -- Delta hedging and the Greeks [`greeks/plot_01_greeks.py`, `greeks/plot_02_delta_hedging.py`]
6. 1976 -- Black-76 [`closed_form/plot_05_black76.py`]
7. 1976 -- Latane-Rendleman implied volatility [`implied_vol/plot_01_implied_volatility.py`]
8. 1978 -- Breeden-Litzenberger risk-neutral density [`implied_vol/plot_02_breeden_litzenberger.py`]
9. 1979 -- Cox-Ross-Rubinstein binomial tree [`trees/plot_01_crr_convergence.py`, `trees/plot_02_crr_american_put.py`]
10. 1979 -- Rendleman-Bartter tree [`trees/plot_03_rendleman_bartter.py`]
11. 1983 -- Garman-Kohlhagen FX options [`closed_form/plot_06_garman_kohlhagen.py`]
12. 1986 -- Boyle's trinomial tree [`trees/plot_04_boyle_trinomial.py`]
13. 1987 -- Barone-Adesi-Whaley American approximation [`trees/plot_05_barone_adesi_whaley.py`]
14. 1987 -- The volatility smile [`implied_vol/plot_03_volatility_smile.py`]
15. 1988 -- Brenner-Subrahmanyam at-the-money approximation [`implied_vol/plot_04_brenner_subrahmanyam.py`]
16. 1996 -- Leisen-Reimer tree [`trees/plot_06_leisen_reimer.py`]

### interest_rate_models

1. 1949 -- Feynman-Kac: bond prices by Monte Carlo [`short_rate/plot_01_feynman_kac_bond_price.py`]
2. 1951 -- Feller's boundary condition [`short_rate/plot_02_feller_condition.py`]
3. 1973 -- Merton's Gaussian short rate [`short_rate/plot_03_merton_short_rate.py`]
4. 1977 -- Vasicek [`short_rate/plot_04_vasicek.py`]
5. 1985 -- Cox-Ingersoll-Ross [`short_rate/plot_05_cir.py`]
6. 1986 -- Ho-Lee [`curve_fitted/plot_01_ho_lee.py`]
7. 1989 -- Jamshidian: bond options [`options/plot_01_jamshidian_bond_option.py`]
8. 1989 -- Jamshidian decomposition: swaptions [`options/plot_02_jamshidian_swaption.py`]
9. 1990 -- Hull-White [`curve_fitted/plot_02_hull_white.py`]
10. 1990 -- Black-Derman-Toy [`curve_fitted/plot_03_black_derman_toy.py`]
11. 1990 -- Futures convexity adjustment [`curve_fitted/plot_04_futures_convexity_adjustment.py`]
12. 1992 -- Heath-Jarrow-Morton drift condition [`hjm/plot_01_hjm_drift_condition.py`]
13. 1996 -- Duffie-Kan affine term structures [`short_rate/plot_06_duffie_kan_affine.py`]
14. 1997 -- BGM: caps and floors by Black's formula [`options/plot_03_bgm_caps_floors.py`]
15. 1997 -- Jamshidian's swap market model: Black swaptions [`options/plot_04_black_swaption.py`]
16. 2001 -- Brigo-Mercurio G2++ [`short_rate/plot_07_g2pp.py`]

### monte_carlo

1. 1934 -- Neyman stratification [`variance_reduction/plot_01_neyman_stratification.py`]
2. 1949 -- Metropolis-Ulam: the Monte Carlo method [`basics/plot_01_crude_monte_carlo.py`]
3. 1953 -- Kahn-Marshall importance sampling [`variance_reduction/plot_02_importance_sampling.py`]
4. 1956 -- Hammersley-Morton antithetic variates [`variance_reduction/plot_03_antithetic_variates.py`]
5. 1958 -- Box-Muller normal variates [`basics/plot_02_box_muller.py`]
6. 1960 -- Halton sequences [`quasi_monte_carlo/plot_01_halton.py`]
7. 1961 -- Koksma-Hlawka and discrepancy [`quasi_monte_carlo/plot_02_koksma_hlawka.py`]
8. 1967 -- Sobol sequences [`quasi_monte_carlo/plot_03_sobol.py`]
9. 1977 -- Boyle: options by Monte Carlo [`basics/plot_03_boyle_option_pricing.py`]
10. 1979 -- McKay-Beckman-Conover Latin hypercube [`variance_reduction/plot_04_latin_hypercube.py`]
11. 1981 -- Lavenberg-Welch control variates [`variance_reduction/plot_05_control_variates.py`]
12. 1987 -- Glynn: likelihood-ratio Greeks [`greeks/plot_01_likelihood_ratio.py`]
13. 1995 -- Barraquand moment matching [`variance_reduction/plot_06_moment_matching.py`]
14. 1995 -- Owen scrambling: randomized QMC [`quasi_monte_carlo/plot_04_owen_scrambling.py`]
15. 1995 -- Paskov-Traub: QMC in high dimension [`quasi_monte_carlo/plot_05_paskov_traub.py`]
16. 1996 -- Broadie-Glasserman pathwise Greeks [`greeks/plot_02_pathwise.py`]
17. 1996 -- Caflisch-Moskowitz Brownian bridge [`quasi_monte_carlo/plot_06_brownian_bridge.py`]
18. 2008 -- Giles multilevel Monte Carlo [`multilevel/plot_01_giles_mlmc.py`]

### exotic_options

1. 1973 -- Merton's down-and-out call [`barriers/plot_01_merton_down_and_out.py`]
2. 1978 -- Margrabe exchange option [`multi_asset/plot_01_margrabe.py`]
3. 1979 -- Goldman-Sosin-Gatto floating lookback [`path_dependent/plot_01_floating_lookback.py`]
4. 1979 -- Geske compound options [`compound/plot_01_geske_compound.py`]
5. 1982 -- Stulz options on the max and min [`multi_asset/plot_02_stulz_max_min.py`]
6. 1990 -- Kemna-Vorst Asian options [`asian/plot_01_kemna_vorst.py`]
7. 1990 -- Rubinstein forward-start options [`compound/plot_02_forward_start.py`]
8. 1991 -- Reiner-Rubinstein barriers [`barriers/plot_02_reiner_rubinstein.py`]
9. 1991 -- Reiner-Rubinstein binaries [`binaries/plot_01_digital_options.py`]
10. 1991 -- Rubinstein chooser [`compound/plot_03_chooser.py`]
11. 1991 -- Conze-Viswanathan fixed lookback [`path_dependent/plot_02_fixed_lookback.py`]
12. 1991 -- Turnbull-Wakeman Asian approximation [`asian/plot_02_turnbull_wakeman.py`]
13. 1992 -- Ikeda-Kunitomo double barriers [`barriers/plot_03_double_barrier.py`]
14. 1992 -- Reiner quantos [`multi_asset/plot_03_quanto.py`]
15. 1997 -- Broadie-Glasserman-Kou discrete barriers [`barriers/plot_04_discrete_barrier_correction.py`]
16. 1999 -- Demeterfi-Derman-Kamal-Zou variance swaps [`path_dependent/plot_03_variance_swap.py`]
17. 2001 -- Longstaff-Schwartz [`american/plot_01_longstaff_schwartz.py`]

### pde_methods

1. 1911 -- Richardson extrapolation [`accuracy/plot_01_richardson_extrapolation.py`]
2. 1928 -- Courant-Friedrichs-Lewy [`stability/plot_01_cfl_explicit_blowup.py`]
3. 1947 -- Crank-Nicolson [`schemes/plot_01_crank_nicolson.py`]
4. 1949 -- Laasonen's implicit scheme [`schemes/plot_02_laasonen_implicit.py`]
5. 1949 -- Thomas algorithm [`schemes/plot_03_thomas_algorithm.py`]
6. 1950 -- Von Neumann stability analysis [`stability/plot_02_von_neumann.py`]
7. 1955 -- Peaceman-Rachford ADI [`multi_factor/plot_01_peaceman_rachford_adi.py`]
8. 1971 -- Cryer's projected SOR [`american/plot_01_cryer_psor.py`]
9. 1973 -- Black-Scholes as the heat equation [`schemes/plot_04_heat_equation_transform.py`]
10. 1977 -- Schwartz: explicit finite differences [`schemes/plot_05_schwartz_explicit.py`]
11. 1977 -- Brennan-Schwartz American put [`american/plot_02_brennan_schwartz.py`]
12. 1978 -- Brennan-Schwartz: explicit schemes are trinomial trees [`schemes/plot_06_explicit_is_trinomial.py`]
13. 1984 -- Rannacher smoothing [`stability/plot_03_rannacher_smoothing.py`]
14. 2000 -- Tavella-Randall non-uniform grids [`accuracy/plot_02_tavella_randall_grid.py`]
15. 2000 -- Zvan-Forsyth-Vetzal barrier options [`schemes/plot_07_barrier_pde.py`]
16. 2002 -- Forsyth-Vetzal penalty method [`american/plot_03_penalty_method.py`]

### stochastic_volatility

1. 1975 -- Cox CEV [`local_vol/plot_01_cox_cev.py`]
2. 1976 -- Merton jump diffusion [`jumps/plot_01_merton_jump_diffusion.py`]
3. 1987 -- Hull-White stochastic volatility [`heston/plot_01_hull_white_mixing.py`]
4. 1990 -- Madan-Seneta variance gamma [`jumps/plot_02_madan_seneta_variance_gamma.py`]
5. 1993 -- Heston [`heston/plot_02_heston.py`]
6. 1994 -- Dupire local volatility [`local_vol/plot_02_dupire.py`]
7. 1996 -- Bates jumps with stochastic volatility [`jumps/plot_03_bates.py`]
8. 1998 -- Madan-Carr-Chang skewed variance gamma [`jumps/plot_04_madan_carr_chang.py`]
9. 1999 -- Carr-Madan FFT [`fourier/plot_01_carr_madan_fft.py`]
10. 2001 -- Lewis's single-integral formula [`fourier/plot_02_lewis_formula.py`]
11. 2002 -- Hagan SABR [`smile/plot_01_sabr.py`]
12. 2002 -- Kou double-exponential jumps [`jumps/plot_05_kou.py`]
13. 2004 -- Gatheral SVI [`smile/plot_02_svi.py`]
14. 2004 -- Lee's moment formula [`smile/plot_03_lee_moment_formula.py`]
15. 2008 -- Andersen QE scheme [`simulation/plot_01_andersen_qe.py`]
16. 2008 -- Fang-Oosterlee COS method [`fourier/plot_03_cos_method.py`]
17. 2010 -- Lord-Koekkoek-van Dijk full truncation [`simulation/plot_02_full_truncation.py`]

### time_series

1. 1927 -- Yule's autoregression [`linear_models/plot_01_yule_autoregression.py`]
2. 1951 -- Hurst exponent [`long_memory/plot_01_hurst_exponent.py`]
3. 1953 -- Kendall: the random walk of prices [`efficiency/plot_01_kendall_autocorrelation.py`]
4. 1963 -- Mandelbrot: fat tails [`stylized_facts/plot_01_mandelbrot_fat_tails.py`]
5. 1965 -- Fama's runs test [`efficiency/plot_02_fama_runs_test.py`]
6. 1970 -- Box-Jenkins ARMA [`linear_models/plot_02_box_jenkins_arma.py`]
7. 1974 -- Akaike information criterion [`linear_models/plot_03_akaike_order_selection.py`]
8. 1975 -- Hill tail index [`stylized_facts/plot_02_hill_tail_index.py`]
9. 1978 -- Ljung-Box portmanteau test [`efficiency/plot_03_ljung_box.py`]
10. 1979 -- Dickey-Fuller unit roots [`unit_roots/plot_01_dickey_fuller.py`]
11. 1980 -- Parkinson's range volatility [`volatility/plot_01_parkinson_range.py`]
12. 1982 -- Engle's ARCH [`volatility/plot_02_engle_arch.py`]
13. 1986 -- Bollerslev's GARCH [`volatility/plot_03_bollerslev_garch.py`]
14. 1987 -- Engle-Granger cointegration [`unit_roots/plot_02_engle_granger.py`]
15. 1988 -- Lo-MacKinlay variance ratio [`efficiency/plot_04_lo_mackinlay.py`]
16. 1991 -- Nelson's EGARCH [`volatility/plot_04_nelson_egarch.py`]
17. 1993 -- Glosten-Jagannathan-Runkle [`volatility/plot_05_gjr_garch.py`]
18. 1994 -- RiskMetrics EWMA [`volatility/plot_06_riskmetrics_ewma.py`]
19. 1998 -- Andersen-Bollerslev realized volatility [`volatility/plot_07_realized_volatility.py`]

### risk

1. 1928 -- Fisher-Tippett extreme values [`extreme_value/plot_01_fisher_tippett_gev.py`]
2. 1937 -- Cornish-Fisher expansion [`var/plot_01_cornish_fisher.py`]
3. 1952 -- Roy's safety-first criterion [`var/plot_02_roy_safety_first.py`]
4. 1959 -- Sklar's copulas [`copulas/plot_01_sklar_gaussian_copula.py`]
5. 1975 -- Pickands-Balkema-de Haan peaks over threshold [`extreme_value/plot_02_peaks_over_threshold.py`]
6. 1991 -- Young's Calmar ratio and drawdowns [`drawdowns/plot_01_calmar_drawdowns.py`]
7. 1994 -- RiskMetrics parametric VaR [`var/plot_03_parametric_var.py`]
8. 1995 -- Kupiec's proportion-of-failures test [`backtesting/plot_01_kupiec.py`]
9. 1996 -- Basel historical VaR and the traffic light [`backtesting/plot_02_basel_traffic_light.py`]
10. 1997 -- Pritsker: Monte Carlo VaR [`var/plot_04_monte_carlo_var.py`]
11. 1998 -- Boudoukh-Richardson-Whitelaw hybrid VaR [`var/plot_05_hybrid_var.py`]
12. 1998 -- Christoffersen's independence test [`backtesting/plot_03_christoffersen.py`]
13. 1999 -- Artzner-Delbaen-Eber-Heath coherence [`expected_shortfall/plot_01_coherence.py`]
14. 1999 -- Barone-Adesi filtered historical simulation [`var/plot_06_filtered_historical.py`]
15. 1999 -- Tasche: Euler risk allocation [`expected_shortfall/plot_02_euler_allocation.py`]
16. 2002 -- Acerbi-Tasche expected shortfall [`expected_shortfall/plot_03_acerbi_tasche.py`]
17. 2005 -- Demarta-McNeil t copula [`copulas/plot_02_t_copula_tail_dependence.py`]

### portfolio

1. 1952 -- Markowitz mean-variance [`mean_variance/plot_01_markowitz_frontier.py`]
2. 1956 -- Kelly criterion [`growth/plot_01_kelly_criterion.py`]
3. 1958 -- Tobin separation [`mean_variance/plot_02_tobin_tangency.py`]
4. 1963 -- Sharpe's single-index model [`covariance/plot_01_single_index.py`]
5. 1966 -- The Sharpe ratio [`performance/plot_01_sharpe_ratio.py`]
6. 1969 -- Merton's portfolio problem [`growth/plot_02_merton_portfolio.py`]
7. 1972 -- Merton's analytic frontier [`mean_variance/plot_03_merton_analytic_frontier.py`]
8. 1987 -- Black-Jones CPPI [`growth/plot_03_cppi.py`]
9. 1989 -- Grinold's fundamental law [`performance/plot_02_fundamental_law.py`]
10. 1992 -- Black-Litterman [`bayesian/plot_01_black_litterman.py`]
11. 1994 -- Sortino ratio [`performance/plot_03_sortino_ratio.py`]
12. 1998 -- Michaud resampled efficiency [`bayesian/plot_02_michaud_resampling.py`]
13. 2000 -- Rockafellar-Uryasev CVaR optimization [`risk_based/plot_01_cvar_optimization.py`]
14. 2004 -- Ledoit-Wolf shrinkage [`covariance/plot_02_ledoit_wolf.py`]
15. 2008 -- Choueifaty most-diversified portfolio [`risk_based/plot_02_most_diversified.py`]
16. 2009 -- DeMiguel-Garlappi-Uppal 1/N [`mean_variance/plot_04_one_over_n.py`]
17. 2010 -- Maillard-Roncalli-Teiletche risk parity [`risk_based/plot_03_risk_parity.py`]
18. 2016 -- Lopez de Prado hierarchical risk parity [`risk_based/plot_04_hierarchical_risk_parity.py`]

### asset_pricing

1. 1938 -- Williams's dividend discount model [`valuation/plot_01_williams_ddm.py`]
2. 1958 -- Modigliani-Miller [`valuation/plot_02_modigliani_miller.py`]
3. 1959 -- Gordon growth [`valuation/plot_03_gordon_growth.py`]
4. 1964 -- Sharpe's CAPM [`capm/plot_01_security_market_line.py`]
5. 1968 -- Jensen's alpha [`capm/plot_02_jensen_alpha.py`]
6. 1972 -- Black's zero-beta CAPM [`capm/plot_03_zero_beta.py`]
7. 1973 -- Fama-MacBeth regressions [`factors/plot_01_fama_macbeth.py`]
8. 1976 -- Ross's APT [`factors/plot_02_ross_apt.py`]
9. 1978 -- Lucas's tree [`consumption/plot_01_lucas_tree.py`]
10. 1979 -- Harrison-Kreps: the stochastic discount factor [`sdf/plot_01_harrison_kreps.py`]
11. 1979 -- Breeden's consumption CAPM [`consumption/plot_02_breeden_ccapm.py`]
12. 1981 -- Shiller's excess volatility [`valuation/plot_04_shiller_excess_volatility.py`]
13. 1985 -- Mehra-Prescott equity premium puzzle [`consumption/plot_03_equity_premium_puzzle.py`]
14. 1989 -- Gibbons-Ross-Shanken test [`capm/plot_04_grs_test.py`]
15. 1991 -- Hansen-Jagannathan bound [`sdf/plot_02_hansen_jagannathan.py`]
16. 1993 -- Fama-French three factors [`factors/plot_03_fama_french.py`]
17. 1997 -- Carhart momentum [`factors/plot_04_carhart_momentum.py`]

### credit

1. 1968 -- Altman Z-score [`scoring/plot_01_altman_z_score.py`]
2. 1974 -- Merton's structural model [`structural/plot_01_merton_structural.py`]
3. 1976 -- Black-Cox first passage [`structural/plot_02_black_cox.py`]
4. 1987 -- Vasicek's large homogeneous portfolio [`portfolio_loss/plot_01_vasicek_lhp.py`]
5. 1994 -- Leland's endogenous default [`structural/plot_03_leland.py`]
6. 1995 -- Jarrow-Turnbull hazard rates [`reduced_form/plot_01_jarrow_turnbull.py`]
7. 1997 -- CreditMetrics transition matrices [`migration/plot_01_creditmetrics.py`]
8. 1997 -- Jarrow-Lando-Turnbull generator matrices [`migration/plot_02_generator_matrix.py`]
9. 1997 -- CreditRisk+ and Panjer's recursion [`portfolio_loss/plot_02_creditrisk_plus.py`]
10. 1999 -- Duffie-Singleton recovery of market value [`reduced_form/plot_02_duffie_singleton.py`]
11. 2000 -- Li's Gaussian copula [`portfolio_loss/plot_03_li_gaussian_copula.py`]
12. 2001 -- Israel-Rosenthal-Wei: finding generators [`migration/plot_03_israel_rosenthal_wei.py`]
13. 2003 -- Crosbie-Bohn distance to default [`structural/plot_04_distance_to_default.py`]
14. 2003 -- O'Kane-Turnbull CDS par spread [`cds/plot_01_cds_par_spread.py`]
15. 2004 -- Basel II IRB formula [`portfolio_loss/plot_04_basel_irb.py`]
16. 2004 -- Hull-White tranche pricing [`portfolio_loss/plot_05_tranche_expected_loss.py`]
17. 2009 -- The ISDA standard model: hazard bootstrapping [`cds/plot_02_isda_hazard_bootstrap.py`]

### market_microstructure

1. 1968 -- Demsetz: the bid-ask spread [`spreads/plot_01_demsetz_spreads.py`]
2. 1976 -- Garman's order-flow dealer [`dealers/plot_01_garman_inventory.py`]
3. 1981 -- Ho-Stoll dealer inventory [`dealers/plot_02_ho_stoll.py`]
4. 1984 -- Roll's spread estimator [`spreads/plot_02_roll_estimator.py`]
5. 1985 -- Kyle's insider [`information/plot_01_kyle.py`]
6. 1985 -- Glosten-Milgrom [`information/plot_02_glosten_milgrom.py`]
7. 1991 -- Hasbrouck's trade-impact VAR [`information/plot_03_hasbrouck_var.py`]
8. 1991 -- Lee-Ready trade classification [`spreads/plot_03_lee_ready.py`]
9. 1996 -- Easley-Kiefer-O'Hara-Paperman PIN [`information/plot_04_pin.py`]
10. 1997 -- The square-root impact law [`execution/plot_01_square_root_impact.py`]
11. 1998 -- Bertsimas-Lo optimal execution [`execution/plot_02_bertsimas_lo.py`]
12. 2000 -- Almgren-Chriss [`execution/plot_03_almgren_chriss.py`]
13. 2002 -- Amihud illiquidity [`liquidity/plot_01_amihud.py`]
14. 2008 -- Avellaneda-Stoikov market making [`dealers/plot_03_avellaneda_stoikov.py`]
15. 2012 -- Corwin-Schultz high-low spread [`spreads/plot_04_corwin_schultz.py`]
16. 2012 -- VPIN [`information/plot_05_vpin.py`]

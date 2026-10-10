"""Regenerate the README's per-subpackage teaser figures.

    python docs/make_readme_subpackage_figures.py                  # all subpackages
    python docs/make_readme_subpackage_figures.py options credit   # just these

Writes docs/source/_static/images/readme_<subpackage>.png, three panels each,
at the same size as readme_hero.png (see make_readme_figure.py). README.md
embeds them by their raw.githubusercontent.com URLs so they also render on PyPI.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "source" / "_static" / "images"


def time_value(axes):
    from financialkit.time_value import amortization_schedule, npv_profile, plot_amortization, plot_compounding_growth, plot_npv_profile

    ax1, ax2, ax3 = axes
    plot_compounding_growth(1.0, 1.0, compoundings=(1, 2, 4, 12, None), ax=ax1)
    ax1.set_title("Bernoulli: compounding more often tends to e")

    plot_npv_profile(npv_profile([-1600.0, 10_000.0, -10_000.0], np.linspace(0.0, 5.0, 501)), ax=ax2, label="pump project")
    ax2.set_title("Lorie-Savage: one project, two IRRs")
    ax2.legend()

    plot_amortization(amortization_schedule(200_000.0, 0.06, 360), ax=ax3)
    ax3.set_title("A 30-year mortgage: interest first, principal later")


def fixed_income(axes):
    from financialkit.fixed_income import NelsonSiegelCurve, bond_cash_flows, fit_nelson_siegel, simulate_yield_changes, yield_curve_pca
    from financialkit.fixed_income.visualizers import plot_pca_loadings, plot_price_yield, plot_yield_curve

    ax1, ax2, ax3 = axes
    plot_price_yield(bond_cash_flows(100.0, 0.04, 30.0), 0.04, dy=0.04, ax=ax1)
    ax1.set_title("Price-yield convexity of a 30-year bond")

    rng = np.random.default_rng(1987)
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    observed = NelsonSiegelCurve(0.048, -0.022, 0.025, 1.6).zero_rate(tenors) + rng.normal(0, 0.0005, tenors.size)
    plot_yield_curve(fit_nelson_siegel(tenors, observed).curve, ax=ax2, label="Nelson-Siegel fit")
    ax2.plot(tenors, 100 * observed, "ko", ms=4, label="observed")
    ax2.set_title("A Nelson-Siegel zero curve")
    ax2.legend()

    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    plot_pca_loadings(yield_curve_pca(simulate_yield_changes(tenors, n_obs=2500, rng=1991), tenors), ax=ax3)
    ax3.set_title("Level, slope and curvature (Litterman-Scheinkman)")


def options(axes):
    from financialkit.options import black_scholes_greeks, black_scholes_price, implied_vol_smile
    from financialkit.options.visualizers import plot_smile

    ax1, ax2, ax3 = axes
    S = np.linspace(60, 140, 300)
    for T in (1.0, 0.5, 0.1):
        ax1.plot(S, black_scholes_price(S, 100.0, T, 0.05, 0.25), label=f"T = {T:g}")
    ax1.plot(S, np.maximum(S - 100.0, 0.0), "k--", lw=1, label="payoff")
    ax1.set_xlabel("spot S")
    ax1.set_ylabel("call value")
    ax1.set_title("Black-Scholes: value decays to the payoff")
    ax1.legend()

    g = black_scholes_greeks(S, 100.0, 0.5, 0.05, 0.25)
    ax2.plot(S, g.delta, label="delta")
    ax2.plot(S, 20 * g.gamma, label="gamma (x20)")
    ax2.plot(S, g.vega / 30, label="vega (/30)")
    ax2.axvline(100.0, color="k", lw=0.5)
    ax2.set_xlabel("spot S")
    ax2.set_title("The Greeks of a call")
    ax2.legend()

    S0, T, r = 100.0, 0.5, 0.02
    K = np.linspace(65, 135, 29)
    flat = implied_vol_smile(black_scholes_price(S0, K, T, r, 0.2), K, S0, T, r)
    smile = implied_vol_smile(0.5 * black_scholes_price(S0, K, T, r, 0.12) + 0.5 * black_scholes_price(S0, K, T, r, 0.32), K, S0, T, r)
    up = (1 - 0.1 * 0.75) / 0.9
    skew_prices = 0.1 * black_scholes_price(S0 * 0.75, K, T, r, 0.3) + 0.9 * black_scholes_price(S0 * up, K, T, r, 0.15)
    skew = implied_vol_smile(skew_prices, K, S0, T, r)
    plot_smile(flat, ax=ax3, label="lognormal: flat", moneyness=True)
    plot_smile(smile, ax=ax3, label="vol mixture: smile", moneyness=True)
    plot_smile(skew, ax=ax3, label="crash risk: skew", moneyness=True)
    ax3.set_title("Implied volatility from prices")
    ax3.legend()


def interest_rate_models(axes):
    from financialkit.fixed_income import NelsonSiegelCurve
    from financialkit.interest_rate_models import VasicekModel, black_derman_toy_tree
    from financialkit.interest_rate_models.visualizers import plot_model_yields, plot_rate_tree, plot_short_rate_paths

    ax1, ax2, ax3 = axes
    for r0, shape in ((0.01, "normal"), (0.0555, "humped"), (0.09, "inverted")):
        plot_model_yields(VasicekModel(r0, 0.25, 0.06, 0.02), t_max=30, ax=ax1, label=f"r0 = {100 * r0:.2f}% ({shape})")
    ax1.set_title("Vasicek yield curves")
    ax1.legend()

    times, r = VasicekModel(0.01, 0.25, 0.06, 0.02).simulate(20.0, 200, 2000, rng=0)
    plot_short_rate_paths(times, r, n_show=12, ax=ax2)
    ax2.set_title("Mean-reverting short-rate paths")

    plot_rate_tree(black_derman_toy_tree(NelsonSiegelCurve(0.05, -0.025, 0.02, 1.5), 0.2, 5.0, 10), ax=ax3)
    ax3.set_title("A Black-Derman-Toy tree fitted to the curve")


def monte_carlo(axes):
    from financialkit.monte_carlo import european_option_mc, halton, sobol
    from financialkit.monte_carlo.visualizers import plot_error_convergence, plot_point_set
    from financialkit.options import black_scholes_price

    ax1, ax2, ax3 = axes
    plot_point_set(sobol(64, 2, scramble=False), ax=ax1, grid=8, title="64 Sobol points: one per cell")

    d = 8
    f = lambda u: np.prod(1 + 0.5 * (u - 0.5), axis=1)  # noqa: E731
    ns = 2 ** np.arange(6, 15)
    errors = {
        "random": [abs(f(np.random.default_rng(n).random((n, d))).mean() - 1) for n in ns],
        "Halton": [abs(f(halton(n, d)).mean() - 1) for n in ns],
        "Sobol": [abs(f(sobol(n, d, rng=0)).mean() - 1) for n in ns],
    }
    for name, e in errors.items():
        plot_error_convergence(ns, e, ax=ax2, label=name, rates=(1.0,) if name == "Sobol" else ())
    ax2.set_title(f"Quasi-Monte Carlo beats random in d = {d}")

    S, T, r, sigma = 42.0, 0.5, 0.10, 0.20
    strikes = np.linspace(30, 55, 11)
    mc = [european_option_mc(S, k, T, r, sigma, n_paths=5000, rng=1) for k in strikes]
    ax3.plot(strikes, black_scholes_price(S, strikes, T, r, sigma), "k", label="Black-Scholes")
    ax3.errorbar(strikes, [m.estimate for m in mc], yerr=[m.half_width for m in mc], fmt="o", capsize=3, label="Monte Carlo, 95% CI")
    ax3.set_xlabel("strike K")
    ax3.set_ylabel("call price")
    ax3.set_title("Boyle: option prices with error bars")
    ax3.legend()


def exotic_options(axes):
    from financialkit.exotic_options import longstaff_schwartz, merton_down_and_out_call, plot_barrier_paths, plot_exercise_boundary, plot_price_profile
    from financialkit.integrators import gbm_paths
    from financialkit.options import CRRTree, black_scholes_price

    ax1, ax2, ax3 = axes
    t, paths = gbm_paths(100.0, 0.05, 0.25, 0.5, 100, 40, rng=0)
    plot_barrier_paths(t, paths, [80.0, 130.0], ax=ax1)
    ax1.set_title("Paths knocked out by a double barrier")

    S = np.linspace(70.0, 140.0, 200)
    K, H, T, r, sigma = 100.0, 90.0, 1.0, 0.05, 0.25
    plot_price_profile(
        S, {"vanilla call": black_scholes_price(S, K, T, r, sigma), "down-and-out call": merton_down_and_out_call(S, K, H, T, r, sigma)}, ax=ax2, markers=(H, K)
    )
    ax2.set_title("Merton's down-and-out call")

    S, K, T, r, sigma = 36.0, 40.0, 1.0, 0.06, 0.2
    plot_exercise_boundary(longstaff_schwartz(S, K, T, r, sigma, n_exercise=50, n_paths=20_000, rng=1), ax=ax3)
    tree = CRRTree(500).price(S, K, T, r, sigma, option_type="put", american=True)
    ax3.plot(tree.times, tree.exercise_boundary, "k", lw=1, label="CRR tree")
    ax3.set_ylim(30, 41)
    ax3.set_title("Longstaff-Schwartz: the American exercise boundary")
    ax3.legend()


def pde_methods(axes):
    from financialkit.pde_methods import CrankNicolson, von_neumann_amplification
    from financialkit.pde_methods.visualizers import plot_amplification, plot_exercise_boundary, plot_value_surface

    ax1, ax2, ax3 = axes
    plot_value_surface(CrankNicolson(200, 100, rannacher_steps=2).price(100.0, 100.0, 1.0, 0.05, 0.2), ax=ax1, s_range=(40, 180))
    ax1.set_title("Crank-Nicolson: the call value V(S, t)")

    sigma, r, dx = 0.2, 0.05, 0.02
    lams = [0.5, 0.9, 1.2]
    res = [von_neumann_amplification(sigma, r, lam * dx**2 / sigma**2, dx, theta=0.0) for lam in lams]
    plot_amplification(res, labels=[f"explicit, $\\lambda$ = {lam}" for lam in lams], ax=ax2)
    ax2.set_title("Von Neumann stability of the explicit scheme")

    res = CrankNicolson(200, 100, rannacher_steps=2, american_method="psor").price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True)
    plot_exercise_boundary(res, ax=ax3)
    ax3.set_title("PSOR: the American put's free boundary")


def stochastic_volatility(axes):
    from financialkit.stochastic_volatility import HestonModel, cev_local_vol, cev_price, dupire_local_vol, heston_paths
    from financialkit.stochastic_volatility.visualizers import plot_local_vol, plot_smile

    ax1, ax2, ax3 = axes
    model = HestonModel(0.04, 1.5, 0.04, 0.5, -0.7)
    paths = heston_paths(model, 100.0, 1.0, 0.02, n_steps=250, n_paths=8, rng=0)
    ax1.plot(paths.times, np.sqrt(np.maximum(paths.variance, 0.0)).T, lw=0.8)
    ax1.set_xlabel("t")
    ax1.set_ylabel("volatility sqrt(v)")
    ax1.set_title("Heston: volatility is itself random")

    K = np.linspace(70, 135, 27)
    for xi in (0.2, 0.5, 0.9):
        plot_smile(K, HestonModel(0.04, 1.5, 0.04, xi, 0.0).implied_vol(100.0, K, 1.0, 0.02), forward=100.0 * np.exp(0.02), ax=ax2, label=f"xi = {xi}")
    ax2.set_title("Vol of vol curves the smile")
    ax2.legend()

    S, r, beta, sigma = 100.0, 0.03, 0.5, 2.5
    K = np.linspace(50, 180, 131)
    T = np.linspace(0.25, 1.5, 51)
    lv = dupire_local_vol(K, T, cev_price(S, K[None, :], T[:, None], r, sigma, beta), r)
    plot_local_vol(lv, maturities=[0.5, 1.0, 1.5], ax=ax3)
    ax3.plot(K, cev_local_vol(K, sigma, beta), "k--", lw=1, label="true CEV local vol")
    ax3.set_ylim(0.1, 0.4)
    ax3.set_title("Dupire local volatility from call prices")
    ax3.legend()


def time_series(axes):
    from scipy import stats

    from financialkit.constants import TRADING_DAYS_PER_YEAR
    from financialkit.time_series import GARCH
    from financialkit.time_series.visualizers import plot_acf, plot_conditional_volatility, plot_tail

    ax1, ax2, ax3 = axes
    rng = np.random.default_rng(1963)
    n = 20_000
    for name, x in (
        ("normal", rng.standard_normal(n)),
        ("Student t(3)", stats.t.rvs(3, size=n, random_state=rng) / np.sqrt(3)),
        ("stable (alpha 1.7)", stats.levy_stable.rvs(1.7, 0.0, size=n, random_state=rng) / np.sqrt(2)),
    ):
        plot_tail(x, ax=ax1, label=name)
    ax1.set_ylim(1 / n, 1)
    ax1.set_xlim(0.1, 200)
    ax1.set_title("Mandelbrot: fat tails are power laws")
    ax1.legend()

    true = GARCH(omega=0.16**2 / TRADING_DAYS_PER_YEAR * 0.02, alpha=0.08, beta=0.90, mu=3e-4)
    r = true.simulate(2000, rng=1986).returns
    plot_conditional_volatility(r, GARCH.fit(r).variance, ax=ax2)
    ax2.set_title("GARCH(1,1) conditional volatility")

    plot_acf(r**2, 20, ax=ax3)
    ax3.set_title("Volatility clustering: ACF of squared returns")


def risk(axes):
    from financialkit.options import black_scholes_price
    from financialkit.risk import monte_carlo_var, rolling_historical_var, t_copula_sample
    from financialkit.risk.visualizers import plot_backtest, plot_copula, plot_var

    ax1, ax2, ax3 = axes
    S0, T, r, sigma = 100.0, 0.25, 0.03, 0.25
    K = S0 * np.exp((r + 0.5 * sigma**2) * T)
    h = 10 / 252

    def straddle(S, t):
        return black_scholes_price(S, K, t, r, sigma) + black_scholes_price(S, K, t, r, sigma, option_type="put")

    def pnl(dX):
        return -100.0 * (straddle(S0 * np.exp(dX[:, 0]), T - h) - straddle(S0, T))

    plot_var(monte_carlo_var(pnl, np.array([[sigma**2]]), 0.99, horizon=h, n_paths=20_000, rng=0), ax=ax1)
    ax1.set_title("Monte Carlo VaR of short straddles")

    rng = np.random.default_rng(7)
    vol = np.r_[np.full(750, 0.01), np.full(500, 0.02)]
    daily = vol * rng.standard_t(5, vol.size) / np.sqrt(5 / 3)
    plot_backtest(daily[250:], rolling_historical_var(daily, 250, 0.99), ax=ax2)
    ax2.set_title("Backtesting 99% historical VaR")

    plot_copula(t_copula_sample(np.array([[1.0, 0.5], [0.5, 1.0]]), 4, 3000, rng=1), ax=ax3, title="A t copula: joint crashes")


def portfolio(axes):
    from financialkit.integrators import gbm_paths
    from financialkit.portfolio import cppi, diversification_ratio, most_diversified_portfolio, risk_parity
    from financialkit.portfolio.visualizers import plot_cppi, plot_weights

    ax1, ax2, ax3 = axes
    names = ["govt", "IG", "HY", "US eq", "EU eq", "EM eq", "gold", "REITs"]
    vol = np.array([0.05, 0.06, 0.10, 0.16, 0.18, 0.22, 0.16, 0.20])
    corr = np.array(
        [
            [1.0, 0.7, 0.1, -0.2, -0.2, -0.1, 0.2, 0.1],
            [0.7, 1.0, 0.5, 0.2, 0.2, 0.3, 0.2, 0.3],
            [0.1, 0.5, 1.0, 0.6, 0.6, 0.6, 0.1, 0.6],
            [-0.2, 0.2, 0.6, 1.0, 0.85, 0.75, 0.0, 0.7],
            [-0.2, 0.2, 0.6, 0.85, 1.0, 0.8, 0.05, 0.65],
            [-0.1, 0.3, 0.6, 0.75, 0.8, 1.0, 0.15, 0.6],
            [0.2, 0.2, 0.1, 0.0, 0.05, 0.15, 1.0, 0.1],
            [0.1, 0.3, 0.6, 0.7, 0.65, 0.6, 0.1, 1.0],
        ]
    )
    cov = corr * np.outer(vol, vol)
    rp = risk_parity(cov)
    plot_weights(rp.weights, names, ax=ax1)
    ax1.set_title("Risk parity: equal risk contributions")

    mdp = most_diversified_portfolio(cov)
    plot_weights(mdp.weights, names, ax=ax2)
    ax2.set_title(f"Most diversified portfolio (D = {diversification_ratio(mdp.weights, cov):.2f})")

    times, S = gbm_paths(100.0, 0.07, 0.25, 3.0, 3 * 252, 400, rng=1)
    plot_cppi(cppi(S, times, floor=0.8, multiplier=4.0, r=0.02), ax=ax3, n_paths=40)
    ax3.set_title("CPPI: wealth paths above a floor")


def asset_pricing(axes):
    from financialkit.asset_pricing import LucasTree, capm_equilibrium, hansen_jagannathan_bound, variance_bound
    from financialkit.asset_pricing.visualizers import plot_excess_volatility, plot_hj_bound, plot_sml

    ax1, ax2, ax3 = axes
    mu = np.array([0.055, 0.065, 0.085, 0.095, 0.12])
    vol = np.array([0.14, 0.15, 0.20, 0.28, 0.30])
    cov = (0.35 + 0.65 * np.eye(5)) * np.outer(vol, vol)
    plot_sml(capm_equilibrium(mu, cov, risk_free=0.03), ax=ax1, labels=["utilities", "staples", "industrials", "energy", "tech"])
    ax1.set_title("The CAPM security market line")

    mu = np.array([1.0698, 1.0080])
    vol = np.array([0.1654, 0.0567])
    bound = hansen_jagannathan_bound(mu, np.array([[1.0, 0.1], [0.1, 1.0]]) * np.outer(vol, vol), np.linspace(0.85, 1.05, 200))
    models = {f"CRRA, gamma = {g}": LucasTree(0.99, g, 0.018, 0.036).moments() for g in (2, 10, 25)}
    plot_hj_bound(bound, models, ax=ax2)
    ax2.set_ylim(0, 1.2)
    ax2.set_title("Hansen-Jagannathan: the equity premium puzzle")

    rng = np.random.default_rng(1981)
    r, rho, Dbar, n = 0.05, 0.9, 1.0, 400
    D = np.empty(n)
    D[0] = Dbar
    for t in range(1, n):
        D[t] = Dbar + rho * (D[t - 1] - Dbar) + 0.05 * rng.standard_normal()
    plot_excess_volatility(variance_bound(D / r, D, r, terminal_price=Dbar / r), ax=ax3, label="price = D / r")
    ax3.set_title("Shiller: prices more volatile than dividends allow")


def credit(axes):
    from financialkit.credit import RATINGS, bootstrap_hazard_curve, creditmetrics_matrix, creditrisk_plus
    from financialkit.credit.visualizers import plot_loss_distribution, plot_survival, plot_transition_matrix

    ax1, ax2, ax3 = axes
    maturities = np.array([0.5, 1, 2, 3, 4, 5, 7, 10])
    spreads = np.array([45, 55, 80, 105, 125, 140, 155, 165]) * 1e-4
    plot_survival(bootstrap_hazard_curve(maturities, spreads, 0.03, 0.4), T=10.0, ax=ax1, label="survival")
    ax1.set_title("Survival curve bootstrapped from CDS spreads")
    ax1.legend(loc="lower left")

    rng = np.random.default_rng(0)
    exposures = np.round(rng.lognormal(np.log(2e5), 0.9, 300), -4)
    pds = np.clip(rng.lognormal(np.log(0.01), 0.8, 300), 1e-4, 0.2)
    plot_loss_distribution(creditrisk_plus(exposures, pds, sigma=1.0, unit=50_000.0), alpha=0.999, ax=ax2, log=True)
    ax2.set_xlim(0, 2.5e7)
    ax2.set_ylim(1e-8, 1)
    ax2.set_title("CreditRisk+ portfolio loss distribution")

    plot_transition_matrix(creditmetrics_matrix(), labels=RATINGS, fmt="{:.1f}", ax=ax3)
    ax3.set_title("CreditMetrics one-year rating migrations (%)")


def market_microstructure(axes):
    from financialkit.market_microstructure import AvellanedaStoikov, almgren_chriss, simulate_glosten_milgrom, simulate_market_making
    from financialkit.market_microstructure.visualizers import plot_execution, plot_glosten_milgrom

    ax1, ax2, ax3 = axes
    lams = [0.0, 2e-7, 2e-6, 2e-5]
    plot_execution([almgren_chriss(1e6, 5.0, 5, 0.95, 2.5e-6, lam, 2.5e-7, 0.0625) for lam in lams], labels=[f"$\\lambda$ = {lam:g}" for lam in lams], ax=ax1)
    ax1.set_xlabel("days")
    ax1.set_title("Almgren-Chriss: optimal liquidation")

    plot_glosten_milgrom(simulate_glosten_milgrom(80, alpha=0.2, v_low=90.0, v_high=110.0, true_value=110.0, rng=9), ax=ax2)
    ax2.set_title("Glosten-Milgrom: quotes learn the value")

    res = simulate_market_making(AvellanedaStoikov(0.1, 2.0, 1.5, 1.0), 100.0, 2.0, 1.0, 200, 140.0, 1.5, n_paths=1, rng=5)
    ax3.plot(res.times, res.mid[0], "k", lw=1, label="mid")
    ax3.plot(res.times[:-1], res.ask[0], color="C1", lw=0.8, label="ask")
    ax3.plot(res.times[:-1], res.bid[0], color="C0", lw=0.8, label="bid")
    ax3.set_xlabel("time")
    ax3.set_ylabel("price")
    ax3.set_title("Avellaneda-Stoikov market making")
    ax3.legend()


FIGURES = {
    f.__name__: f
    for f in (
        time_value,
        fixed_income,
        options,
        interest_rate_models,
        monte_carlo,
        exotic_options,
        pde_methods,
        stochastic_volatility,
        time_series,
        risk,
        portfolio,
        asset_pricing,
        credit,
        market_microstructure,
    )
}


def main(names):
    for name in names or FIGURES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        FIGURES[name](axes)
        out = OUT_DIR / f"readme_{name}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])

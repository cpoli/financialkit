"""Option pricers: closed forms, discrete dividends, parity and bounds,
Greeks and hedging, implied volatility, the vanna-volga smile and
densities, lattices and the American approximations.
"""

from financialkit.options.systems.american import barone_adesi_whaley, bjerksund_stensland
from financialkit.options.systems.closed_form import bachelier_price, black76_price, black_scholes_price, garman_kohlhagen_price
from financialkit.options.systems.density import breeden_litzenberger_density
from financialkit.options.systems.dividends import discrete_dividend_price
from financialkit.options.systems.greeks import black_scholes_greeks
from financialkit.options.systems.hedging import delta_hedge
from financialkit.options.systems.implied_vol import bachelier_implied_volatility, brenner_subrahmanyam_vol, implied_vol_smile, implied_volatility
from financialkit.options.systems.lets_be_rational import normalised_black_call, normalised_implied_volatility
from financialkit.options.systems.parity import no_arbitrage_bounds, parity_residual, put_call_parity
from financialkit.options.systems.trees import BinomialTree, CRRTree, LeisenReimerTree, RendlemanBartterTree, TrinomialTree
from financialkit.options.systems.vanna_volga import fx_pillar_strikes, vanna_volga_price, vanna_volga_smile

__all__ = [
    "bachelier_price",
    "black_scholes_price",
    "black76_price",
    "garman_kohlhagen_price",
    "put_call_parity",
    "parity_residual",
    "no_arbitrage_bounds",
    "black_scholes_greeks",
    "delta_hedge",
    "implied_volatility",
    "bachelier_implied_volatility",
    "normalised_black_call",
    "normalised_implied_volatility",
    "brenner_subrahmanyam_vol",
    "implied_vol_smile",
    "fx_pillar_strikes",
    "vanna_volga_price",
    "vanna_volga_smile",
    "breeden_litzenberger_density",
    "BinomialTree",
    "CRRTree",
    "RendlemanBartterTree",
    "LeisenReimerTree",
    "TrinomialTree",
    "barone_adesi_whaley",
    "bjerksund_stensland",
    "discrete_dividend_price",
]

"""The shared input checks, and the uniform messages the pricers raise through them."""

import re

import numpy as np
import pytest

from financialkit import exotic_options as eo
from financialkit import interest_rate_models as irm
from financialkit import options as op
from financialkit._validation import finite, in_interval, nonnegative, one_of, option_type, positive
from financialkit.options.utils.common import check_nonnegative, option_sign


def test_sign_checks_name_the_first_bad_entry():
    positive(S=[1.0, 2.0], K=np.nan)
    nonnegative(T=[0.0, 1.0])
    with pytest.raises(ValueError, match=r"^K must be positive, got 0$"):
        positive(S=1.0, K=[2.0, 0.0, -1.0])
    with pytest.raises(ValueError, match=r"^T must be non-negative, got -0.25$"):
        nonnegative(T=[[1.0], [-0.25]])


def test_finite_rejects_nan_and_inf():
    finite(r=[0.0, -0.01])
    with pytest.raises(ValueError, match=r"^r must be finite, got nan$"):
        finite(r=np.nan)
    with pytest.raises(ValueError, match=r"^q must be finite, got -inf$"):
        finite(r=0.05, q=[0.0, -np.inf])


@pytest.mark.parametrize(
    ("closed", "inside", "outside", "brackets"),
    [("neither", 0.5, 0.0, "(0, 1)"), ("left", 0.0, 1.0, "[0, 1)"), ("right", 1.0, 0.0, "(0, 1]"), ("both", [0.0, 1.0], 1.5, "[0, 1]")],
)
def test_in_interval_respects_closed_ends(closed, inside, outside, brackets):
    in_interval(0.0, 1.0, closed=closed, x=inside)
    with pytest.raises(ValueError, match=rf"^x must lie in {re.escape(brackets)}, got {outside:g}$"):
        in_interval(0.0, 1.0, closed=closed, x=outside)
    with pytest.raises(ValueError, match="closed must be one of"):
        in_interval(0.0, 1.0, closed="open", x=inside)


def test_choice_checks():
    one_of("kind", "cap", ("cap", "floor"))
    with pytest.raises(ValueError, match=r"^kind must be 'cap' or 'floor', got 'swap'$"):
        one_of("kind", "swap", ("cap", "floor"))
    assert (option_type("call"), option_type("put")) == (1, -1)
    with pytest.raises(ValueError, match=r"^underlying_type must be 'call' or 'put', got 'Call'$"):
        option_type("Call", name="underlying_type")


def test_options_helpers_delegate_to_the_shared_checks():
    assert option_sign("put") == -1
    with pytest.raises(ValueError, match=r"^S must be non-negative, got -1$"):
        check_nonnegative(S=-1.0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: op.black_scholes_price(-1.0, 100.0, 1.0, 0.05, 0.2), "S must be non-negative, got -1"),
        (lambda: op.black_scholes_greeks(100.0, 100.0, 1.0, 0.05, [0.2, -0.1]), "sigma must be non-negative, got -0.1"),
        (lambda: op.barone_adesi_whaley(100.0, 100.0, 0.0, 0.05, 0.2), "T must be positive, got 0"),
        (lambda: op.CRRTree(10).price(100.0, 0.0, 1.0, 0.05, 0.2), "K must be positive, got 0"),
        (lambda: op.black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="straddle"), "option_type must be 'call' or 'put', got 'straddle'"),
        (lambda: eo.barrier_price(100.0, 100.0, 0.0, 1.0, 0.05, 0.2), "H must be positive, got 0"),
        (lambda: eo.one_touch_price(100.0, -90.0, 1.0, 0.05, 0.2), "H must be positive, got -90"),
        (lambda: eo.floating_lookback_price(100.0, 1.0, 0.05, 0.2, S_extreme=0.0), "S_extreme must be positive, got 0"),
        (lambda: eo.stulz_price(100.0, 100.0, 100.0, 1.0, 0.05, 0.2, 0.3, 0.5, kind="mean"), "kind must be 'max' or 'min', got 'mean'"),
        (lambda: eo.compound_option_price(100.0, 5.0, 0.0, 0.5, 1.0, 0.05, 0.2), "K2 must be positive, got 0"),
        (lambda: eo.bivariate_normal_cdf(0.0, 0.0, 1.0), "rho must lie in (-1, 1), got 1"),
        (lambda: irm.VasicekModel(0.03, 0.0, 0.05, 0.01), "kappa must be positive, got 0"),
        (lambda: irm.CIRModel(-0.01, 0.5, 0.05, 0.1), "r0 must be non-negative, got -0.01"),
        (lambda: irm.HullWhiteModel(0.03, 0.1, -0.01), "sigma must be non-negative, got -0.01"),
        (lambda: irm.G2PlusPlusModel(0.03, 0.5, 0.01, 0.1, 0.01, -1.0), "rho must lie in (-1, 1), got -1"),
        (lambda: irm.cap_floor_price(0.03, 0.03, 2.0, 0.2, kind="collar"), "kind must be 'cap' or 'floor', got 'collar'"),
        (lambda: irm.black_derman_toy_tree(0.03, 0.2, 1.0, 4).bond_option(0.5, 1.0, 0.98, "straddle"), "option_type must be 'call' or 'put', got 'straddle'"),
    ],
)
def test_pricers_raise_uniform_messages(call, message):
    with pytest.raises(ValueError) as err:
        call()
    assert str(err.value) == message

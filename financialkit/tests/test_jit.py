"""The optional-numba shim: njit is numba.njit when installed, a no-op otherwise."""

import financialkit.integrators as integrators
from financialkit import _jit


def test_integrators_reexport_the_shim():
    assert integrators.njit is _jit.njit
    assert integrators.HAS_NUMBA is _jit.HAS_NUMBA


def test_identity_njit_bare_decorator_returns_function_unchanged():
    def f(x):
        return 2 * x

    assert _jit._identity_njit(f) is f
    assert f(3) == 6


def test_identity_njit_with_options_returns_function_unchanged():
    def f(x):
        return 2 * x

    assert _jit._identity_njit(cache=True)(f) is f
    assert _jit._identity_njit(cache=False)(f) is f
    assert f(3) == 6

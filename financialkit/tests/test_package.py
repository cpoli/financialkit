"""Package-level smoke tests: financialkit imports cleanly and every declared
subpackage is actually importable and internally consistent."""

import financialkit as fk


def test_top_level_import_exposes_declared_subpackages():
    for name in fk.__all__:
        assert hasattr(fk, name), f"financialkit.{name} listed in __all__ but not importable"


def test_version_is_a_string():
    assert isinstance(fk.__version__, str)
    assert fk.__version__.count(".") == 2


def test_domain_versions_match_top_level_version():
    """Every domain subpackage's own __version__ should track the
    top-level package version (mirrors physicskit's convention)."""
    for name in fk.__all__:
        module = getattr(fk, name)
        if hasattr(module, "__version__"):
            assert module.__version__ == fk.__version__


def test_constants_module_has_expected_names():
    assert fk.constants.TRADING_DAYS_PER_YEAR == 252
    assert fk.constants.CALENDAR_DAYS_PER_YEAR == 365
    assert fk.constants.DEFAULT_MAX_ITER > 0
    assert len(fk.constants.DayCount) == 5


def test_integrators_module_has_expected_callables():
    assert callable(fk.integrators.euler_maruyama)
    assert callable(fk.integrators.milstein)
    assert callable(fk.integrators.gbm_paths)


def test_time_value_module_has_expected_callables():
    assert callable(fk.time_value.npv)
    assert callable(fk.time_value.irr)
    assert callable(fk.time_value.year_fraction)

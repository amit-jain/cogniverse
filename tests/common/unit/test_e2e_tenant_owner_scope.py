"""Every tenant an e2e fixture or test mints is deleted when its owner ends.

``unique_id`` / ``own_tenant`` bind the tenant to the fixture or test body
executing at mint time: a module-scoped fixture's tenant lives until module
teardown (also when the fixture's setup fails after the mint), a test body's
tenant is deleted before the next test starts. The synthetic session below
records every mint, delete and test boundary and pins the exact sequence.
"""

from __future__ import annotations

import pytest

from tests.e2e import conftest as e2e_conftest

pytest_plugins = ["pytester"]

_SYNTHETIC_CONFTEST = """
import functools
from pathlib import Path

import pytest

from tests.e2e import conftest as e2e_conftest

EVENTS = Path(__file__).parent / "events.log"


def _record(event):
    with EVENTS.open("a") as fh:
        fh.write(event + "\\n")


def _fake_delete(minted):
    _record(f"delete {minted.rsplit('_', 1)[0]}")


e2e_conftest.delete_minted_tenant_and_wait = _fake_delete


def mint(prefix):
    tid = e2e_conftest.unique_id(prefix)
    _record(f"mint {tid.rsplit('_', 1)[0]}")
    return tid


pytest_fixture_setup = e2e_conftest.pytest_fixture_setup
pytest_runtest_call = e2e_conftest.pytest_runtest_call


def pytest_runtest_logstart(nodeid, location):
    _record(f"start {nodeid.rsplit('::', 1)[-1]}")


def pytest_runtest_logfinish(nodeid, location):
    _record(f"finish {nodeid.rsplit('::', 1)[-1]}")
"""

_SYNTHETIC_MODULE = """
import pytest

from conftest import mint


@pytest.fixture(scope="module")
def module_tenant():
    return mint("modscope")


@pytest.fixture(scope="module")
def broken_module_tenant():
    mint("broken")
    raise RuntimeError("setup failed after the mint")


def test_a(module_tenant):
    mint("fn_a")


def test_b(module_tenant):
    pass


def test_c(broken_module_tenant):
    pass


def test_d():
    pass
"""


def test_minted_tenants_are_deleted_when_their_owner_scope_ends(pytester):
    pytester.makeconftest(_SYNTHETIC_CONFTEST)
    pytester.makepyfile(test_owned=_SYNTHETIC_MODULE)

    result = pytester.runpytest("-p", "no:cacheprovider", "-q")

    result.assert_outcomes(passed=3, errors=1)
    events = (pytester.path / "events.log").read_text().splitlines()
    assert events == [
        "start test_a",
        "mint modscope",
        "mint fn_a",
        "delete fn_a",
        "finish test_a",
        "start test_b",
        "finish test_b",
        "start test_c",
        "mint broken",
        "finish test_c",
        "start test_d",
        "delete broken",
        "delete modscope",
        "finish test_d",
    ]


def test_own_tenant_refuses_the_shared_seeded_tenant():
    with pytest.raises(ValueError) as raised:
        e2e_conftest.own_tenant("flywheel_org:anything")
    assert str(raised.value) == (
        "'flywheel_org:anything' belongs to the shared seeded tenant "
        "'flywheel_org:production', which the session keeps; mint a test "
        "tenant with unique_id()"
    )


def test_own_tenant_outside_any_scope_has_no_owner():
    assert e2e_conftest._TENANT_OWNERS == []
    with pytest.raises(RuntimeError) as raised:
        e2e_conftest.own_tenant("opt_deadbeef:t1")
    assert str(raised.value) == (
        "own_tenant('opt_deadbeef:t1') called outside fixture setup or a test "
        "body: no scope can tear it down"
    )

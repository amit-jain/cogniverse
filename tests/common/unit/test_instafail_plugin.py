"""The instant-failure reporter prints each failing report as it happens."""

pytest_plugins = ["pytester"]

_CHAIN_FLAGS = ("--tb=long", "-q", "-p", "no:cacheprovider")


def _outcome_lines(lines):
    return [
        line for line in lines if line.startswith(("FAILED", "ERROR", "PASSED", "."))
    ]


def test_call_failure_traceback_prints_before_the_summary(pytester):
    pytester.makepyfile(
        test_sample="""
        def test_boom():
            assert 1 == 2

        def test_fine():
            assert True
        """
    )
    result = pytester.runpytest("-p", "tests.fixtures.instafail", *_CHAIN_FLAGS)
    lines = result.outlines

    header = [
        i
        for i, line in enumerate(lines)
        if "call failure: test_sample.py::test_boom" in line
    ]
    summary = [
        i
        for i, line in enumerate(lines)
        if line.startswith("=") and " FAILURES " in line
    ]
    assert len(header) == 1
    assert len(summary) == 1
    assert header[0] < summary[0]
    assert sum(line == "E       assert 1 == 2" for line in lines) == 2
    result.assert_outcomes(passed=1, failed=1)


def test_setup_error_traceback_prints_immediately(pytester):
    pytester.makepyfile(
        test_setup_sample="""
        import pytest

        @pytest.fixture
        def broken():
            raise RuntimeError("fixture exploded")

        def test_uses_broken(broken):
            assert True
        """
    )
    result = pytester.runpytest("-p", "tests.fixtures.instafail", *_CHAIN_FLAGS)
    lines = result.outlines

    header = [
        i
        for i, line in enumerate(lines)
        if "setup failure: test_setup_sample.py::test_uses_broken" in line
    ]
    summary = [
        i for i, line in enumerate(lines) if line.startswith("=") and " ERRORS " in line
    ]
    assert len(header) == 1
    assert len(summary) == 1
    assert header[0] < summary[0]
    assert sum(line == "E       RuntimeError: fixture exploded" for line in lines) == 2
    result.assert_outcomes(errors=1)


def test_passing_run_prints_no_failure_headers(pytester):
    pytester.makepyfile(
        test_green="""
        def test_ok():
            assert True
        """
    )
    result = pytester.runpytest("-p", "tests.fixtures.instafail", *_CHAIN_FLAGS)
    assert [line for line in result.outlines if " failure: " in line] == []
    result.assert_outcomes(passed=1)

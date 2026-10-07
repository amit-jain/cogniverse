"""The node-order plugin runs a module's tests reversed or shuffled."""

import os
import re
from pathlib import Path

from tests.fixtures.node_order import ordered

REPO_ROOT = Path(__file__).resolve().parents[3]

pytest_plugins = ["pytester"]

_FLAGS = ("-p", "tests.fixtures.node_order", "-p", "no:cacheprovider", "-q")
_NAMES = [f"test_{index}" for index in range(8)]


def _module(pytester):
    """Eight tests that record, through a module-level list, the order they ran."""
    body = "RAN = []\n\n" + "".join(
        f"def {name}():\n    RAN.append({name!r})\n\n" for name in _NAMES
    )
    pytester.makepyfile(test_order=body)
    pytester.makeconftest(
        """
        import json, os

        def pytest_sessionfinish(session):
            import test_order
            with open(os.environ["RAN_OUT"], "w") as out:
                json.dump(test_order.RAN, out)
        """
    )


def _run(pytester, monkeypatch, *args):
    import json

    out = pytester.path / "ran.json"
    monkeypatch.setenv("RAN_OUT", str(out))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])),
    )
    result = pytester.runpytest_subprocess(*_FLAGS, *args)
    result.assert_outcomes(passed=len(_NAMES))
    return result, json.loads(out.read_text())


def test_collected_order_is_unchanged_by_default(pytester, monkeypatch):
    _module(pytester)
    _, ran = _run(pytester, monkeypatch)
    assert ran == _NAMES


def test_reverse_runs_last_first_and_prints_the_order(pytester, monkeypatch):
    _module(pytester)
    result, ran = _run(pytester, monkeypatch, "--node-order=reverse")
    assert ran == list(reversed(_NAMES))
    header = result.outlines.index("node order: reverse")
    assert result.outlines[header + 1 : header + 1 + len(_NAMES)] == [
        f"  test_order.py::{name}" for name in reversed(_NAMES)
    ]


def test_a_seeded_shuffle_runs_the_order_its_seed_draws(pytester, monkeypatch):
    _module(pytester)
    result, ran = _run(
        pytester, monkeypatch, "--node-order=shuffle", "--node-order-seed=7"
    )
    assert ran == ordered(_NAMES, "shuffle", 7)
    assert ran != _NAMES and ran != list(reversed(_NAMES))
    assert "node order: shuffle (seed 7)" in result.outlines


def test_an_unseeded_shuffle_prints_the_seed_that_replays_it(pytester, monkeypatch):
    _module(pytester)
    result, ran = _run(pytester, monkeypatch, "--node-order=shuffle")
    seeds = [
        int(match.group(1))
        for line in result.outlines
        if (match := re.fullmatch(r"node order: shuffle \(seed (\d+)\)", line))
    ]
    assert len(seeds) == 1
    assert ran == ordered(_NAMES, "shuffle", seeds[0])
    _, replayed = _run(
        pytester, monkeypatch, "--node-order=shuffle", f"--node-order-seed={seeds[0]}"
    )
    assert replayed == ran

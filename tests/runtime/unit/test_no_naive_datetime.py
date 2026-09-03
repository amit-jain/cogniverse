"""Persisted timestamps in the runtime are timezone-aware.

A naive ``datetime.now()`` persisted or returned by the API is wrong by
the host offset on any non-UTC machine, and comparing it against an
aware ``fromisoformat()`` value raises TypeError. Generic detector: it
fails on any future naive ``datetime.now()`` in the runtime package.
"""

import pathlib
import re

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_RUNTIME_ROOT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "libs"
    / "runtime"
    / "cogniverse_runtime"
)

_NAIVE_NOW = re.compile(r"datetime\.now\(\)")


def test_runtime_has_no_naive_datetime_now():
    offenders = []
    for path in _RUNTIME_ROOT.rglob("*.py"):
        for lineno, line in enumerate(
            path.read_text(errors="replace").splitlines(), start=1
        ):
            if _NAIVE_NOW.search(line):
                offenders.append(f"{path.relative_to(_RUNTIME_ROOT)}:{lineno}")
    assert offenders == []

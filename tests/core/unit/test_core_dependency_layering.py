"""Core sits below the evaluation stack and must not depend on it.

cogniverse-evaluation depends on foundation + sdk; core declaring
cogniverse-evaluation was a backwards edge (higher layer pulled by a lower
one). Core's code imports nothing from cogniverse_evaluation, so the
dependency was also unused. This pins the layering so it cannot regress.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

pytestmark = [pytest.mark.unit]


def _firstparty_deps(dist: str) -> set[str]:
    pyproject = REPO / "libs" / dist.replace("cogniverse-", "") / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    return {
        d.split(">")[0].split("=")[0].split("[")[0].strip()
        for d in data.get("project", {}).get("dependencies", [])
        if d.startswith("cogniverse-")
    }


def test_core_does_not_depend_on_evaluation():
    assert "cogniverse-evaluation" not in _firstparty_deps("cogniverse-core")


def test_evaluation_depends_on_core_direction_is_downward():
    # Sanity check the correct direction still holds: evaluation may depend on
    # the lower layers, never the reverse.
    eval_deps = _firstparty_deps("cogniverse-evaluation")
    assert (
        "cogniverse-core" not in eval_deps
    )  # evaluation builds on foundation, not core
    assert "cogniverse-foundation" in eval_deps

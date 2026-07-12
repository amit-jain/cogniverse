"""cogniverse_agents must declare the first-party packages it imports.

graph/graph_manager.py imports cogniverse_vespa at module-init time (via
graph/__init__.py), so ``import cogniverse_agents.graph`` fails with
ModuleNotFoundError on a from-wheel install unless cogniverse-vespa is a
declared dependency. This pins the declared transitive closure.
"""

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


def _closure(root_dist: str) -> set[str]:
    seen: set[str] = set()
    stack = [root_dist]
    while stack:
        for dep in _firstparty_deps(stack.pop()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return {c.replace("cogniverse-", "cogniverse_") for c in seen}


def test_agents_declares_cogniverse_vespa():
    closure = _closure("cogniverse-agents")
    assert "cogniverse_vespa" in closure, (
        "cogniverse_agents.graph imports cogniverse_vespa at module load, but "
        f"cogniverse-vespa is not in the declared dependency closure {sorted(closure)}"
    )

"""The runtime must declare its hard dependencies as base deps.

``main.py`` mounts the search and ingestion routers unconditionally, and
both import ``cogniverse_agents`` at module load
(``search.service.SearchService``, ``graph.graph_schema.Mention``). If
``cogniverse-agents`` is only an optional extra, a base
``pip install cogniverse-runtime`` produces an image that cannot import the
app to start it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import cogniverse_runtime


def _base_dep_names() -> set[str]:
    pyproject = (
        Path(cogniverse_runtime.__file__).resolve().parent.parent / "pyproject.toml"
    )
    data = tomllib.loads(pyproject.read_text())
    names = set()
    for spec in data["project"]["dependencies"]:
        # Strip version pins / extras / markers to the bare distribution name.
        name = spec.split(";")[0].split("[")[0]
        for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
            name = name.split(op)[0]
        names.add(name.strip())
    return names


@pytest.mark.unit
def test_runtime_declares_agents_as_base_dependency():
    assert "cogniverse-agents" in _base_dep_names()


@pytest.mark.unit
def test_runtime_declares_all_module_level_workspace_deps():
    # The routers imported by main.py at module load reach into these
    # workspace packages; each must be a base dependency, not an extra.
    base = _base_dep_names()
    for required in ("cogniverse-sdk", "cogniverse-core", "cogniverse-agents"):
        assert required in base, f"{required} missing from base dependencies"

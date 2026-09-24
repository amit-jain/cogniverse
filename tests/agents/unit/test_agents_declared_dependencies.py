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


SPACY_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
SPACY_MODEL_HASH = (
    "sha256:1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
)


def _requirement_names(specs: list[str]) -> list[str]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    return [canonicalize_name(Requirement(spec).name) for spec in specs]


def _locked_package(lock: dict, name: str) -> dict:
    matches = [p for p in lock["package"] if p["name"] == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def test_agents_exports_spacy_without_the_model_distribution():
    agents = tomllib.loads((REPO / "libs" / "agents" / "pyproject.toml").read_text())
    dependencies = agents["project"]["dependencies"]
    extras = agents["project"]["optional-dependencies"]

    assert dependencies.count("spacy==3.8.14") == 1
    assert "en-core-web-sm" not in _requirement_names(dependencies)
    for extra, specs in extras.items():
        assert "en-core-web-sm" not in _requirement_names(specs), extra
    assert "en-core-web-sm" not in agents["tool"]["uv"]["sources"]


def test_the_runtime_models_group_owns_the_model_artifact():
    root = tomllib.loads((REPO / "pyproject.toml").read_text())

    assert root["dependency-groups"]["runtime-models"] == ["en-core-web-sm==3.8.0"]
    assert root["tool"]["uv"]["sources"]["en-core-web-sm"] == {"url": SPACY_MODEL_URL}
    assert root["tool"]["uv"]["default-groups"] == ["dev", "runtime-models"]
    assert "en-core-web-sm" not in _requirement_names(root["project"]["dependencies"])


def test_the_lock_pins_the_model_wheel_to_the_group_only():
    lock = tomllib.loads((REPO / "uv.lock").read_text())

    assert _locked_package(lock, "en-core-web-sm") == {
        "name": "en-core-web-sm",
        "version": "3.8.0",
        "source": {"url": SPACY_MODEL_URL},
        "wheels": [{"url": SPACY_MODEL_URL, "hash": SPACY_MODEL_HASH}],
    }
    agents = _locked_package(lock, "cogniverse-agents")
    assert "en-core-web-sm" not in {d["name"] for d in agents["dependencies"]}
    assert "en-core-web-sm" not in {
        d["name"] for d in agents["metadata"]["requires-dist"]
    }
    workspace = _locked_package(lock, "cogniverse")
    assert [d["name"] for d in workspace["dev-dependencies"]["runtime-models"]] == [
        "en-core-web-sm"
    ]
    assert workspace["metadata"]["requires-dev"]["runtime-models"] == [
        {"name": "en-core-web-sm", "url": SPACY_MODEL_URL}
    ]

"""Released wheels install outside the workspace; the spaCy model is explicit.

``scripts/build_packages.sh`` builds the release set in a disposable Git
checkout. Each published root is then installed into a fresh virtualenv with
no workspace, project or uv source mappings: internal packages come only from
the built ``dist/`` and third-party packages from the public index. Relationship
analysis needs ``en_core_web_sm``, which is not a package dependency: without it
imports succeed and the feature raises ``SpaCyModelUnavailableError``; after the
README's hash-pinned install it parses a fixed sentence exactly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from tests.cli.integration.test_release_scripts import (
    _EXPECTED_RELEASE,
    _assert_release,
    _make_checkout,
    _output,
    _run_build,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
UV = shutil.which("uv")
VERSION = "0.2.0"

SPACY_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
SPACY_MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"

_AGENTS_CLOSURE = {
    "cogniverse-sdk",
    "cogniverse-foundation",
    "cogniverse-core",
    "cogniverse-synthetic",
    "cogniverse-vespa",
    "cogniverse-agents",
}
_APP_CLOSURE = _AGENTS_CLOSURE | {
    "cogniverse-evaluation",
    "cogniverse-telemetry-phoenix",
}
BASE_CLOSURES = {
    "cogniverse-agents": _AGENTS_CLOSURE,
    "cogniverse-runtime": _APP_CLOSURE | {"cogniverse-runtime"},
    "cogniverse-dashboard": _APP_CLOSURE | {"cogniverse-dashboard"},
}
ROOT_IMPORTS = {
    "cogniverse-agents": "cogniverse_agents",
    "cogniverse-runtime": "cogniverse_runtime.main",
    "cogniverse-dashboard": "cogniverse_dashboard",
}
THREADS = 8
# The lock's torch for Linux is the CPU build (``torch==2.8.0+cpu``), published
# only on the PyTorch CPU index; dashboard reaches it through embedding-atlas.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

_PROBE = r"""
import importlib
import importlib.metadata as md
import json
import sys
import threading
from collections import Counter

importlib.import_module(sys.argv[1])
from cogniverse_agents.routing.relationship_extraction_tools import (
    SpaCyDependencyAnalyzer,
    SpaCyModelUnavailableError,
)

TEXT = "Barack Obama visited Chicago in 2012"
analyzer = SpaCyDependencyAnalyzer()
outcomes = []
barrier = threading.Barrier(THREADS)


def call():
    barrier.wait()
    try:
        outcomes.append(("ok", analyzer.extract_semantic_relationships(TEXT)))
    except SpaCyModelUnavailableError as exc:
        outcomes.append(("raised", (
            exc.model_name, str(exc).split(" It doesn't")[0],
            type(exc.__cause__).__name__,
        )))


threads = [threading.Thread(target=call) for _ in range(THREADS)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

report = {
    "cogniverse": sorted(
        (d.metadata["Name"], d.version) for d in md.distributions()
        if d.metadata["Name"].startswith("cogniverse")
    ),
    "spacy": md.version("spacy"),
    "outcomes": sorted(
        [kind, value, count]
        for (kind, value), count in Counter(
            (kind, json.dumps(value, sort_keys=True)) for kind, value in outcomes
        ).items()
    ),
}
try:
    dist = md.distribution("en-core-web-sm")
except md.PackageNotFoundError:
    report["model"] = None
else:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    report["model"] = {
        "dist": [dist.metadata["Name"], dist.version],
        "direct_url": json.loads(dist.read_text("direct_url.json"))["url"],
        "meta": {k: nlp.meta[k] for k in ("lang", "name", "version", "spacy_version")},
        "pipeline": nlp.pipe_names,
        "deps": [[t.text, t.dep_, t.head.text] for t in nlp(TEXT)],
    }
print(json.dumps(report))
""".replace("THREADS", str(THREADS))

UNPROVISIONED_ERROR = [
    "en_core_web_sm",
    "spaCy model 'en_core_web_sm' could not be loaded: OSError: [E050] Can't find "
    "model 'en_core_web_sm'.",
    "OSError",
]
RELATIONSHIPS = [
    {
        "subject": "Obama",
        "relation": "visit",
        "object": "Chicago",
        "confidence": 0.8,
        "grammatical_pattern": "nsubj-ROOT-dobj",
    },
    {
        "subject": "visited",
        "relation": "in",
        "object": "2012",
        "confidence": 0.7,
        "grammatical_pattern": "prep-in",
    },
]
MODEL = {
    "dist": ["en_core_web_sm", "3.8.0"],
    "direct_url": SPACY_MODEL_URL,
    "meta": {
        "lang": "en",
        "name": "core_web_sm",
        "version": "3.8.0",
        "spacy_version": ">=3.8.0,<3.9.0",
    },
    "pipeline": ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    "deps": [
        ["Barack", "compound", "Obama"],
        ["Obama", "nsubj", "visited"],
        ["visited", "ROOT", "visited"],
        ["Chicago", "dobj", "visited"],
        ["in", "prep", "visited"],
        ["2012", "pobj", "in"],
    ],
}


def _clean_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH"}
    }


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=cwd, env=_clean_env(), capture_output=True, text=True, timeout=1800
    )


def _documented_model_requirement() -> str:
    readme = (REPO / "libs" / "agents" / "README.md").read_text()
    [requirement] = re.findall(
        r'^pip install "(en-core-web-sm @ [^"]+)"$', readme, re.M
    )
    return requirement


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> Path:
    repo = _make_checkout(tmp_path_factory.mktemp("release"), tag=f"v{VERSION}")
    result = _run_build(repo)
    assert result.returncode == 0, _output(result)
    manifest = _assert_release(repo, VERSION)
    assert {p["name"] for p in manifest["packages"]} == _EXPECTED_RELEASE
    return repo / "dist"


@pytest.fixture(scope="module")
def constraints(release) -> Path:
    """Internal packages at the release version; third-party at the locked set."""
    path = release.parent / "release-constraints.txt"
    exported = _run(
        [
            UV,
            "export",
            "--frozen",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
            "--no-emit-workspace",
            "--no-default-groups",
            "--all-packages",
            "--extra",
            "cpu",
            "-o",
            str(path),
        ],
        release.parent,
    )
    assert exported.returncode == 0, exported.stderr
    locked = path.read_text()
    assert "cogniverse" not in locked
    assert "en-core-web-sm" not in locked
    path.write_text(
        locked + "".join(f"{n}=={VERSION}\n" for n in sorted(_EXPECTED_RELEASE))
    )
    return path


def _installed_file_mismatches(site_packages: Path, wheel: Path) -> list[str]:
    mismatched = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.endswith(".dist-info/RECORD"):
                continue
            installed = site_packages / member
            if not installed.is_file() or installed.read_bytes() != archive.read(
                member
            ):
                mismatched.append(member)
    return mismatched


def test_the_agents_wheel_exports_spacy_but_not_the_model(release):
    wheel = release / f"cogniverse_agents-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        [member] = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = archive.read(member).decode()
    requires = [
        line.removeprefix("Requires-Dist: ")
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: ")
    ]

    assert "spacy==3.8.14" in requires
    assert "en-core-web-sm" not in {
        canonicalize_name(Requirement(line).name) for line in requires
    }
    assert [line for line in requires if "@" in line] == []


@pytest.mark.parametrize("root", sorted(BASE_CLOSURES))
def test_clean_install_resolves_from_release_and_needs_the_model_explicitly(
    root, release, constraints, tmp_path
):
    work = tmp_path / "outside-workspace"
    work.mkdir()
    venv = work / "venv"
    python = venv / "bin" / "python"
    created = _run(
        [UV, "venv", "--no-config", "--seed", "--python", "3.12", str(venv)], work
    )
    assert created.returncode == 0, created.stderr

    installed = _run(
        [
            UV,
            "pip",
            "install",
            "--no-config",
            "--python",
            str(python),
            "--find-links",
            str(release),
            "--constraints",
            str(constraints),
            "--extra-index-url",
            TORCH_CPU_INDEX,
            "--index-strategy",
            "unsafe-best-match",
            f"{root}=={VERSION}",
        ],
        work,
    )
    assert installed.returncode == 0, installed.stderr

    site_packages = venv / "lib" / "python3.12" / "site-packages"
    for name in sorted(BASE_CLOSURES[root]):
        wheel = release / f"{name.replace('-', '_')}-{VERSION}-py3-none-any.whl"
        assert _installed_file_mismatches(site_packages, wheel) == [], name

    before = _run([str(python), "-c", _PROBE, ROOT_IMPORTS[root]], work)
    assert before.returncode == 0, before.stderr
    assert json.loads(before.stdout) == {
        "cogniverse": sorted([n, VERSION] for n in BASE_CLOSURES[root]),
        "spacy": "3.8.14",
        "outcomes": [["raised", json.dumps(UNPROVISIONED_ERROR), THREADS]],
        "model": None,
    }

    requirement = _documented_model_requirement()
    assert (
        requirement == f"en-core-web-sm @ {SPACY_MODEL_URL}#sha256={SPACY_MODEL_SHA256}"
    )
    tampered = requirement.replace(SPACY_MODEL_SHA256, "0" * 64)
    refused = _run([str(python), "-m", "pip", "install", tampered], work)
    assert refused.returncode == 1
    assert "THESE PACKAGES DO NOT MATCH THE HASHES" in refused.stderr
    assert f"Expected sha256 {'0' * 64}" in refused.stderr
    assert f"Got        {SPACY_MODEL_SHA256}" in refused.stderr
    still_missing = _run([str(python), "-c", _PROBE, ROOT_IMPORTS[root]], work)
    assert still_missing.returncode == 0, still_missing.stderr
    assert json.loads(still_missing.stdout) == json.loads(before.stdout)

    provisioned = _run([str(python), "-m", "pip", "install", requirement], work)
    assert provisioned.returncode == 0, provisioned.stderr

    after = _run([str(python), "-c", _PROBE, ROOT_IMPORTS[root]], work)
    assert after.returncode == 0, after.stderr
    assert json.loads(after.stdout) == {
        "cogniverse": sorted([n, VERSION] for n in BASE_CLOSURES[root]),
        "spacy": "3.8.14",
        "outcomes": [["ok", json.dumps(RELATIONSHIPS, sort_keys=True), THREADS]],
        "model": MODEL,
    }

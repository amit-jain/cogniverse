"""Released wheels install outside the workspace; the spaCy model is explicit.

``scripts/build_packages.sh`` builds the release set in a disposable Git
checkout. Each published root is then installed into a fresh virtualenv with no
workspace, project or uv source mappings and no third-party constraints: internal
packages come only from the built ``dist/``, everything else from the public
index as a wheel consumer resolves it. pip installs with no flags; uv with exactly
the flag the root's README documents. Relationship analysis needs
``en_core_web_sm``, which is not a package dependency: without it imports succeed
and the feature raises ``SpaCyModelUnavailableError``; after the README's
hash-pinned install it parses a fixed sentence exactly.
"""

from __future__ import annotations

import hashlib
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

from tests.cli.integration.release_build import (
    EXPECTED_RELEASE,
    describe_run,
    make_checkout,
    run_build,
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
INSTALLERS = ("pip", "uv")

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
            exc.model_name, str(exc),
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
    "model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid "
    "path to a data directory. Install it as described under 'spaCy pipeline' in "
    "the cogniverse-agents README.",
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


_WORKSPACE_ENV = {
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "PYTHONPATH",
    "UV_CONSTRAINT",
    "UV_OVERRIDE",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_PRERELEASE",
    "PIP_CONSTRAINT",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_PRE",
}


def _run(argv: list[str], cwd: Path, extra_env: dict[str, str]):
    env = {k: v for k, v in os.environ.items() if k not in _WORKSPACE_ENV}
    return subprocess.run(
        argv,
        cwd=cwd,
        env={**env, **extra_env},
        capture_output=True,
        text=True,
        timeout=3600,
    )


def _documented_model_requirement() -> str:
    readme = (REPO / "libs" / "agents" / "README.md").read_text()
    [requirement] = re.findall(
        r'^pip install "(en-core-web-sm @ [^"]+)"$', readme, re.M
    )
    return requirement


def _documented_uv_flags(root: str) -> list[str]:
    if root == "cogniverse-agents":
        return []
    readme = (
        REPO / "libs" / root.removeprefix("cogniverse-") / "README.md"
    ).read_text()
    [flags] = re.findall(rf"`uv pip install ((?:--\S+ )+){root}`", readme)
    return flags.split()


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> Path:
    repo = make_checkout(tmp_path_factory.mktemp("release"), tag=f"v{VERSION}")
    result = run_build(repo)
    assert result.returncode == 0, describe_run(result)
    dist = repo / "dist"
    manifest = json.loads((dist / "BUILD_MANIFEST.json").read_text())
    assert manifest["version"] == VERSION
    assert {p["name"] for p in manifest["packages"]} == EXPECTED_RELEASE
    for package in manifest["packages"]:
        wheel = package["wheel"]
        assert package["version"] == VERSION, package["name"]
        assert wheel["filename"] == (
            f"{package['name'].replace('-', '_')}-{VERSION}-py3-none-any.whl"
        )
        digest = hashlib.sha256((dist / wheel["filename"]).read_bytes()).hexdigest()
        assert wheel["sha256"] == digest, wheel["filename"]
    return dist


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    """Installer caches private to this module, removed with everything they hold."""
    root = tmp_path_factory.mktemp("installer-caches")
    yield {"PIP_CACHE_DIR": str(root / "pip"), "UV_CACHE_DIR": str(root / "uv")}
    shutil.rmtree(root)


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


def test_the_readmes_document_exactly_one_uv_flag_for_the_phoenix_roots():
    assert _documented_uv_flags("cogniverse-agents") == []
    assert _documented_uv_flags("cogniverse-runtime") == ["--prerelease=allow"]
    assert _documented_uv_flags("cogniverse-dashboard") == ["--prerelease=allow"]


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


def _install(installer: str, root: str, python: Path, release: Path, work: Path, env):
    if installer == "pip":
        argv = [str(python), "-m", "pip", "install", "--find-links", str(release)]
    else:
        argv = [UV, "pip", "install", "--no-config", "--python", str(python)]
        argv += ["--find-links", str(release), *_documented_uv_flags(root)]
    return _run([*argv, f"{root}=={VERSION}"], work, env)


@pytest.mark.parametrize("installer", INSTALLERS)
@pytest.mark.parametrize("root", sorted(BASE_CLOSURES))
def test_clean_install_resolves_from_release_and_needs_the_model_explicitly(
    root, installer, release, caches, tmp_path
):
    work = tmp_path / "outside-workspace"
    work.mkdir()
    try:
        _clean_install_lifecycle(root, installer, release, work, caches)
    finally:
        shutil.rmtree(work)


def _clean_install_lifecycle(root, installer, release, work, caches):
    venv = work / "venv"
    python = venv / "bin" / "python"
    created = _run(
        [UV, "venv", "--no-config", "--seed", "--python", "3.12", str(venv)],
        work,
        caches,
    )
    assert created.returncode == 0, created.stderr

    installed = _install(installer, root, python, release, work, caches)
    assert installed.returncode == 0, installed.stderr

    site_packages = venv / "lib" / "python3.12" / "site-packages"
    for name in sorted(BASE_CLOSURES[root]):
        wheel = release / f"{name.replace('-', '_')}-{VERSION}-py3-none-any.whl"
        assert _installed_file_mismatches(site_packages, wheel) == [], name

    probe = [str(python), "-c", _PROBE, ROOT_IMPORTS[root]]
    before = _run(probe, work, caches)
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
    refused = _run([str(python), "-m", "pip", "install", tampered], work, caches)
    assert refused.returncode == 1
    assert "THESE PACKAGES DO NOT MATCH THE HASHES" in refused.stderr
    assert f"Expected sha256 {'0' * 64}" in refused.stderr
    assert f"Got        {SPACY_MODEL_SHA256}" in refused.stderr
    still_missing = _run(probe, work, caches)
    assert still_missing.returncode == 0, still_missing.stderr
    assert json.loads(still_missing.stdout) == json.loads(before.stdout)

    provisioned = _run([str(python), "-m", "pip", "install", requirement], work, caches)
    assert provisioned.returncode == 0, provisioned.stderr

    after = _run(probe, work, caches)
    assert after.returncode == 0, after.stderr
    assert json.loads(after.stdout) == {
        "cogniverse": sorted([n, VERSION] for n in BASE_CLOSURES[root]),
        "spacy": "3.8.14",
        "outcomes": [["ok", json.dumps(RELATIONSHIPS, sort_keys=True), THREADS]],
        "model": MODEL,
    }

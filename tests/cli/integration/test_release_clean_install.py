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
from packaging.version import Version

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
    "torch": next((d.version for d in md.distributions() if d.metadata["Name"] == "torch"), None),
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


_DISTRIBUTIONS_PROBE = (
    "import importlib.metadata as md, json; print(json.dumps(["
    "[d.metadata['Name'], d.version, d.requires or []] for d in md.distributions()]))"
)


def _prereleases(distributions, requested: list[str]) -> tuple[set[str], set[str]]:
    """Installed pre-releases, and those no requirement asks for explicitly.

    ``requested`` is the install line: the target and any extra requirements.
    A pre-release is explicit when an applicable requirement has a specifier
    that admits pre-releases (``graphql-core>=3.3.0a0``, ``x==0.65b0``; never
    ``!=``). pip admits only those without a flag; anything else came from
    admitting pre-releases wholesale. A requirement applies when its marker
    holds here for no extra or for an extra something installed its
    distribution with.
    """
    requires = {
        canonicalize_name(name): [Requirement(line) for line in lines]
        for name, _, lines in distributions
    }
    extras: dict[str, set[str]] = {}
    applicable: list[Requirement] = []
    pending = [(None, Requirement(line)) for line in requested]
    seen = set()
    while pending:
        owner, requirement = pending.pop()
        if (owner, str(requirement)) in seen:
            continue
        if requirement.marker is not None and not any(
            requirement.marker.evaluate({"extra": extra})
            for extra in {"", *extras.get(owner, set())}
        ):
            continue
        seen.add((owner, str(requirement)))
        applicable.append(requirement)
        name = canonicalize_name(requirement.name)
        added = set(requirement.extras) - extras.get(name, set())
        extras.setdefault(name, set()).update(requirement.extras)
        pending.extend((name, dependency) for dependency in requires.get(name, []))
        if added:
            # Requirements of this distribution skipped before its extras
            # were known are reconsidered.
            seen = {entry for entry in seen if entry[0] != name}
    explicit = {
        canonicalize_name(requirement.name)
        for requirement in applicable
        if any(spec.prereleases for spec in requirement.specifier)
    }
    installed = {
        canonicalize_name(name)
        for name, version, _ in distributions
        if Version(version).is_prerelease
    }
    return installed, installed - explicit


def test_only_an_applicable_pre_release_specifier_requests_a_pre_release():
    distributions = [
        [
            "root",
            "1.0",
            [
                "s3fs",
                "opentelemetry-proto",
                "boto3",
                "x",
                "google-cloud[logging]",
                "graphql-core",
            ],
        ],
        ["s3fs", "2025.9.0", ["aiohttp!=4.0.0a0,!=4.0.0a1"]],
        ["aiohttp", "4.0.0a2", []],
        ["opentelemetry-proto", "1.36.0", ['protobuf<6.0.0dev; extra == "protobuf"']],
        ["protobuf", "6.0.0rc1", []],
        ["boto3", "1.40.0", ["botocore[crt]<2.0a0; extra == 'crt'"]],
        ["botocore", "2.0.0a1", []],
        ["x", "1.0", ['y==0.65b0; python_version < "3"', "z==0.65b0"]],
        ["y", "0.65b0", []],
        ["z", "0.65b0", []],
        [
            "google-cloud",
            "1.0",
            ['opentelemetry-exporter-gcp-logging>=1.9.0a0; extra == "logging"'],
        ],
        ["opentelemetry-exporter-gcp-logging", "1.9.0a0", []],
        ["graphql-core", "3.3.0rc1", []],
    ]

    installed, unrequested = _prereleases(
        distributions, ["root", "graphql-core>=3.3.0a0"]
    )

    assert installed == {
        "aiohttp",
        "protobuf",
        "botocore",
        "y",
        "z",
        "opentelemetry-exporter-gcp-logging",
        "graphql-core",
    }
    assert unrequested == {"aiohttp", "protobuf", "botocore", "y"}


_INHERITED_ENV = {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}


def _run(argv: list[str], cwd: Path, extra_env: dict[str, str]):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _INHERITED_ENV and not k.startswith(("UV_", "PIP_"))
    }
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


def _readme(root: str) -> str:
    return (REPO / "libs" / root.removeprefix("cogniverse-") / "README.md").read_text()


def _documented_pip_target(root: str) -> str:
    [target] = re.findall(
        rf"^pip install ({root}(?:\[[a-z,-]+\])?)$", _readme(root), re.M
    )
    return target


def _documented_uv_install(root: str) -> tuple[list[str], str]:
    """The extra requirements and target of the README's uv line; agents
    documents none."""
    lines = re.findall(
        rf'`uv pip install ({root}(?:\[[a-z,-]+\])?)((?: "[^"]+")*)`', _readme(root)
    )
    if not lines:
        return [], _documented_pip_target(root)
    [(target, requirements)] = lines
    return re.findall(r'"([^"]+)"', requirements), target


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
    yield {"UV_CACHE_DIR": str(root / "uv")}
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


def test_the_readmes_document_the_installs_the_clean_install_runs():
    assert _documented_pip_target("cogniverse-agents") == "cogniverse-agents"
    assert _documented_pip_target("cogniverse-runtime") == "cogniverse-runtime[vespa]"
    assert _documented_pip_target("cogniverse-dashboard") == "cogniverse-dashboard"
    assert _documented_uv_install("cogniverse-agents") == ([], "cogniverse-agents")
    assert _documented_uv_install("cogniverse-runtime") == (
        ["graphql-core>=3.3.0a0"],
        "cogniverse-runtime[vespa]",
    )
    assert _documented_uv_install("cogniverse-dashboard") == (
        ["graphql-core>=3.3.0a0"],
        "cogniverse-dashboard",
    )
    for root in ("cogniverse-runtime", "cogniverse-dashboard"):
        assert "--prerelease" not in _readme(root), root


_PLANTED_ENV = {
    "UV_INDEX": "unreachable=http://127.0.0.1:9/simple",
    "UV_DEFAULT_INDEX": "http://127.0.0.1:9/simple",
    "UV_INDEX_URL": "http://127.0.0.1:9/simple",
    "UV_EXTRA_INDEX_URL": "http://127.0.0.1:9/simple",
    "UV_FIND_LINKS": "/nonexistent-find-links",
    "UV_NO_INDEX": "1",
    "UV_INDEX_STRATEGY": "unsafe-best-match",
    "UV_TORCH_BACKEND": "cpu",
    "UV_EXCLUDE_NEWER": "2000-01-01T00:00:00Z",
    "UV_CONSTRAINT": "/nonexistent-constraints.txt",
    "UV_OVERRIDE": "/nonexistent-overrides.txt",
    "UV_PRERELEASE": "disallow",
    "PIP_NO_INDEX": "1",
    "PIP_INDEX_URL": "http://127.0.0.1:9/simple",
    "PIP_EXTRA_INDEX_URL": "http://127.0.0.1:9/simple",
    "PIP_FIND_LINKS": "/nonexistent-find-links",
    "PIP_CONSTRAINT": "/nonexistent-constraints.txt",
    "PIP_PRE": "1",
}
_NO_INDEX_CONFIG = {
    "pip/pip.conf": "[global]\nno-index = true\n",
    "uv/uv.toml": "no-index = true\n",
}


def test_parent_installer_settings_do_not_reach_the_clean_installs(
    tmp_path, monkeypatch, caches
):
    """Planted installer variables and user-level no-index configs are ignored.

    The controls show the planted user configs would block an index install
    if the installers read them.
    """
    xdg = tmp_path / "xdg"
    for relative, text in _NO_INDEX_CONFIG.items():
        (xdg / relative).parent.mkdir(parents=True)
        (xdg / relative).write_text(text)
    pip_config = tmp_path / "pip-config-file.conf"
    pip_config.write_text("[global]\nno-index = true\n")
    for name, value in {**_PLANTED_ENV, "PIP_CONFIG_FILE": str(pip_config)}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    created = _run(
        [UV, "venv", "--no-config", "--seed", "--python", "3.12", str(venv)],
        tmp_path,
        caches,
    )
    assert created.returncode == 0, created.stderr

    seen = _run([str(python), "-c", _INSTALLER_ENV_PROBE], tmp_path, caches)
    assert seen.returncode == 0, seen.stderr
    assert json.loads(seen.stdout) == ["UV_CACHE_DIR"]

    pip_dry_run = _run(
        [*_pip_install(python, caches), "--dry-run", "--no-deps", "packaging==26.0"],
        tmp_path,
        caches,
    )
    assert pip_dry_run.returncode == 0, pip_dry_run.stderr
    assert "Would install packaging-26.0" in pip_dry_run.stdout
    uv_dry_run = _run(
        [*_uv_pip_install(python), "--dry-run", "--no-deps", "packaging==26.0"],
        tmp_path,
        caches,
    )
    assert uv_dry_run.returncode == 0, uv_dry_run.stderr
    assert re.search(r"^\s*\+ packaging==26\.0$", uv_dry_run.stderr, re.MULTILINE), (
        uv_dry_run.stderr
    )

    pip_reads_config = _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-deps",
            "packaging==26.0",
        ],
        tmp_path,
        caches,
    )
    assert pip_reads_config.returncode == 1
    assert "No matching distribution found for packaging==26.0" in (
        pip_reads_config.stderr
    )
    uv_reads_config = _run(
        [
            UV,
            "pip",
            "install",
            "--python",
            str(python),
            "--dry-run",
            "--no-deps",
            "packaging==26.0",
        ],
        tmp_path,
        caches,
    )
    assert uv_reads_config.returncode == 1
    assert "packaging was not found in the provided package locations" in (
        uv_reads_config.stderr
    )


_INSTALLER_ENV_PROBE = (
    "import json, os; print(json.dumps(sorted("
    "k for k in os.environ if k.startswith(('UV_', 'PIP_')))))"
)


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


def _pip_install(python: Path, caches: dict[str, str]) -> list[str]:
    """pip ignoring environment variables and user/site config files."""
    cache_dir = Path(caches["UV_CACHE_DIR"]).parent / "pip"
    return [
        str(python),
        "-m",
        "pip",
        "--isolated",
        "install",
        "--cache-dir",
        str(cache_dir),
    ]


def _uv_pip_install(python: Path) -> list[str]:
    return [UV, "pip", "install", "--no-config", "--python", str(python)]


def _install(installer: str, root: str, python: Path, release: Path, work: Path, env):
    if installer == "pip":
        argv = [*_pip_install(python, env), "--find-links", str(release)]
        target = _documented_pip_target(root)
    else:
        requirements, target = _documented_uv_install(root)
        argv = [
            *_uv_pip_install(python),
            "--find-links",
            str(release),
            f"{target}=={VERSION}",
            *requirements,
        ]
        return _run(argv, work, env)
    return _run([*argv, f"{target}=={VERSION}"], work, env)


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

    listed = _run([str(python), "-c", _DISTRIBUTIONS_PROBE], work, caches)
    assert listed.returncode == 0, listed.stderr
    if installer == "uv":
        extra_requirements, target = _documented_uv_install(root)
    else:
        extra_requirements, target = [], _documented_pip_target(root)
    prereleases, unrequested = _prereleases(
        json.loads(listed.stdout), [target, *extra_requirements]
    )
    assert unrequested == set(), sorted(unrequested)
    if root != "cogniverse-agents":
        assert "graphql-core" in prereleases, sorted(prereleases)

    site_packages = venv / "lib" / "python3.12" / "site-packages"
    for name in sorted(BASE_CLOSURES[root]):
        wheel = release / f"{name.replace('-', '_')}-{VERSION}-py3-none-any.whl"
        assert _installed_file_mismatches(site_packages, wheel) == [], name

    probe = [str(python), "-c", _PROBE, ROOT_IMPORTS[root]]
    before = _run(probe, work, caches)
    assert before.returncode == 0, before.stderr
    before_report = json.loads(before.stdout)
    torch = before_report.pop("torch")
    if root == "cogniverse-dashboard":
        assert Version(torch).local is None, torch
        assert str(Version(torch)) == torch
    else:
        assert torch is None or Version(torch).local is None, torch
    assert before_report == {
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
    refused = _run([*_pip_install(python, caches), tampered], work, caches)
    assert refused.returncode == 1
    assert "THESE PACKAGES DO NOT MATCH THE HASHES" in refused.stderr
    words = f" {' '.join(refused.stderr.split())} "
    assert f" Expected sha256 {'0' * 64} " in words
    assert f" Got {SPACY_MODEL_SHA256} " in words
    still_missing = _run(probe, work, caches)
    assert still_missing.returncode == 0, still_missing.stderr
    assert json.loads(still_missing.stdout) == json.loads(before.stdout)

    provisioned = _run([*_pip_install(python, caches), requirement], work, caches)
    assert provisioned.returncode == 0, provisioned.stderr

    after = _run(probe, work, caches)
    assert after.returncode == 0, after.stderr
    after_report = json.loads(after.stdout)
    assert after_report.pop("torch") == torch
    assert after_report == {
        "cogniverse": sorted([n, VERSION] for n in BASE_CLOSURES[root]),
        "spacy": "3.8.14",
        "outcomes": [["ok", json.dumps(RELATIONSHIPS, sort_keys=True), THREADS]],
        "model": MODEL,
    }

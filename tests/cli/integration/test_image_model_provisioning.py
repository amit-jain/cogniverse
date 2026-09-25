"""The runtime and dashboard images provision the pinned spaCy model from the lock.

The model is not an exported dependency of cogniverse-agents; the root
``runtime-models`` dependency group owns it. Each image's builder stage runs
its package sync and then syncs that group from the frozen lock. These tests
run the exact ``uv sync`` commands the Dockerfiles contain into throwaway
environments and load the model they installed.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
UV = shutil.which("uv")

SPACY_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
SPACY_MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
MODEL_SYNC = "uv sync --only-group runtime-models --inexact --frozen"
IMAGE_SYNCS = {
    "runtime": [
        "uv sync --package cogniverse-runtime --extra all --no-dev --frozen",
        MODEL_SYNC,
    ],
    "dashboard": [
        "uv sync --package cogniverse-dashboard --no-dev --frozen",
        MODEL_SYNC,
    ],
}
TEXT = "Barack Obama visited Chicago in 2012"

_PROBE = r"""
import importlib.metadata as md
import json
import spacy
from cogniverse_agents.routing.relationship_extraction_tools import (
    SpaCyDependencyAnalyzer,
)

nlp = spacy.load("en_core_web_sm")
dist = md.distribution("en-core-web-sm")
doc = nlp(TEXT)
print(json.dumps({
    "dist": [dist.metadata["Name"], dist.version],
    "direct_url": json.loads(dist.read_text("direct_url.json"))["url"],
    "meta": {k: nlp.meta[k] for k in ("lang", "name", "version", "spacy_version")},
    "pipeline": nlp.pipe_names,
    "deps": [[t.text, t.dep_, t.head.text] for t in doc],
    "relationships": SpaCyDependencyAnalyzer().extract_semantic_relationships(TEXT),
}))
""".replace("TEXT", repr(TEXT))

EXPECTED_PROBE = {
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
    "relationships": [
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
    ],
}


def _uv_syncs(dockerfile: Path) -> list[str]:
    lines = dockerfile.read_text().splitlines()
    code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    joined = re.sub(r"\\\n\s*", " ", code)
    return [
        " ".join(match.split()) for match in re.findall(r"\buv sync\b[^&;\n]*", joined)
    ]


def _run(command: str, env_dir: Path, cwd: Path = REPO) -> subprocess.CompletedProcess:
    argv = shlex.split(command)
    argv[0] = UV
    return subprocess.run(
        argv,
        cwd=cwd,
        env={
            **{k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
            "UV_PROJECT_ENVIRONMENT": str(env_dir),
        },
        capture_output=True,
        text=True,
        timeout=900,
    )


def _freeze(env_dir: Path) -> set[str]:
    out = subprocess.run(
        [UV, "pip", "freeze", "--python", str(env_dir / "bin" / "python")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(out.splitlines())


def _provision(image: str, env_dir: Path) -> tuple[set[str], set[str]]:
    """Run the Dockerfile's own uv sync commands in order.

    Returns the installed set after the first (package) sync and after the last.
    """
    installed = []
    for command in _uv_syncs(REPO / "libs" / image / "Dockerfile"):
        result = _run(command, env_dir)
        assert result.returncode == 0, (command, result.stderr)
        installed.append(_freeze(env_dir))
    return installed[0], installed[-1]


def _probe(env_dir: Path) -> dict:
    result = subprocess.run(
        [str(env_dir / "bin" / "python"), "-c", _PROBE],
        cwd=env_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("image", sorted(IMAGE_SYNCS))
def test_the_dockerfile_syncs_its_package_then_the_model_group(image):
    dockerfile = REPO / "libs" / image / "Dockerfile"

    assert _uv_syncs(dockerfile) == IMAGE_SYNCS[image]


@pytest.mark.parametrize("image", sorted(IMAGE_SYNCS))
def test_the_image_sync_installs_and_loads_the_pinned_model(image, tmp_path):
    env_dir = tmp_path / image

    after_package, after_model = _provision(image, env_dir)

    assert not any(line.startswith("en-core-web-sm") for line in after_package)
    assert after_model - after_package == {f"en-core-web-sm @ {SPACY_MODEL_URL}"}
    assert after_package - after_model == set()
    assert _probe(env_dir) == EXPECTED_PROBE


def test_concurrent_image_syncs_each_provision_the_model(tmp_path):
    barrier = threading.Barrier(len(IMAGE_SYNCS))
    outcomes: dict[str, tuple[float, float, set[str], set[str]]] = {}
    errors: list[BaseException] = []

    def build(image: str) -> None:
        try:
            barrier.wait(timeout=60)
            start = time.monotonic()
            after_package, after_model = _provision(image, tmp_path / image)
            outcomes[image] = (start, time.monotonic(), after_package, after_model)
        except BaseException as exc:  # recorded and re-raised on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=build, args=(i,)) for i in IMAGE_SYNCS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1800)

    assert errors == []
    assert set(outcomes) == set(IMAGE_SYNCS)
    (s1, e1, *_), (s2, e2, *_) = outcomes.values()
    assert max(s1, s2) < min(e1, e2)
    for image, (_, _, after_package, after_model) in outcomes.items():
        assert after_model - after_package == {f"en-core-web-sm @ {SPACY_MODEL_URL}"}, (
            image
        )
        assert _probe(tmp_path / image) == EXPECTED_PROBE, image


def test_a_model_wheel_that_does_not_match_the_lock_fails_the_sync(tmp_path):
    workspace = tmp_path / "workspace"
    listed = subprocess.run(
        ["git", "ls-files", "pyproject.toml", "uv.lock", "libs/*/pyproject.toml"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for relative in listed:
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, workspace / relative)
    lock = workspace / "uv.lock"
    assert lock.read_text().count(SPACY_MODEL_SHA256) == 1
    lock.write_text(lock.read_text().replace(SPACY_MODEL_SHA256, "0" * 64))
    env_dir = tmp_path / "env"

    model_sync = _uv_syncs(REPO / "libs" / "runtime" / "Dockerfile")[-1]

    result = _run(model_sync, env_dir, cwd=workspace)

    assert result.returncode == 1
    words = f" {' '.join(result.stderr.split())} "
    assert f" Hash mismatch for `en-core-web-sm @ {SPACY_MODEL_URL}` " in words
    assert f" Expected: sha256:{'0' * 64} " in words
    assert f" Computed: sha256:{SPACY_MODEL_SHA256} " in words
    site_packages = env_dir / "lib" / "python3.12" / "site-packages"
    assert site_packages.is_dir()
    assert list(site_packages.glob("en_core_web_sm*")) == []


@pytest.mark.parametrize(
    "selection",
    [
        "uv sync --frozen --extra cpu",
        "uv sync --all-packages --all-extras --no-extra cuda --no-extra rocm --frozen",
    ],
)
def test_workspace_syncs_keep_the_model_through_the_default_group(selection, tmp_path):
    result = _run(f"{selection} --dry-run", tmp_path / "env")

    assert result.returncode == 0, result.stderr
    assert re.search(
        rf"^\s*\+ en-core-web-sm==3\.8\.0 \(from {re.escape(SPACY_MODEL_URL)}\)$",
        result.stderr,
        re.MULTILINE,
    ), result.stderr

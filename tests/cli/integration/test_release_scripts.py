"""``scripts/build_packages.sh`` run in disposable Git checkouts of the real
package manifests, building through the real Hatch-VCS backend."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKOUT_PATHS = (
    ".gitignore",
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "Readme.md",
    "libs",
    "scripts",
)
_MANIFEST = "BUILD_MANIFEST.json"
_BUILD_TIMEOUT = 900

_PUBLICATION_ROOTS = (
    "cogniverse-core",
    "cogniverse-agents",
    "cogniverse-vespa",
    "cogniverse-runtime",
    "cogniverse-dashboard",
)
_EXPECTED_RELEASE = {
    "cogniverse-sdk",
    "cogniverse-foundation",
    "cogniverse-core",
    "cogniverse-evaluation",
    "cogniverse-synthetic",
    "cogniverse-vespa",
    "cogniverse-agents",
    "cogniverse-telemetry-phoenix",
    "cogniverse-runtime",
    "cogniverse-dashboard",
}

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Release Test",
    "GIT_AUTHOR_EMAIL": "release-test@example.invalid",
    "GIT_COMMITTER_NAME": "Release Test",
    "GIT_COMMITTER_EMAIL": "release-test@example.invalid",
}

_METADATA_STRIPPING_BACKEND = """\
import os
import zipfile

from hatchling.build import *  # noqa: F403
from hatchling.build import build_wheel as _build_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = _build_wheel(wheel_directory, config_settings, metadata_directory)
    path = os.path.join(wheel_directory, name)
    with zipfile.ZipFile(path) as source:
        kept = [
            (info, source.read(info))
            for info in source.infolist()
            if not info.filename.endswith(".dist-info/METADATA")
        ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for info, data in kept:
            target.writestr(info, data)
    return name
"""

_METADATA_VERSION_REWRITING_BACKEND = """\
import os
import zipfile

from hatchling.build import *  # noqa: F403
from hatchling.build import build_wheel as _build_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = _build_wheel(wheel_directory, config_settings, metadata_directory)
    path = os.path.join(wheel_directory, name)
    version = name.split("-")[1]
    with zipfile.ZipFile(path) as source:
        entries = [(info, source.read(info)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for info, data in entries:
            if info.filename.endswith(".dist-info/METADATA"):
                data = data.replace(
                    f"Version: {version}\\n".encode(), b"Version: 9.9.9\\n", 1
                )
            target.writestr(info, data)
    return name
"""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **_GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_checkout(root: Path, tag: str | None) -> Path:
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *_CHECKOUT_PATHS,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    for relative in filter(None, listed.split("\0")):
        source = _REPO_ROOT / relative
        if not source.is_file():
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "release inputs")
    if tag is not None:
        _git(root, "tag", "-a", tag, "-m", tag)
    return root


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)


def _build_command(repo: Path, *args: str) -> tuple[list[str], dict[str, str]]:
    env = {**os.environ, **_GIT_ENV, "UV_PROJECT_ENVIRONMENT": sys.prefix}
    return [str(repo / "scripts" / "build_packages.sh"), *args], env


def _run_build(repo: Path, *args: str) -> subprocess.CompletedProcess:
    command, env = _build_command(repo, *args)
    return subprocess.run(
        command,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=_BUILD_TIMEOUT,
    )


def _output(result: subprocess.CompletedProcess) -> str:
    return (
        f"exit={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(directory: Path) -> dict[str, str]:
    return {path.name: _sha256(path) for path in directory.iterdir()}


def _wheel_metadata(path: Path):
    with zipfile.ZipFile(path) as archive:
        [member] = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        return BytesParser().parsebytes(archive.read(member))


def _sdist_metadata(path: Path):
    with tarfile.open(path) as archive:
        [member] = [
            name
            for name in archive.getnames()
            if name.count("/") == 1 and name.endswith("/PKG-INFO")
        ]
        return BytesParser().parsebytes(archive.extractfile(member).read())


def _artifact_names(version: str) -> set[str]:
    names = set()
    for package in _EXPECTED_RELEASE:
        stem = f"{package.replace('-', '_')}-{version}"
        names |= {f"{stem}-py3-none-any.whl", f"{stem}.tar.gz"}
    return names


def _internal_requirements(metadata, workspace: set[str]) -> list[str]:
    names = {
        canonicalize_name(Requirement(line).name)
        for line in metadata.get_all("Requires-Dist") or []
    }
    return sorted(names & workspace)


def _workspace_names(repo: Path) -> set[str]:
    return {
        canonicalize_name(tomllib.loads(path.read_text())["project"]["name"])
        for path in (repo / "libs").glob("*/pyproject.toml")
    }


def _assert_release(repo: Path, expected_version: str) -> dict:
    dist = repo / "dist"
    manifest = json.loads((dist / _MANIFEST).read_text())
    workspace = _workspace_names(repo)

    assert manifest["version"] == expected_version
    names = [package["name"] for package in manifest["packages"]]
    assert len(names) == len(set(names))
    assert set(names) == _EXPECTED_RELEASE

    built_before: set[str] = set()
    produced_names = set()
    for package in manifest["packages"]:
        wheel = dist / package["wheel"]["filename"]
        sdist = dist / package["sdist"]["filename"]
        wheel_metadata = _wheel_metadata(wheel)
        sdist_metadata = _sdist_metadata(sdist)
        wheel_version = wheel_metadata["Version"]
        sdist_version = sdist_metadata["Version"]
        manifest_version = package["version"]

        assert Version(wheel_version) == Version(sdist_version)
        assert manifest_version == wheel_version
        assert canonicalize_name(wheel_metadata["Name"]) == package["name"]
        assert canonicalize_name(sdist_metadata["Name"]) == package["name"]
        assert package["wheel"]["sha256"] == _sha256(wheel)
        assert package["sdist"]["sha256"] == _sha256(sdist)

        requires = _internal_requirements(wheel_metadata, workspace)
        assert package["requires"] == requires
        assert set(requires) <= built_before, (package["name"], requires)
        built_before.add(package["name"])
        produced_names |= {wheel.name, sdist.name}

    expected_release_artifact_names = _artifact_names(expected_version)
    assert set(produced_names) == expected_release_artifact_names
    return manifest


def _dev_version(repo: Path, public: str) -> str:
    """The manifest version, once shown to be ``public`` plus this HEAD's hash."""
    version = Version(json.loads((repo / "dist" / _MANIFEST).read_text())["version"])
    head = _git(repo, "rev-parse", "HEAD")
    assert version.public == public
    local = version.local or ""
    assert local.startswith("g") and len(local) >= 8
    assert head.startswith(local[1:])
    return str(version)


def _break_backend_child(repo: Path) -> None:
    pyproject = repo / "libs" / "vespa" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text()
        + "\n[tool.hatch.build.targets.sdist.force-include]\n"
        + '"missing-build-input" = "missing-build-input"\n'
    )


def _use_vespa_backend(repo: Path, source: str) -> None:
    package = repo / "libs" / "vespa"
    pyproject = package / "pyproject.toml"
    text = pyproject.read_text()
    assert text.count('build-backend = "hatchling.build"\n') == 1
    pyproject.write_text(
        text.replace(
            'build-backend = "hatchling.build"\n',
            'build-backend = "release_probe_backend"\nbackend-path = ["."]\n',
        )
    )
    (package / "release_probe_backend.py").write_text(source)


def _strip_wheel_metadata(repo: Path) -> None:
    _use_vespa_backend(repo, _METADATA_STRIPPING_BACKEND)


def _rewrite_wheel_metadata_version(repo: Path) -> None:
    _use_vespa_backend(repo, _METADATA_VERSION_REWRITING_BACKEND)


def _add_undeclared_internal_dependency(repo: Path) -> None:
    pyproject = repo / "libs" / "runtime" / "pyproject.toml"
    text = pyproject.read_text()
    assert text.count('    "cogniverse-sdk",\n') == 1
    pyproject.write_text(
        text.replace(
            '    "cogniverse-sdk",\n',
            '    "cogniverse-sdk",\n    "cogniverse-messaging",\n',
        )
    )


def test_release_set_is_the_closure_of_the_publication_roots():
    requirements = {}
    for path in (_REPO_ROOT / "libs").glob("*/pyproject.toml"):
        project = tomllib.loads(path.read_text())["project"]
        requirements[canonicalize_name(project["name"])] = {
            canonicalize_name(Requirement(line).name)
            for line in project["dependencies"]
            + [
                line
                for extra in project.get("optional-dependencies", {}).values()
                for line in extra
            ]
        }
    closure = set()
    pending = list(_PUBLICATION_ROOTS)
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        pending.extend(requirements[name] & requirements.keys())

    assert closure == _EXPECTED_RELEASE


def test_release_tag_builds_declared_set_at_tag_version(tmp_path):
    repo = _make_checkout(tmp_path / "tagged", tag="v0.2.0")

    result = _run_build(repo)

    assert result.returncode == 0, _output(result)
    _assert_release(repo, "0.2.0")
    assert set(os.listdir(repo / "dist")) == _artifact_names("0.2.0") | {_MANIFEST}


def test_untagged_commit_builds_pep440_dev_version(tmp_path):
    repo = _make_checkout(tmp_path / "dev", tag="v0.2.0")
    _commit_all(repo, "unreleased change")

    result = _run_build(repo)

    assert result.returncode == 0, _output(result)
    expected = _dev_version(repo, "0.2.1.dev1")
    _assert_release(repo, expected)
    assert set(os.listdir(repo / "dist")) == _artifact_names(expected) | {_MANIFEST}


def test_parallel_builds_in_separate_roots_keep_their_own_artifacts(tmp_path):
    tagged = _make_checkout(tmp_path / "tagged", tag="v0.3.0")
    dev = _make_checkout(tmp_path / "dev", tag="v0.3.0")
    _commit_all(dev, "first unreleased change")
    _commit_all(dev, "second unreleased change")

    barrier = threading.Barrier(2)
    outcomes: dict[str, tuple[float, float, subprocess.CompletedProcess]] = {}

    def build(label: str, repo: Path) -> None:
        command, env = _build_command(repo)
        barrier.wait()
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=_BUILD_TIMEOUT,
        )
        outcomes[label] = (started, time.monotonic(), result)

    threads = [
        threading.Thread(target=build, args=("tagged", tagged)),
        threading.Thread(target=build, args=("dev", dev)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tagged_start, tagged_end, tagged_result = outcomes["tagged"]
    dev_start, dev_end, dev_result = outcomes["dev"]
    assert tagged_start < dev_end and dev_start < tagged_end
    assert tagged_result.returncode == 0, _output(tagged_result)
    assert dev_result.returncode == 0, _output(dev_result)
    dev_version = _dev_version(dev, "0.3.1.dev2")
    _assert_release(tagged, "0.3.0")
    _assert_release(dev, dev_version)
    assert set(os.listdir(tagged / "dist")) == _artifact_names("0.3.0") | {_MANIFEST}
    assert set(os.listdir(dev / "dist")) == _artifact_names(dev_version) | {_MANIFEST}


@pytest.mark.parametrize(
    ("break_release", "expected_messages"),
    [
        pytest.param(
            _break_backend_child,
            [
                "Forced include not found",
                "Failed to build package: vespa",
            ],
            id="failed-backend-child",
        ),
        pytest.param(
            _strip_wheel_metadata,
            [
                "cogniverse_vespa-0.2.0-py3-none-any.whl: missing "
                "cogniverse_vespa-0.2.0.dist-info/METADATA",
            ],
            id="missing-wheel-metadata",
        ),
        pytest.param(
            _rewrite_wheel_metadata_version,
            [
                "cogniverse_vespa-0.2.0-py3-none-any.whl: METADATA version 9.9.9 "
                "does not match filename version 0.2.0",
            ],
            id="metadata-disagrees-with-filename",
        ),
        pytest.param(
            _add_undeclared_internal_dependency,
            [
                "cogniverse-runtime requires cogniverse-messaging, which the "
                "release set does not build before it",
            ],
            id="undeclared-internal-dependency",
        ),
    ],
)
def test_failed_build_exits_nonzero_and_leaves_previous_artifacts_untouched(
    tmp_path, break_release, expected_messages
):
    repo = _make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, _output(previous)
    dist = repo / "dist"
    (dist / _MANIFEST).unlink()
    old_artifact_hashes_before = _hashes(dist)
    assert set(old_artifact_hashes_before) == _artifact_names("0.1.0")

    break_release(repo)
    _commit_all(repo, "break the release")
    _git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    failed_build = _run_build(repo)

    assert failed_build.returncode != 0, _output(failed_build)
    output = failed_build.stdout + failed_build.stderr
    for message in expected_messages:
        assert message in output, _output(failed_build)
    old_artifact_hashes_after = _hashes(dist)
    assert old_artifact_hashes_after == old_artifact_hashes_before


def test_failed_build_removes_the_previous_manifest(tmp_path):
    repo = _make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, _output(previous)
    _assert_release(repo, "0.1.0")

    _break_backend_child(repo)
    _commit_all(repo, "break the release")
    failed_build = _run_build(repo)

    assert failed_build.returncode != 0, _output(failed_build)
    assert set(os.listdir(repo / "dist")) == _artifact_names("0.1.0")


def test_rebuild_without_clean_leaves_unrelated_artifacts_untouched(tmp_path):
    repo = _make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, _output(previous)
    dist = repo / "dist"
    (dist / "operator-notes.txt").write_text("kept across builds\n")
    old_artifact_hashes_before = {
        name: digest for name, digest in _hashes(dist).items() if name != _MANIFEST
    }

    _commit_all(repo, "next release")
    _git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    result = _run_build(repo)

    assert result.returncode == 0, _output(result)
    _assert_release(repo, "0.2.0")
    old_artifact_hashes_after = {
        name: digest
        for name, digest in _hashes(dist).items()
        if name in old_artifact_hashes_before
    }
    assert old_artifact_hashes_after == old_artifact_hashes_before
    assert set(os.listdir(dist)) == (
        set(old_artifact_hashes_before) | _artifact_names("0.2.0") | {_MANIFEST}
    )


def test_existing_artifact_with_different_bytes_is_never_replaced(tmp_path):
    repo = _make_checkout(tmp_path / "release", tag="v0.2.0")
    first = _run_build(repo)
    assert first.returncode == 0, _output(first)
    dist = repo / "dist"
    first_manifest = (dist / _MANIFEST).read_text()

    identical = _run_build(repo)
    assert identical.returncode == 0, _output(identical)
    assert (dist / _MANIFEST).read_text() == first_manifest

    wheel = dist / "cogniverse_core-0.2.0-py3-none-any.whl"
    wheel.write_bytes(wheel.read_bytes() + b"previously published bytes")
    old_artifact_hashes_before = {
        name: digest for name, digest in _hashes(dist).items() if name != _MANIFEST
    }
    failed_build = _run_build(repo)

    assert failed_build.returncode != 0, _output(failed_build)
    assert (
        f"{dist / wheel.name} differs from the artifact built by this invocation"
        in failed_build.stdout + failed_build.stderr
    ), _output(failed_build)
    old_artifact_hashes_after = _hashes(dist)
    assert old_artifact_hashes_after == old_artifact_hashes_before


def test_clean_in_disposable_copy_leaves_only_this_release(tmp_path):
    repo = _make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, _output(previous)
    (repo / "dist" / "operator-notes.txt").write_text("removed by --clean\n")

    _commit_all(repo, "next release")
    _git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    result = _run_build(repo, "--verbose", "--clean")

    assert result.returncode == 0, _output(result)
    _assert_release(repo, "0.2.0")
    assert set(os.listdir(repo / "dist")) == _artifact_names("0.2.0") | {_MANIFEST}


def test_unknown_option_exits_nonzero_without_building(tmp_path):
    repo = _make_checkout(tmp_path / "release", tag="v0.2.0")

    result = _run_build(repo, "--no-such-option")

    assert result.returncode == 2, _output(result)
    assert "Unknown option: --no-such-option" in result.stderr, _output(result)
    assert not (repo / "dist").exists()

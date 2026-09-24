"""``scripts/build_packages.sh`` run in disposable Git checkouts of the real
package manifests, building through the real Hatch-VCS backend, and
``scripts/publish_packages.sh`` publishing those builds with the real Twine to an
owned pypiserver reached at the PyPI and TestPyPI URLs."""

from __future__ import annotations

import datetime
import hashlib
import http.client
import http.server
import json
import os
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path

import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename
from packaging.version import Version

from tests.cli.integration.release_build import (
    BUILD_TIMEOUT,
    CHECKOUT_PATHS,
    EXPECTED_RELEASE,
    GIT_ENV,
    REPO_ROOT,
    build_command,
    describe_run,
    git,
    make_checkout,
)

pytestmark = pytest.mark.integration

_MANIFEST = "BUILD_MANIFEST.json"

_PUBLICATION_ROOTS = (
    "cogniverse-core",
    "cogniverse-agents",
    "cogniverse-vespa",
    "cogniverse-runtime",
    "cogniverse-dashboard",
)


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

_PKG_INFO_HIDING_BACKEND = """\
import io
import os
import tarfile

from hatchling.build import *  # noqa: F403
from hatchling.build import build_sdist as _build_sdist
from hatchling.build import build_wheel as _build_wheel

PKG_INFO_AS_DIRECTORY = {as_directory}


def build_sdist(sdist_directory, config_settings=None):
    name = _build_sdist(sdist_directory, config_settings)
    path = os.path.join(sdist_directory, name)
    pkg_info = name.removesuffix(".tar.gz") + "/PKG-INFO"
    with tarfile.open(path) as source:
        members = [
            (member, source.extractfile(member).read() if member.isfile() else None)
            for member in source.getmembers()
        ]
    with tarfile.open(path, "w:gz") as target:
        for member, data in members:
            if member.name == pkg_info:
                member.name = pkg_info + ".orig"
                if PKG_INFO_AS_DIRECTORY:
                    directory = tarfile.TarInfo(pkg_info)
                    directory.type = tarfile.DIRTYPE
                    directory.mode = 0o755
                    target.addfile(directory)
            target.addfile(member, None if data is None else io.BytesIO(data))
    return name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    if os.path.isfile("PKG-INFO.orig"):
        with open("PKG-INFO.orig") as pkg_info:
            for line in pkg_info:
                if line.startswith("Version: "):
                    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = line[9:].strip()
                    break
    return _build_wheel(wheel_directory, config_settings, metadata_directory)
"""


def _commit_all(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)


def _build_env(**overrides: str) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("SETUPTOOLS_SCM_PRETEND_VERSION")
    }
    return {**env, **GIT_ENV, "UV_PROJECT_ENVIRONMENT": sys.prefix, **overrides}


def _build_command(
    repo: Path, *args: str, **env_overrides: str
) -> tuple[list[str], dict[str, str]]:
    command, _ = build_command(repo, *args)
    return command, _build_env(**env_overrides)


def _run_build(
    repo: Path, *args: str, **env_overrides: str
) -> subprocess.CompletedProcess:
    command, env = _build_command(repo, *args, **env_overrides)
    return subprocess.run(
        command,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
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
    for package in EXPECTED_RELEASE:
        stem = f"{package.replace('-', '_')}-{version}"
        names |= {f"{stem}-py3-none-any.whl", f"{stem}.tar.gz"}
    return names


def _internal_requirements(metadata, workspace: set[str]) -> list[str]:
    names = {
        canonicalize_name(Requirement(line).name)
        for line in metadata.get_all("Requires-Dist") or []
    }
    return sorted(names & workspace)


def _internal_requirement_lines(metadata, workspace: set[str]) -> list[Requirement]:
    return [
        requirement
        for requirement in map(Requirement, metadata.get_all("Requires-Dist") or [])
        if canonicalize_name(requirement.name) in workspace
    ]


def _staging_directories(result: subprocess.CompletedProcess) -> list[str]:
    return re.findall(r"^Staging directory: (.+)$", result.stdout, re.MULTILINE)


def _temporary_files(directory: Path) -> list[str]:
    """Entries left in ``directory`` other than uv's persistent lock files."""
    return sorted(
        name
        for name in os.listdir(directory)
        if not re.fullmatch(r"uv-[0-9a-f]{16}\.lock", name)
    )


def _backend_version(repo: Path, out_dir: Path) -> str:
    """The version hatch-vcs computes for this checkout, built in place."""
    subprocess.run(
        [
            "uv",
            "build",
            "--no-sources",
            "--sdist",
            "libs/sdk",
            "--out-dir",
            str(out_dir),
        ],
        cwd=repo,
        env=_build_env(),
        capture_output=True,
        check=True,
        timeout=BUILD_TIMEOUT,
    )
    [sdist] = out_dir.glob("*.tar.gz")
    return str(parse_sdist_filename(sdist.name)[1])


def _assert_workspace_unchanged(repo: Path) -> None:
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "libs",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=repo,
        env={**os.environ, **GIT_ENV},
        check=True,
    )
    lock = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        cwd=repo,
        env=_build_env(),
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
    )
    assert lock.returncode == 0, describe_run(lock)


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
    assert set(names) == EXPECTED_RELEASE

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
        assert wheel_metadata.get_all("Requires-Dist") == sdist_metadata.get_all(
            "Requires-Dist"
        )
        readme = (
            repo / "libs" / package["name"].removeprefix("cogniverse-") / "README.md"
        ).read_text()
        for metadata in (wheel_metadata, sdist_metadata):
            assert metadata["Description-Content-Type"] == "text/markdown"
            assert metadata.get_payload(decode=True).decode() == readme
        internal = _internal_requirement_lines(wheel_metadata, workspace)
        assert {canonicalize_name(line.name) for line in internal} == set(requires)
        assert [str(line.specifier) for line in internal] == [
            f"=={manifest_version}"
        ] * len(internal), (package["name"], [str(line) for line in internal])
        built_before.add(package["name"])
        produced_names |= {wheel.name, sdist.name}

    expected_release_artifact_names = _artifact_names(expected_version)
    assert set(produced_names) == expected_release_artifact_names
    return manifest


def _dev_version(repo: Path, public: str) -> str:
    """The manifest version, once shown to be ``public`` plus this HEAD's hash."""
    version = Version(json.loads((repo / "dist" / _MANIFEST).read_text())["version"])
    head = git(repo, "rev-parse", "HEAD")
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


def _hide_sdist_pkg_info(repo: Path) -> None:
    _use_vespa_backend(repo, _PKG_INFO_HIDING_BACKEND.format(as_directory=False))


def _replace_sdist_pkg_info_with_directory(repo: Path) -> None:
    _use_vespa_backend(repo, _PKG_INFO_HIDING_BACKEND.format(as_directory=True))


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


def _constrain_internal_requirement(repo: Path) -> None:
    pyproject = repo / "libs" / "runtime" / "pyproject.toml"
    text = pyproject.read_text()
    assert text.count('    "cogniverse-sdk",\n') == 1
    pyproject.write_text(
        text.replace('    "cogniverse-sdk",\n', '    "cogniverse-sdk>=0.1",\n')
    )


def test_release_set_is_the_closure_of_the_publication_roots():
    requirements = {}
    for path in (REPO_ROOT / "libs").glob("*/pyproject.toml"):
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

    assert closure == EXPECTED_RELEASE


def test_release_tag_builds_declared_set_at_tag_version(tmp_path):
    repo = make_checkout(tmp_path / "tagged", tag="v0.2.0")
    temp = tmp_path / "temp"
    temp.mkdir()

    result = _run_build(repo, TMPDIR=str(temp))

    assert result.returncode == 0, describe_run(result)
    assert _backend_version(repo, tmp_path / "backend") == "0.2.0"
    _assert_release(repo, "0.2.0")
    assert set(os.listdir(repo / "dist")) == _artifact_names("0.2.0") | {_MANIFEST}
    [staging] = _staging_directories(result)
    assert Path(staging).parent == temp
    assert not Path(staging).exists()
    assert _temporary_files(temp) == []
    _assert_workspace_unchanged(repo)


def test_untagged_commit_builds_pep440_dev_version(tmp_path):
    repo = make_checkout(tmp_path / "dev", tag="v0.2.0")
    _commit_all(repo, "unreleased change")

    result = _run_build(repo)

    assert result.returncode == 0, describe_run(result)
    expected = _dev_version(repo, "0.2.1.dev1")
    assert _backend_version(repo, tmp_path / "backend") == expected
    _assert_release(repo, expected)
    assert set(os.listdir(repo / "dist")) == _artifact_names(expected) | {_MANIFEST}
    _assert_workspace_unchanged(repo)


def test_parallel_builds_in_separate_roots_keep_their_own_artifacts(tmp_path):
    tagged = make_checkout(tmp_path / "tagged", tag="v0.3.0")
    dev = make_checkout(tmp_path / "dev", tag="v0.3.0")
    _commit_all(dev, "first unreleased change")
    _commit_all(dev, "second unreleased change")

    temp = tmp_path / "shared-temp"
    temp.mkdir()
    barrier = threading.Barrier(2)
    outcomes: dict[str, tuple[float, float, subprocess.CompletedProcess]] = {}

    def build(label: str, repo: Path) -> None:
        command, env = _build_command(repo, TMPDIR=str(temp))
        barrier.wait()
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
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
    assert tagged_result.returncode == 0, describe_run(tagged_result)
    assert dev_result.returncode == 0, describe_run(dev_result)
    dev_version = _dev_version(dev, "0.3.1.dev2")
    _assert_release(tagged, "0.3.0")
    _assert_release(dev, dev_version)
    assert set(os.listdir(tagged / "dist")) == _artifact_names("0.3.0") | {_MANIFEST}
    assert set(os.listdir(dev / "dist")) == _artifact_names(dev_version) | {_MANIFEST}
    [tagged_staging] = _staging_directories(tagged_result)
    [dev_staging] = _staging_directories(dev_result)
    assert tagged_staging != dev_staging
    assert Path(tagged_staging).parent == Path(dev_staging).parent == temp
    assert not Path(tagged_staging).exists() and not Path(dev_staging).exists()
    assert _temporary_files(temp) == []


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
            _hide_sdist_pkg_info,
            [
                "cogniverse_vespa-0.2.0.tar.gz: missing cogniverse_vespa-0.2.0/PKG-INFO",
            ],
            id="missing-sdist-pkg-info",
        ),
        pytest.param(
            _replace_sdist_pkg_info_with_directory,
            [
                "cogniverse_vespa-0.2.0.tar.gz: cogniverse_vespa-0.2.0/PKG-INFO "
                "is not a regular file",
            ],
            id="sdist-pkg-info-not-a-file",
        ),
        pytest.param(
            _add_undeclared_internal_dependency,
            [
                "cogniverse-runtime requires cogniverse-messaging, which the "
                "release set does not build before it",
            ],
            id="undeclared-internal-dependency",
        ),
        pytest.param(
            _constrain_internal_requirement,
            [
                "error: internal requirement 'cogniverse-sdk>=0.1' already has a "
                "version specifier",
                "Failed to stage the source of runtime",
            ],
            id="constrained-internal-requirement",
        ),
    ],
)
def test_failed_build_exits_nonzero_and_leaves_previous_artifacts_untouched(
    tmp_path, break_release, expected_messages
):
    repo = make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, describe_run(previous)
    dist = repo / "dist"
    (dist / _MANIFEST).unlink()
    old_artifact_hashes_before = _hashes(dist)
    assert set(old_artifact_hashes_before) == _artifact_names("0.1.0")

    break_release(repo)
    _commit_all(repo, "break the release")
    git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    temp = tmp_path / "temp"
    temp.mkdir()
    failed_build = _run_build(repo, TMPDIR=str(temp))

    assert failed_build.returncode != 0, describe_run(failed_build)
    output = failed_build.stdout + failed_build.stderr
    for message in expected_messages:
        assert message in output, describe_run(failed_build)
    old_artifact_hashes_after = _hashes(dist)
    assert old_artifact_hashes_after == old_artifact_hashes_before
    [staging] = _staging_directories(failed_build)
    assert Path(staging).parent == temp
    assert not Path(staging).exists()
    assert _temporary_files(temp) == []


def test_failed_copy_into_dist_exits_nonzero_without_manifest(tmp_path):
    repo = make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, describe_run(previous)
    dist = repo / "dist"
    (dist / _MANIFEST).unlink()
    old_artifact_hashes_before = _hashes(dist)
    assert set(old_artifact_hashes_before) == _artifact_names("0.1.0")

    _commit_all(repo, "next release")
    git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    dist.chmod(0o555)
    try:
        failed_build = _run_build(repo)
    finally:
        dist.chmod(0o755)

    assert failed_build.returncode != 0, describe_run(failed_build)
    assert (
        f"Failed to copy cogniverse_sdk-0.2.0-py3-none-any.whl into {dist}"
        in failed_build.stderr
    ), describe_run(failed_build)
    old_artifact_hashes_after = _hashes(dist)
    assert old_artifact_hashes_after == old_artifact_hashes_before


_TRUNCATING_CP = """\
#!/bin/bash
size=$(stat -c %s "$1")
head -c $((size / 2)) "$1" > "$2"
echo "cp: error writing '$2': No space left on device" >&2
exit 1
"""


def test_interrupted_copy_leaves_no_artifact_under_its_real_name(tmp_path):
    repo = make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, describe_run(previous)
    dist = repo / "dist"
    (dist / _MANIFEST).unlink()
    old_artifact_hashes_before = _hashes(dist)

    _commit_all(repo, "next release")
    git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "cp").write_text(_TRUNCATING_CP)
    (shim / "cp").chmod(0o755)
    command, env = _build_command(repo)
    env["PATH"] = f"{shim}{os.pathsep}{env['PATH']}"
    failed_build = subprocess.run(
        command,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
    )

    assert failed_build.returncode != 0, describe_run(failed_build)
    assert (
        f"Failed to copy cogniverse_sdk-0.2.0-py3-none-any.whl into {dist}"
        in failed_build.stderr
    ), describe_run(failed_build)
    old_artifact_hashes_after = _hashes(dist)
    assert old_artifact_hashes_after == old_artifact_hashes_before


def test_parallel_test_runs_write_separate_logs(tmp_path):
    repos = [
        make_checkout(tmp_path / "first", tag="v0.2.0"),
        make_checkout(tmp_path / "second", tag="v0.2.0"),
    ]
    barrier = threading.Barrier(2)
    results: dict[int, subprocess.CompletedProcess] = {}

    def build(index: int) -> None:
        command, env = _build_command(repos[index], "--test")
        env["UV_NO_SYNC"] = "1"
        env["TMPDIR"] = str(tmp_path)
        barrier.wait()
        results[index] = subprocess.run(
            command,
            cwd=repos[index],
            env=env,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
        )

    threads = [threading.Thread(target=build, args=(index,)) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    log_dirs = []
    for index, repo in enumerate(repos):
        result = results[index]
        assert result.returncode == 0, describe_run(result)
        [line] = [line for line in result.stdout.splitlines() if "Test logs: " in line]
        log_dir = Path(line.split("Test logs: ", 1)[1])
        assert log_dir.parent == tmp_path
        log_dirs.append(log_dir)
        packages = sorted(name.removeprefix("cogniverse-") for name in EXPECTED_RELEASE)
        assert sorted(path.name for path in log_dir.iterdir()) == [
            f"{package}.log" for package in packages
        ]
        for package in packages:
            log = (log_dir / f"{package}.log").read_text()
            assert f"rootdir: {repo}\n" in log, log
            assert f"ERROR: file or directory not found: tests/{package}/\n" in log, log
    assert log_dirs[0] != log_dirs[1]


def test_checkout_inputs_are_tracked_files_only(tmp_path):
    untracked = REPO_ROOT / "libs" / "sdk" / "release-input-probe-untracked.txt"
    assert not untracked.exists()
    untracked.write_text("untracked developer file\n")
    try:
        repo = make_checkout(tmp_path / "release", tag=None)
    finally:
        untracked.unlink()

    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--", *CHECKOUT_PATHS],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    copied = set(git(repo, "ls-files").splitlines())
    assert copied == {
        path for path in tracked.split("\0") if path and (REPO_ROOT / path).is_file()
    }
    assert "libs/sdk/release-input-probe-untracked.txt" not in copied


def test_failed_build_removes_the_previous_manifest(tmp_path):
    repo = make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, describe_run(previous)
    _assert_release(repo, "0.1.0")

    _break_backend_child(repo)
    _commit_all(repo, "break the release")
    failed_build = _run_build(repo)

    assert failed_build.returncode != 0, describe_run(failed_build)
    assert set(os.listdir(repo / "dist")) == _artifact_names("0.1.0")


def test_rebuild_without_clean_leaves_unrelated_artifacts_untouched(tmp_path):
    repo = make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, describe_run(previous)
    dist = repo / "dist"
    (dist / "operator-notes.txt").write_text("kept across builds\n")
    old_artifact_hashes_before = {
        name: digest for name, digest in _hashes(dist).items() if name != _MANIFEST
    }

    _commit_all(repo, "next release")
    git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    result = _run_build(repo)

    assert result.returncode == 0, describe_run(result)
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
    repo = make_checkout(tmp_path / "release", tag="v0.2.0")
    first = _run_build(repo)
    assert first.returncode == 0, describe_run(first)
    dist = repo / "dist"
    first_manifest = (dist / _MANIFEST).read_text()

    identical = _run_build(repo)
    assert identical.returncode == 0, describe_run(identical)
    assert (dist / _MANIFEST).read_text() == first_manifest

    wheel = dist / "cogniverse_core-0.2.0-py3-none-any.whl"
    wheel.write_bytes(wheel.read_bytes() + b"previously published bytes")
    old_artifact_hashes_before = {
        name: digest for name, digest in _hashes(dist).items() if name != _MANIFEST
    }
    failed_build = _run_build(repo)

    assert failed_build.returncode != 0, describe_run(failed_build)
    assert (
        f"{dist / wheel.name} differs from the artifact built by this invocation"
        in failed_build.stdout + failed_build.stderr
    ), describe_run(failed_build)
    old_artifact_hashes_after = _hashes(dist)
    assert old_artifact_hashes_after == old_artifact_hashes_before


def test_clean_in_disposable_copy_leaves_only_this_release(tmp_path):
    repo = make_checkout(tmp_path / "release", tag="v0.1.0")
    previous = _run_build(repo)
    assert previous.returncode == 0, describe_run(previous)
    (repo / "dist" / "operator-notes.txt").write_text("removed by --clean\n")

    _commit_all(repo, "next release")
    git(repo, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    result = _run_build(repo, "--verbose", "--clean")

    assert result.returncode == 0, describe_run(result)
    _assert_release(repo, "0.2.0")
    assert set(os.listdir(repo / "dist")) == _artifact_names("0.2.0") | {_MANIFEST}


def test_unknown_option_exits_nonzero_without_building(tmp_path):
    repo = make_checkout(tmp_path / "release", tag="v0.2.0")

    result = _run_build(repo, "--no-such-option")

    assert result.returncode == 2, describe_run(result)
    assert "Unknown option: --no-such-option" in result.stderr, describe_run(result)
    assert not (repo / "dist").exists()


_TWINE = "twine==7.0.0"
_PUBLISH_REQUIREMENTS = REPO_ROOT / "scripts" / "publish-requirements.txt"
_PYPISERVER = "pypiserver[passlib]==2.4.2"
_PUBLISH_TIMEOUT = 900
_REGISTRY_HOSTS = ("test.pypi.org", "upload.pypi.org", "pypi.org", "pypi.python.org")
_TESTPYPI_INDEX = "https://test.pypi.org/simple/"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROXY_VARIABLES = {
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
}


class _Disconnected(Exception):
    pass


class _RegistryProxy(http.server.ThreadingHTTPServer):
    """HTTPS proxy terminating TLS for the (Test)PyPI hosts and relaying every
    request to the owned pypiserver, recording each and injecting faults per
    uploaded filename or, once per queued entry, per index path:
    ``server-error`` answers 503 without relaying; ``disconnect`` relays, then
    drops the connection partway through the registry's response."""

    daemon_threads = True

    def __init__(self, upstream_port: int, tls_context: ssl.SSLContext):
        super().__init__(("127.0.0.1", 0), _ConnectHandler)
        self.upstream_port = upstream_port
        self.tls_context = tls_context
        self.faults: dict[str, str] = {}
        self.index_faults: dict[str, list[str]] = {}
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        return self.server_address[1]

    def record(self, entry: dict) -> None:
        with self._lock:
            self.requests.append(entry)

    def clear(self) -> None:
        with self._lock:
            self.requests.clear()

    def uploads(self) -> list[tuple[str, str, int | None]]:
        return [
            (entry["host"], entry["filename"], entry["status"])
            for entry in self.requests
            if entry["method"] == "POST"
        ]

    def index_reads(self) -> list[tuple[str, str]]:
        return [
            (entry["host"], entry["path"])
            for entry in self.requests
            if entry["method"] == "GET" and entry["path"].startswith("/simple/")
        ]


class _ConnectHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 120

    def do_CONNECT(self):
        host = self.path.rsplit(":", 1)[0]
        self.send_response(200, "Connection established")
        self.end_headers()
        self.close_connection = True
        try:
            tunnel = self.server.tls_context.wrap_socket(
                self.connection, server_side=True
            )
        except (ssl.SSLError, OSError):
            return
        handler = type("_TunnelHandler", (_RegistryHandler,), {"host": host})
        try:
            handler(tunnel, self.client_address, self.server)
        except (_Disconnected, ssl.SSLError, OSError):
            pass
        finally:
            tunnel.close()

    def log_message(self, format, *args):
        pass


class _RegistryHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 120
    host = ""

    def do_GET(self):
        self._relay()

    def do_POST(self):
        self._relay()

    def log_message(self, format, *args):
        pass

    def _relay(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        match = re.search(rb'name="content"; filename="([^"]+)"', body)
        filename = match.group(1).decode() if match else None
        entry = {
            "method": self.command,
            "host": self.host,
            "path": self.path,
            "filename": filename,
            "status": None,
        }
        self.server.record(entry)
        fault = self.server.faults.get(filename) if filename else None
        if self.command == "GET" and self.server.index_faults.get(self.path):
            fault = self.server.index_faults[self.path].pop(0)

        if fault == "server-error":
            status, reason = 503, "Service Unavailable"
            content_type, payload = "text/plain", b"registry unavailable"
        else:
            upstream = http.client.HTTPConnection(
                "127.0.0.1", self.server.upstream_port, timeout=120
            )
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower()
                in {"authorization", "content-type", "user-agent", "accept"}
            }
            path = "/" if self.path == "/legacy/" else self.path
            upstream.request(self.command, path, body=body or None, headers=headers)
            response = upstream.getresponse()
            payload = response.read()
            status, reason = response.status, response.reason
            content_type = response.getheader("Content-Type", "text/plain")
            upstream.close()

        entry["status"] = status
        self.send_response(status, reason)
        self.send_header("Content-Type", content_type)
        if fault == "disconnect":
            self.send_header("Content-Length", str(len(payload) + 64))
            self.end_headers()
            self.wfile.write(payload)
            self.connection.shutdown(socket.SHUT_RDWR)
            raise _Disconnected(filename)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@dataclass
class _Registry:
    packages: Path
    token: str
    proxy: _RegistryProxy

    def files(self) -> dict[str, str]:
        return _hashes(self.packages)


@dataclass
class _PublishTools:
    env: dict[str, str]
    authority: Path
    tls_context: ssl.SSLContext


def _certificate_authority(directory: Path) -> tuple[Path, ssl.SSLContext]:
    now = datetime.datetime.now(datetime.UTC)
    validity = (now - datetime.timedelta(minutes=5), now + datetime.timedelta(days=1))
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Release Test CA")])
    ca_certificate = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(validity[0])
        .not_valid_after(validity[1])
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_certificate = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _REGISTRY_HOSTS[0])])
        )
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(validity[0])
        .not_valid_after(validity[1])
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(h) for h in _REGISTRY_HOSTS]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    authority = directory / "authority.pem"
    authority.write_bytes(ca_certificate.public_bytes(serialization.Encoding.PEM))
    chain = directory / "server.pem"
    chain.write_bytes(server_certificate.public_bytes(serialization.Encoding.PEM))
    key = directory / "server.key"
    key.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(chain, key)
    return authority, context


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_http(url: str, process: subprocess.Popen, log: Path) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        assert process.poll() is None, log.read_text()
        try:
            with opener.open(url, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"{url} not ready:\n{log.read_text()}")


@pytest.fixture(scope="module")
def release_dists(tmp_path_factory) -> dict[str, Path]:
    tagged = make_checkout(tmp_path_factory.mktemp("tagged"), tag="v0.2.0")
    rebuilt = make_checkout(tmp_path_factory.mktemp("rebuilt"), tag=None)
    readme = rebuilt / "libs" / "sdk" / "README.md"
    readme.write_text(readme.read_text() + "\nRebuilt from a different commit.\n")
    _commit_all(rebuilt, "change the sdk description")
    git(rebuilt, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
    dev = make_checkout(tmp_path_factory.mktemp("dev"), tag="v0.2.0")
    _commit_all(dev, "unreleased change")
    for repo in (tagged, rebuilt, dev):
        result = _run_build(repo)
        assert result.returncode == 0, describe_run(result)
    return {"tagged": tagged / "dist", "rebuilt": rebuilt / "dist", "dev": dev / "dist"}


@pytest.fixture(scope="module")
def publish_tools(tmp_path_factory) -> _PublishTools:
    home = tmp_path_factory.mktemp("publish-home")
    cache = subprocess.run(
        ["uv", "cache", "dir"], capture_output=True, text=True, check=True
    ).stdout.strip()
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _PROXY_VARIABLES and not name.startswith("TWINE_")
    }
    env.update(
        HOME=str(home),
        UV_CACHE_DIR=cache,
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    tool = home / "publish-tool"
    for argv in (
        ["uv", "venv", "--no-config", "--python", "3.12", str(tool)],
        [
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            str(tool / "bin" / "python"),
            "--require-hashes",
            "-r",
            str(_PUBLISH_REQUIREMENTS),
        ],
        [str(tool / "bin" / "python"), "-c", "import twine"],
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            _PYPISERVER,
            "python",
            "-c",
            "import passlib, pypiserver",
        ],
    ):
        subprocess.run(
            argv,
            cwd=home,
            env=env,
            capture_output=True,
            check=True,
            timeout=_PUBLISH_TIMEOUT,
        )
    shutil.rmtree(tool)
    env["UV_OFFLINE"] = "1"
    authority, context = _certificate_authority(tmp_path_factory.mktemp("tls"))
    return _PublishTools(env=env, authority=authority, tls_context=context)


@pytest.fixture
def registry(tmp_path, publish_tools):
    root = tmp_path / "registry"
    packages = root / "packages"
    packages.mkdir(parents=True)
    token = secrets.token_urlsafe(24)
    htpasswd = root / "htpasswd"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            _PYPISERVER,
            "python",
            "-c",
            "import sys; from passlib.apache import HtpasswdFile; "
            "f = HtpasswdFile(sys.argv[1], new=True); "
            "f.set_password('__token__', sys.argv[2]); f.save()",
            str(htpasswd),
            token,
        ],
        cwd=root,
        env=publish_tools.env,
        capture_output=True,
        check=True,
    )
    port = _free_port()
    log = root / "pypiserver.log"
    with log.open("w") as log_file:
        server = subprocess.Popen(
            [
                "uv",
                "run",
                "--no-project",
                "--with",
                _PYPISERVER,
                "pypi-server",
                "run",
                "-p",
                str(port),
                "-i",
                "127.0.0.1",
                "-P",
                str(htpasswd),
                "-a",
                "update",
                "--disable-fallback",
                "--backend",
                "simple-dir",
                "--hash-algo",
                "sha256",
                str(packages),
            ],
            cwd=root,
            env=publish_tools.env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    proxy = None
    try:
        _wait_for_http(f"http://127.0.0.1:{port}/simple/", server, log)
        proxy = _RegistryProxy(port, publish_tools.tls_context)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        yield _Registry(packages=packages, token=token, proxy=proxy)
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=60)


def _proxy_env(tools: _PublishTools, proxy_port: int) -> dict[str, str]:
    proxy = f"http://127.0.0.1:{proxy_port}"
    return {
        **tools.env,
        "HTTPS_PROXY": proxy,
        "https_proxy": proxy,
        "HTTP_PROXY": proxy,
        "http_proxy": proxy,
        "REQUESTS_CA_BUNDLE": str(tools.authority),
        "SSL_CERT_FILE": str(tools.authority),
        "CURL_CA_BUNDLE": str(tools.authority),
        "VERIFY_TIMEOUT": "0",
    }


def _publish_env(
    tools: _PublishTools, proxy_port: int, token: str, **overrides: str
) -> dict[str, str]:
    return {
        **_proxy_env(tools, proxy_port),
        "PYPI_TOKEN": token,
        "TEST_PYPI_TOKEN": token,
        **overrides,
    }


def _registry_env(
    tools: _PublishTools, registry: _Registry, **overrides: str
) -> dict[str, str]:
    return _publish_env(tools, registry.proxy.port, registry.token, **overrides)


def _publication_root(root: Path, dist: Path) -> Path:
    (root / "scripts").mkdir(parents=True)
    for script in (
        "publish_packages.sh",
        "release_manifest.py",
        _PUBLISH_REQUIREMENTS.name,
    ):
        shutil.copy2(REPO_ROOT / "scripts" / script, root / "scripts" / script)
    shutil.copytree(dist, root / "dist")
    return root


def _run_publish(
    root: Path, env: dict[str, str], *args: str, answer: str | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(root / "scripts" / "publish_packages.sh"), *args],
        cwd=root,
        env=env,
        input=answer,
        stdin=subprocess.DEVNULL if answer is None else None,
        capture_output=True,
        text=True,
        timeout=_PUBLISH_TIMEOUT,
    )


def _manifest_files(dist: Path) -> dict[str, str]:
    manifest = json.loads((dist / _MANIFEST).read_text())
    return {
        entry["filename"]: entry["sha256"]
        for package in manifest["packages"]
        for entry in (package["wheel"], package["sdist"])
    }


def _manifest_names(dist: Path) -> list[str]:
    manifest = json.loads((dist / _MANIFEST).read_text())
    return [package["name"] for package in manifest["packages"]]


def test_publication_with_every_child_failing_exits_nonzero(
    tmp_path, release_dists, publish_tools
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    names = _manifest_names(dist)
    env = _publish_env(
        publish_tools, _free_port(), "unused-token", CONTINUE_ON_ERROR="true"
    )

    all_failed = _run_publish(root, env, "--test", "--yes", answer="yes\n")

    assert all_failed.returncode != 0, describe_run(all_failed)
    output = all_failed.stdout + all_failed.stderr
    for name in names:
        assert f"Failed to publish {name} 0.2.0 (twine exited 1)" in output, (
            describe_run(all_failed)
        )
    assert f"Failed: 10 package(s): {', '.join(names)}" in output, describe_run(
        all_failed
    )
    assert "Uploaded or already present: 0 package(s)" in output, describe_run(
        all_failed
    )


@pytest.mark.parametrize(
    ("target_args", "upload_host", "index_host"),
    [
        pytest.param(("--test",), "test.pypi.org", "test.pypi.org", id="testpypi"),
        pytest.param((), "upload.pypi.org", "pypi.org", id="pypi"),
    ],
)
def test_publication_uploads_and_verifies_exactly_the_manifest_artifacts(
    tmp_path,
    release_dists,
    publish_tools,
    registry,
    target_args,
    upload_host,
    index_host,
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)

    result = _run_publish(
        root, _registry_env(publish_tools, registry), *target_args, "--yes", "--verbose"
    )

    assert result.returncode == 0, describe_run(result)
    assert "password: <hidden>" in result.stdout + result.stderr, describe_run(result)
    assert registry.token not in result.stdout + result.stderr
    successful_upload_names = set(registry.files())
    expected_names = set(expected)
    assert successful_upload_names == expected_names
    assert registry.files() == expected
    assert registry.proxy.uploads() == [(upload_host, name, 200) for name in expected]
    assert registry.proxy.index_reads() == [
        (index_host, f"/simple/{name}/") for name in _manifest_names(dist)
    ]
    assert "Uploaded or already present: 10 package(s)" in result.stdout, describe_run(
        result
    )
    assert (
        f"Verified 20 file(s) at https://{index_host}/simple/ against the "
        "manifest digests" in result.stdout
    ), describe_run(result)


def test_republishing_identical_artifacts_is_the_accepted_duplicate_skip(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)
    env = _registry_env(publish_tools, registry)
    first = _run_publish(root, env, "--test", "--yes")
    assert first.returncode == 0, describe_run(first)
    registry.proxy.clear()

    genuine_duplicates = _run_publish(root, env, "--test", "--yes")

    assert genuine_duplicates.returncode == 0, describe_run(genuine_duplicates)
    assert registry.files() == expected
    assert registry.proxy.uploads() == [
        ("test.pypi.org", name, 400) for name in expected
    ]
    assert "Uploaded or already present: 10 package(s)" in genuine_duplicates.stdout


def test_duplicate_name_with_different_registry_bytes_fails_publication(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)
    rebuilt = _manifest_files(release_dists["rebuilt"])
    sdk_wheel, sdk_sdist = list(expected)[:2]
    assert [name for name in expected if rebuilt[name] != expected[name]] == [
        sdk_wheel,
        sdk_sdist,
    ]
    shutil.copy2(release_dists["rebuilt"] / sdk_wheel, registry.packages / sdk_wheel)

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", "--yes"
    )

    assert result.returncode != 0, describe_run(result)
    assert registry.files() == {**expected, sdk_wheel: rebuilt[sdk_wheel]}
    assert registry.proxy.uploads() == [("test.pypi.org", sdk_wheel, 400)] + [
        ("test.pypi.org", name, 200) for name in list(expected)[1:]
    ]
    assert (
        f"error: {_TESTPYPI_INDEX}: {sdk_wheel} has sha256 {rebuilt[sdk_wheel]}, "
        f"manifest has {expected[sdk_wheel]}" in result.stderr
    ), describe_run(result)


def test_authentication_refusal_fails_without_storing_any_file(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    names = _manifest_names(dist)
    first_wheel = next(iter(_manifest_files(dist)))
    refused_token = "not-" + registry.token
    env = _publish_env(publish_tools, registry.proxy.port, refused_token)

    result = _run_publish(root, env, "--test", "--yes", "--verbose")

    assert result.returncode != 0, describe_run(result)
    assert "password: <hidden>" in result.stdout + result.stderr, describe_run(result)
    assert registry.token not in result.stdout + result.stderr
    assert refused_token not in result.stdout + result.stderr
    assert registry.files() == {}
    assert registry.proxy.uploads() == [("test.pypi.org", first_wheel, 403)]
    assert registry.proxy.index_reads() == []
    output = result.stdout + result.stderr
    assert "Failed: 1 package(s): cogniverse-sdk" in output, describe_run(result)
    assert f"Not attempted: 9 package(s): {', '.join(names[1:])}" in output, (
        describe_run(result)
    )


def test_server_error_mid_release_fails_and_a_rerun_completes_it(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)
    files = list(expected)
    names = _manifest_names(dist)
    vespa_sdist = "cogniverse_vespa-0.2.0.tar.gz"
    assert files.index(vespa_sdist) == 11
    registry.proxy.faults[vespa_sdist] = "server-error"
    env = _registry_env(publish_tools, registry)

    partial_failure = _run_publish(root, env, "--test", "--yes")

    assert partial_failure.returncode != 0, describe_run(partial_failure)
    assert registry.files() == {name: expected[name] for name in files[:11]}
    assert (
        registry.proxy.uploads()
        == [("test.pypi.org", name, 200) for name in files[:11]]
        + [("test.pypi.org", vespa_sdist, 503)] * 5
    )
    assert registry.proxy.index_reads() == []
    output = partial_failure.stdout + partial_failure.stderr
    assert "Failed: 1 package(s): cogniverse-vespa" in output, describe_run(
        partial_failure
    )
    assert f"Not attempted: 4 package(s): {', '.join(names[6:])}" in output, (
        describe_run(partial_failure)
    )

    registry.proxy.faults.clear()
    registry.proxy.clear()
    resumed = _run_publish(root, env, "--test", "--yes")

    assert resumed.returncode == 0, describe_run(resumed)
    assert registry.files() == expected
    assert registry.proxy.uploads() == [
        ("test.pypi.org", name, 400) for name in files[:11]
    ] + [("test.pypi.org", name, 200) for name in files[11:]]


def test_disconnect_after_the_registry_stored_a_file_fails_and_a_rerun_completes(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)
    files = list(expected)
    core_wheel, core_sdist = files[4:6]
    assert core_wheel == "cogniverse_core-0.2.0-py3-none-any.whl"
    registry.proxy.faults[core_wheel] = "disconnect"
    env = _registry_env(publish_tools, registry, CONTINUE_ON_ERROR="true")

    partial_failure = _run_publish(root, env, "--test", "--yes")

    assert partial_failure.returncode != 0, describe_run(partial_failure)
    assert registry.files() == {
        name: digest for name, digest in expected.items() if name != core_sdist
    }
    assert registry.proxy.uploads() == [
        ("test.pypi.org", name, 200) for name in files if name != core_sdist
    ]
    assert registry.proxy.index_reads() == []
    output = partial_failure.stdout + partial_failure.stderr
    assert "Failed: 1 package(s): cogniverse-core" in output, describe_run(
        partial_failure
    )
    assert "Uploaded or already present: 9 package(s)" in output, describe_run(
        partial_failure
    )

    registry.proxy.faults.clear()
    registry.proxy.clear()
    resumed = _run_publish(root, env, "--test", "--yes")

    assert resumed.returncode == 0, describe_run(resumed)
    assert registry.files() == expected
    assert registry.proxy.uploads() == [
        ("test.pypi.org", name, 200 if name == core_sdist else 400) for name in files
    ]


def test_publish_tool_with_a_mismatched_hash_is_refused_before_any_upload(
    tmp_path, release_dists, publish_tools, registry
):
    root = _publication_root(tmp_path / "publish", release_dists["tagged"])
    requirements = root / "scripts" / _PUBLISH_REQUIREMENTS.name
    pinned = requirements.read_text()
    [twine_hashes] = re.findall(
        rf"^{re.escape(_TWINE)} \\\n((?:    --hash=sha256:[0-9a-f]{{64}}.*\n)+)",
        pinned,
        re.MULTILINE,
    )
    requirements.write_text(
        pinned.replace(
            twine_hashes,
            re.sub(r"sha256:[0-9a-f]{64}", "sha256:" + "0" * 64, twine_hashes),
        )
    )

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", "--yes"
    )

    assert result.returncode == 1, describe_run(result)
    assert registry.proxy.requests == []
    assert registry.files() == {}
    assert f"Hash mismatch for `{_TWINE}`" in result.stderr, describe_run(result)
    assert (
        "Nothing was uploaded: could not install the publish tool from "
        f"{requirements}" in _ANSI.sub("", result.stderr)
    ), describe_run(result)


def test_publish_tool_runs_on_python_3_12_whatever_uv_would_pick(
    tmp_path, release_dists, publish_tools, registry
):
    root = _publication_root(tmp_path / "publish", release_dists["tagged"])
    env = _registry_env(publish_tools, registry, UV_PYTHON="3.11")

    result = _run_publish(root, env, "--test", "--dry-run")

    assert result.returncode == 0, describe_run(result)
    assert re.search(r"^Using CPython 3\.12\.\d+ ", result.stderr, re.MULTILINE), (
        describe_run(result)
    )
    assert registry.proxy.requests == []


def test_dry_run_verifies_artifacts_without_contacting_the_registry(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", "--dry-run"
    )

    assert result.returncode == 0, describe_run(result)
    dry_run_registry_requests = registry.proxy.requests
    assert dry_run_registry_requests == []
    assert registry.files() == {}
    lines = result.stdout.splitlines()
    assert "Target: TestPyPI" in lines
    assert f"Twine: {_TWINE}" in lines
    assert (
        "Mode: DRY RUN (verifies artifacts; uploads nothing and does not query the "
        "index for the release packages)" in lines
    )
    wheels_first = [name for name in expected if name.endswith(".whl")] + [
        name for name in expected if name.endswith(".tar.gz")
    ]
    assert [_ANSI.sub("", line) for line in lines if line.startswith("Checking ")] == [
        f"Checking {name}: PASSED" for name in wheels_first
    ]
    assert [line for line in lines if line.startswith("[DRY RUN]")] == [
        f"[DRY RUN] Would upload {name}" for name in expected
    ]
    assert lines[-1] == (
        "DRY RUN complete: nothing was uploaded and the index was not queried for the "
        "release packages"
    )
    assert "Uploaded or already present" not in result.stdout


@pytest.mark.parametrize(
    "mode_args", [("--yes",), ("--dry-run",)], ids=["publish", "dry-run"]
)
def test_local_version_is_refused_before_any_upload(
    tmp_path, release_dists, publish_tools, registry, mode_args
):
    dist = release_dists["dev"]
    root = _publication_root(tmp_path / "publish", dist)
    version = Version(json.loads((dist / _MANIFEST).read_text())["version"])
    assert version.public == "0.2.1.dev1"

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", *mode_args
    )

    assert result.returncode == 1, describe_run(result)
    assert registry.proxy.requests == []
    assert registry.files() == {}
    assert (
        f"error: release version {version} has the local segment +{version.local}, "
        "which PyPI and TestPyPI reject; publish a build of a release tag"
        in result.stderr
    ), describe_run(result)
    assert "[DRY RUN]" not in result.stdout


def test_artifact_differing_from_the_manifest_is_refused_before_any_upload(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)
    tampered = root / "dist" / "cogniverse_core-0.2.0.tar.gz"
    tampered.write_bytes(tampered.read_bytes() + b"not the built bytes")

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", "--yes"
    )

    assert result.returncode == 1, describe_run(result)
    assert registry.proxy.requests == []
    assert registry.files() == {}
    assert (
        f"error: {tampered}: sha256 {_sha256(tampered)} does not match the manifest "
        f"sha256 {expected[tampered.name]}" in result.stderr
    ), describe_run(result)


def test_missing_manifest_is_refused_before_any_upload(
    tmp_path, release_dists, publish_tools, registry
):
    root = _publication_root(tmp_path / "publish", release_dists["tagged"])
    (root / "dist" / _MANIFEST).unlink()

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", "--yes"
    )

    assert result.returncode == 1, describe_run(result)
    assert registry.proxy.requests == []
    assert (
        f"error: {root / 'dist' / _MANIFEST} not found; run "
        "scripts/build_packages.sh first" in result.stderr
    ), describe_run(result)


@pytest.mark.parametrize("answer", [None, "no\n"], ids=["no-input", "declined"])
def test_unconfirmed_publication_uploads_nothing_and_exits_nonzero(
    tmp_path, release_dists, publish_tools, registry, answer
):
    root = _publication_root(tmp_path / "publish", release_dists["tagged"])

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", answer=answer
    )

    assert result.returncode == 1, describe_run(result)
    assert registry.proxy.requests == []
    assert registry.files() == {}
    assert "Publishing cancelled: nothing was uploaded" in result.stderr, describe_run(
        result
    )


def test_unknown_publish_option_exits_2_without_contacting_the_registry(
    tmp_path, release_dists, publish_tools, registry
):
    root = _publication_root(tmp_path / "publish", release_dists["tagged"])

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", "--no-such-option"
    )

    assert result.returncode == 2, describe_run(result)
    assert "Unknown option: --no-such-option" in result.stderr, describe_run(result)
    assert registry.proxy.requests == []


def test_concurrent_publications_of_different_builds_report_what_the_registry_holds(
    tmp_path, release_dists, publish_tools, registry
):
    roots = {
        label: _publication_root(tmp_path / label, release_dists[label])
        for label in ("tagged", "rebuilt")
    }
    expected = {label: _manifest_files(release_dists[label]) for label in roots}
    assert set(expected["tagged"]) == set(expected["rebuilt"])
    assert expected["tagged"] != expected["rebuilt"]
    env = _registry_env(publish_tools, registry)
    barrier = threading.Barrier(2)
    outcomes: dict[str, tuple[float, float, subprocess.CompletedProcess]] = {}

    def publish(label: str) -> None:
        barrier.wait()
        started = time.monotonic()
        result = _run_publish(roots[label], env, "--test", "--yes")
        outcomes[label] = (started, time.monotonic(), result)

    threads = [threading.Thread(target=publish, args=(label,)) for label in roots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tagged_start, tagged_end, tagged_result = outcomes["tagged"]
    rebuilt_start, rebuilt_end, rebuilt_result = outcomes["rebuilt"]
    assert tagged_start < rebuilt_end and rebuilt_start < tagged_end
    held = registry.files()
    assert set(held) == set(expected["tagged"])
    for name, digest in held.items():
        assert digest in {expected["tagged"][name], expected["rebuilt"][name]}
    for label, result in (("tagged", tagged_result), ("rebuilt", rebuilt_result)):
        assert (result.returncode == 0) == (held == expected[label]), (
            label,
            describe_run(result),
        )
    assert not (tagged_result.returncode == 0 and rebuilt_result.returncode == 0)


@pytest.mark.parametrize("fault", ["server-error", "disconnect"])
def test_index_check_retries_a_transient_index_failure_until_every_file_is_served(
    tmp_path, release_dists, publish_tools, registry, fault
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    names = _manifest_names(dist)
    registry.proxy.index_faults["/simple/cogniverse-core/"] = [fault]
    env = _registry_env(publish_tools, registry, VERIFY_TIMEOUT="120")

    result = _run_publish(root, env, "--test", "--yes")

    assert result.returncode == 0, describe_run(result)
    assert registry.files() == _manifest_files(dist)
    assert (
        registry.proxy.index_reads()
        == [("test.pypi.org", f"/simple/{name}/") for name in names] * 2
    )
    assert (
        "Verified 20 file(s) at https://test.pypi.org/simple/ against the manifest "
        "digests" in result.stdout
    ), describe_run(result)


def test_index_check_fails_when_the_index_stays_unavailable_past_the_timeout(
    tmp_path, release_dists, publish_tools, registry
):
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    registry.proxy.index_faults["/simple/cogniverse-core/"] = ["server-error"]

    result = _run_publish(
        root, _registry_env(publish_tools, registry), "--test", "--yes"
    )

    assert result.returncode == 1, describe_run(result)
    assert registry.files() == _manifest_files(dist)
    assert (
        "error: https://test.pypi.org/simple/ does not serve "
        "cogniverse_core-0.2.0-py3-none-any.whl, cogniverse_core-0.2.0.tar.gz "
        "(last error: https://test.pypi.org/simple/cogniverse-core/: HTTP 503)"
        in result.stderr
    ), describe_run(result)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("TWINE_REPOSITORY_URL", "http://127.0.0.1:9/legacy/"),
        ("TWINE_REPOSITORY", "pypi"),
    ],
)
def test_ambient_twine_repository_setting_is_refused_before_any_upload(
    tmp_path, release_dists, publish_tools, registry, variable, value
):
    root = _publication_root(tmp_path / "publish", release_dists["tagged"])
    env = _registry_env(publish_tools, registry, **{variable: value})

    result = _run_publish(root, env, "--test", "--yes")

    assert result.returncode == 1, describe_run(result)
    assert registry.proxy.requests == []
    assert registry.files() == {}
    assert (
        f"error: {variable} is set; publish_packages.sh publishes only to PyPI, or "
        "TestPyPI with --test; unset it" in result.stderr
    ), describe_run(result)


@pytest.mark.parametrize(
    ("job", "publish_step", "secret", "upload_host", "index_url"),
    [
        pytest.param(
            "publish-testpypi",
            "Publish to TestPyPI",
            "TEST_PYPI_TOKEN",
            "test.pypi.org",
            "https://test.pypi.org/simple/",
            id="testpypi",
        ),
        pytest.param(
            "publish-pypi",
            "Publish to PyPI",
            "PYPI_TOKEN",
            "upload.pypi.org",
            "https://pypi.org/simple/",
            id="pypi",
        ),
    ],
)
def test_workflow_publish_job_verifies_then_publishes_the_manifest_set(
    tmp_path,
    release_dists,
    publish_tools,
    registry,
    job,
    publish_step,
    secret,
    upload_host,
    index_url,
):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "publish-packages.yml").read_text()
    )
    steps = {step["name"]: step for step in workflow["jobs"][job]["steps"]}
    assert [name for name, step in steps.items() if "run" in step] == [
        "Install uv",
        "Verify release artifacts",
        publish_step,
    ]
    assert "if" not in steps["Verify release artifacts"]
    assert steps[publish_step]["if"] == "${{ !inputs.dry_run }}"
    dist = release_dists["tagged"]
    root = _publication_root(tmp_path / "publish", dist)
    expected = _manifest_files(dist)
    secrets_ = {
        name: registry.token if name == secret else f"{name}-for-another-index"
        for name in ("PYPI_TOKEN", "TEST_PYPI_TOKEN")
    }

    def step_env(name: str) -> dict[str, str]:
        declared = {
            variable: re.sub(
                r"\$\{\{ secrets\.(\w+) \}\}",
                lambda match: secrets_[match.group(1)],
                value,
            )
            for variable, value in steps[name].get("env", {}).items()
        }
        return {**_proxy_env(publish_tools, registry.proxy.port), **declared}

    def run_step(name: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-eo",
                "pipefail",
                "-c",
                steps[name]["run"],
            ],
            cwd=root,
            env=step_env(name),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PUBLISH_TIMEOUT,
        )

    verified = run_step("Verify release artifacts")

    assert verified.returncode == 0, describe_run(verified)
    assert registry.proxy.requests == []
    assert registry.files() == {}
    assert (
        "DRY RUN complete: nothing was uploaded and the index was not queried for the "
        "release packages" in verified.stdout
    ), describe_run(verified)

    published = run_step(publish_step)

    assert published.returncode == 0, describe_run(published)
    assert registry.files() == expected
    assert registry.proxy.uploads() == [(upload_host, name, 200) for name in expected]
    assert (
        f"Verified 20 file(s) at {index_url} against the manifest digests"
        in published.stdout
    ), describe_run(published)


def test_release_notes_install_every_package_in_the_manifest(tmp_path, release_dists):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "publish-packages.yml").read_text()
    )
    [step] = [
        step
        for step in workflow["jobs"]["create-release"]["steps"]
        if step["name"] == "Generate release notes"
    ]
    tag_version = "${{ steps.version.outputs.version }}"
    work = tmp_path / "release"
    work.mkdir()
    shutil.copy2(release_dists["tagged"] / _MANIFEST, work / _MANIFEST)

    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-eo",
            "pipefail",
            "-c",
            step["run"].replace(tag_version, "0.2.0"),
        ],
        cwd=work,
        env={
            **os.environ,
            **{
                name: value.replace(tag_version, "0.2.0")
                for name, value in step.get("env", {}).items()
            },
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, describe_run(result)
    lines = (work / "release_notes.md").read_text().splitlines()
    installs = [line for line in lines if line.startswith("pip install ")]
    assert installs == [
        *(
            f"pip install {name}==0.2.0"
            for name in _manifest_names(release_dists["tagged"])
        ),
        "pip install cogniverse-runtime==0.2.0",
    ]
    assert set(_manifest_names(release_dists["tagged"])) == EXPECTED_RELEASE
    assert lines[0] == "# Cogniverse SDK v0.2.0"


@pytest.mark.parametrize("build", ["tagged", "dev"])
def test_wheel_rebuilt_from_each_published_sdist_keeps_the_pins(
    tmp_path, release_dists, build
):
    dist = release_dists[build]
    manifest = json.loads((dist / _MANIFEST).read_text())
    clean = tmp_path / "clean"
    clean.mkdir()

    for package in manifest["packages"]:
        out_dir = tmp_path / "rebuilt" / package["name"]
        rebuilt = subprocess.run(
            [
                "uv",
                "build",
                "--no-sources",
                "--wheel",
                str(dist / package["sdist"]["filename"]),
                "--out-dir",
                str(out_dir),
            ],
            cwd=clean,
            env=_build_env(),
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
        )
        assert rebuilt.returncode == 0, describe_run(rebuilt)
        [wheel] = out_dir.glob("*.whl")
        assert wheel.name == package["wheel"]["filename"]
        published = _wheel_metadata(dist / package["wheel"]["filename"])
        rebuilt_metadata = _wheel_metadata(wheel)
        assert rebuilt_metadata["Version"] == manifest["version"]
        assert rebuilt_metadata.get_all("Requires-Dist") == published.get_all(
            "Requires-Dist"
        )
        internal = _internal_requirement_lines(rebuilt_metadata, EXPECTED_RELEASE)
        assert (
            sorted({canonicalize_name(line.name) for line in internal})
            == (package["requires"])
        )
        assert {str(line.specifier) for line in internal} <= {
            f"=={manifest['version']}"
        }
    assert os.listdir(clean) == []

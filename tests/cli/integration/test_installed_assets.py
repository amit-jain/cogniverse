"""The cogniverse-cli wheel and sdist carry the deployment assets the CLI's path
helpers resolve when no checkout is present: the Helm chart with its
dependencies, the Argo workflow templates and the shared configuration tree.

Canonical asset sets are the tracked files under the directories the same
helpers resolve in a checkout, hashed from the source tree at test time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import textwrap
import threading
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml
from cogniverse_cli.config import (
    LLM_SERVING_LOCAL,
    LLM_SERVING_MODAL,
    get_chart_path,
    get_configs_path,
    get_device_values_file,
    get_llm_serving_values_file,
    get_values_file,
    get_workflows_path,
)
from packaging.requirements import Requirement

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI_ROOT = _REPO_ROOT / "libs" / "cli"
_DATA_PREFIX = "cogniverse_cli/data/"
_HOST_BACKENDS = ("cpu", "cuda", "rocm")
_BUILD_FAILURE_EXIT = 2
_MISSING_ASSETS = "cogniverse-cli is missing required deployment assets: "

# Package-data subdirectory -> the directory the helper resolves in a checkout.
_ASSET_DIRS = {
    "charts/cogniverse": get_chart_path(_REPO_ROOT),
    "workflows": get_workflows_path(_REPO_ROOT),
    "configs": get_configs_path(_REPO_ROOT),
}

_RENDER_VALUES = ("runtime.qualityMonitor.tenantId=test-tenant",)
_PROD_RENDER_VALUES = (
    *_RENDER_VALUES,
    "minio.rootPassword=test-minio",
    "openshell.server.sshHandshakeSecret=test-handshake",
    "phoenix.postgres.auth.password=test-postgres",
    "redis.auth.password=test-redis",
)

_RESOLVE_PROBE = textwrap.dedent(
    """
    import json
    import sys
    import sysconfig

    import yaml

    from cogniverse_cli.config import (
        LLM_SERVING_LOCAL,
        LLM_SERVING_MODAL,
        get_chart_path,
        get_configs_path,
        get_device_values_file,
        get_llm_serving_values_file,
        get_values_file,
        get_workflows_path,
        llm_serving_mode_from_values,
        resolve_project_root,
    )

    def text(path):
        return None if path is None else str(path)

    modal_overlay = get_llm_serving_values_file(LLM_SERVING_MODAL)
    print(json.dumps({
        "python": list(sys.version_info[:2]),
        "purelib": sysconfig.get_paths()["purelib"],
        "project_root": text(resolve_project_root()),
        "paths": {
            "chart": text(get_chart_path()),
            "values_k3s": text(get_values_file()),
            "values_prod": text(get_values_file(prod=True)),
            "device": {
                backend: text(get_device_values_file(backend))
                for backend in sys.argv[1:]
            },
            "llm_serving": {
                mode: text(get_llm_serving_values_file(mode))
                for mode in (LLM_SERVING_LOCAL, LLM_SERVING_MODAL)
            },
            "workflows": text(get_workflows_path()),
            "configs": text(get_configs_path()),
        },
        "serving_mode_of_modal_overlay": llm_serving_mode_from_values(
            yaml.safe_load(modal_overlay.read_text())
        ),
    }))
    """
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_env() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("VIRTUAL_ENV", "PYTHONPATH", "UV_NO_SYNC", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(name, None)
    return environment


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=asset-test",
            "-c",
            "user.email=asset-test@example.invalid",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        check=True,
    )


def _tracked_files(cwd: Path, *paths: str) -> list[str]:
    listed = _git(cwd, "ls-files", "-z", "--cached", "--", *paths).stdout
    return [name for name in listed.decode().split("\0") if name]


def _tracked_hashes(directory: Path) -> dict[str, str]:
    return {
        name: _sha256((directory / name).read_bytes())
        for name in _tracked_files(directory, ".")
    }


def _canonical_asset_hashes() -> dict[str, str]:
    return {
        f"{subdir}/{name}": digest
        for subdir, directory in _ASSET_DIRS.items()
        for name, digest in _tracked_hashes(directory).items()
    }


def _canonical_subset(subdir: str) -> dict[str, str]:
    prefix = f"{subdir}/"
    return {
        name.removeprefix(prefix): digest
        for name, digest in _canonical_asset_hashes().items()
        if name.startswith(prefix)
    }


def _tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path.read_bytes())
        for path in directory.rglob("*")
        if path.is_file()
    }


def _source_snapshot(directory: Path) -> dict[str, str]:
    """Source files, leaving out git metadata and the interpreter's bytecode
    cache of the loaded build hook."""
    return {
        name: digest
        for name, digest in _tree_hashes(directory).items()
        if not {".git", "__pycache__"} & set(Path(name).parts)
    }


def _wheel_asset_hashes(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            name.removeprefix(_DATA_PREFIX): _sha256(archive.read(name))
            for name in archive.namelist()
            if name.startswith(_DATA_PREFIX)
        }


def _sdist_members(sdist: Path) -> dict[str, bytes]:
    with tarfile.open(sdist) as archive:
        return {
            member.name.split("/", 1)[1]: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def _sdist_asset_hashes(sdist: Path) -> dict[str, str]:
    return {
        name.removeprefix(_DATA_PREFIX): _sha256(content)
        for name, content in _sdist_members(sdist).items()
        if name.startswith(_DATA_PREFIX)
    }


def _uv_build(
    source: Path, out_dir: Path, target: str | None, **env: str
) -> subprocess.CompletedProcess:
    """``uv build`` of one target, or with no target the documented path: the
    sdist, then the wheel built from it."""
    selection = [f"--{target}"] if target else []
    return subprocess.run(
        ["uv", "build", *selection, "--out-dir", str(out_dir), str(source)],
        cwd=source,
        env={**_clean_env(), **env},
        capture_output=True,
        text=True,
        timeout=300,
    )


def _build(source: Path, out_dir: Path, target: str) -> Path:
    build = _uv_build(source, out_dir, target)
    assert build.returncode == 0, build.stdout + build.stderr
    pattern = "*.whl" if target == "wheel" else "*.tar.gz"
    (artifact,) = out_dir.glob(pattern)
    return artifact


def _artifacts(out_dir: Path) -> list[str]:
    return sorted(
        path.name for path in (*out_dir.glob("*.whl"), *out_dir.glob("*.tar.gz"))
    )


def _disposable_checkout(destination: Path) -> Path:
    """A git checkout of the CLI package and its asset sources, copied from the
    working tree of this repository, that a test may mutate or delete."""
    for name in _tracked_files(
        _REPO_ROOT, ".gitignore", "libs/cli", *(str(p) for p in _ASSET_DIRS.values())
    ):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / name, target)
    _git(destination, "init", "-q", "-b", "main")
    _git(destination, "add", "-A")
    _git(destination, "commit", "-q", "-m", "snapshot")
    return destination / "libs" / "cli"


def _unpack_sdist(sdist: Path, destination: Path) -> Path:
    with tarfile.open(sdist) as archive:
        archive.extractall(destination, filter="data")
    (project,) = destination.iterdir()
    outside_git = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert outside_git.returncode == 128, outside_git.stdout
    return project


def _workspace_requirements(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(
            archive.read(name).decode()
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        )
    return sorted(
        requirement.name
        for requirement in (
            Requirement(line.removeprefix("Requires-Dist:").strip())
            for line in metadata.splitlines()
            if line.startswith("Requires-Dist:")
        )
        if requirement.name.startswith("cogniverse-")
    )


def _helm_template(chart: Path, values_files: list[Path], set_values: tuple[str, ...]):
    command = ["helm", "template", "cogniverse", str(chart)]
    for values_file in values_files:
        command.extend(["-f", str(values_file)])
    for value in set_values:
        command.extend(["--set", value])
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


def _resolve_probe(venv: Path, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(venv / "bin" / "python"), "-c", _RESOLVE_PROBE, *_HOST_BACKENDS],
        cwd=cwd,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def checkout_wheel(tmp_path_factory) -> Path:
    return _build(_CLI_ROOT, tmp_path_factory.mktemp("checkout-wheel"), "wheel")


@pytest.mark.integration
def test_required_assets_are_the_files_the_consumers_name() -> None:
    hook = tomllib.loads((_CLI_ROOT / "pyproject.toml").read_text())["tool"]["hatch"][
        "build"
    ]["hooks"]["custom"]

    assert hook["asset-roots"] == [
        directory.relative_to(_REPO_ROOT).as_posix()
        for directory in _ASSET_DIRS.values()
    ]
    assert list(_ASSET_DIRS) == hook["asset-roots"]

    chart = _ASSET_DIRS["charts/cogniverse"]
    chart_files = {
        "Chart.yaml",
        "Chart.lock",
        "values.yaml",
        get_values_file(_REPO_ROOT).name,
        get_values_file(_REPO_ROOT, prod=True).name,
        *(get_device_values_file(b, _REPO_ROOT).name for b in _HOST_BACKENDS),
        *(
            overlay.name
            for mode in (LLM_SERVING_LOCAL, LLM_SERVING_MODAL)
            if (overlay := get_llm_serving_values_file(mode, _REPO_ROOT)) is not None
        ),
        *re.findall(
            r'\.Files\.Get "([^"]+)"',
            "".join(
                p.read_text()
                for p in sorted((chart / "templates").rglob("*"))
                if p.is_file()
            ),
        ),
    }
    for dependency in yaml.safe_load((chart / "Chart.lock").read_text())[
        "dependencies"
    ]:
        archive = f"charts/{dependency['name']}-{dependency['version']}.tgz"
        unpacked = f"charts/{dependency['name']}/Chart.yaml"
        chart_files.add(archive if (chart / archive).is_file() else unpacked)
    workflow_files = {p.name for p in _ASSET_DIRS["workflows"].glob("*.yaml")}

    assert sorted(hook["required-assets"]) == sorted(
        {f"charts/cogniverse/{name}" for name in chart_files}
        | {f"workflows/{name}" for name in workflow_files}
        | {"configs/config.json"}
    )


@pytest.mark.integration
def test_installed_cli_resolves_every_consumer_to_the_packaged_assets(
    checkout_wheel: Path, tmp_path: Path
) -> None:
    wheels = {"cogniverse-cli": checkout_wheel}
    pending = _workspace_requirements(checkout_wheel)
    while pending:
        package = pending.pop()
        if package in wheels:
            continue
        wheels[package] = _build(
            _REPO_ROOT / "libs" / package.removeprefix("cogniverse-"),
            tmp_path / package,
            "wheel",
        )
        pending.extend(_workspace_requirements(wheels[package]))
    assert sorted(wheels) == [
        "cogniverse-cli",
        "cogniverse-foundation",
        "cogniverse-sdk",
    ]

    venv = tmp_path / "venv"
    for command in (
        ["uv", "venv", "--python", "3.12", "--no-project", str(venv)],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            *(str(wheel) for wheel in wheels.values()),
        ],
    ):
        step = subprocess.run(
            command,
            cwd=tmp_path,
            env=_clean_env(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert step.returncode == 0, step.stdout + step.stderr

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    probe = _resolve_probe(venv, unrelated)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    resolved = json.loads(probe.stdout)

    data = Path(resolved["purelib"]) / "cogniverse_cli" / "data"
    chart = data / "charts" / "cogniverse"
    assert Path(resolved["purelib"]).is_relative_to(venv)
    assert resolved["python"] == [3, 12]
    assert resolved["project_root"] is None
    assert resolved["paths"] == {
        "chart": str(chart),
        "values_k3s": str(chart / "values.k3s.yaml"),
        "values_prod": str(chart / "values.prod.yaml"),
        "device": {
            backend: str(chart / f"values.{backend}.yaml") for backend in _HOST_BACKENDS
        },
        "llm_serving": {
            LLM_SERVING_LOCAL: None,
            LLM_SERVING_MODAL: str(chart / "values.modal-llm.yaml"),
        },
        "workflows": str(data / "workflows"),
        "configs": str(data / "configs"),
    }
    assert resolved["serving_mode_of_modal_overlay"] == LLM_SERVING_MODAL

    foreign_workspace = tmp_path / "foreign-workspace"
    for name, content in {
        "pyproject.toml": '[tool.uv.workspace]\nmembers = []\n\n[project]\nname = "demo"\n',
        "workflows/stray.yaml": "kind: WorkflowTemplate\n",
        "configs/config.json": "{}\n",
        "charts/cogniverse/Chart.yaml": "name: demo\n",
        **{
            f"charts/cogniverse/values.{name}.yaml": "{}\n"
            for name in ("k3s", "prod", "modal-llm", *_HOST_BACKENDS)
        },
    }.items():
        (foreign_workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (foreign_workspace / name).write_text(content)
    from_foreign_workspace = _resolve_probe(venv, foreign_workspace)
    assert from_foreign_workspace.returncode == 0, (
        from_foreign_workspace.stdout + from_foreign_workspace.stderr
    )
    assert json.loads(from_foreign_workspace.stdout) == resolved

    installed_chart_hashes = _tree_hashes(chart)
    installed_workflow_hashes = _tree_hashes(data / "workflows")
    installed_config_hashes = _tree_hashes(data / "configs")
    canonical_chart_hashes = _canonical_subset("charts/cogniverse")
    canonical_workflow_hashes = _canonical_subset("workflows")
    canonical_config_hashes = _canonical_subset("configs")
    assert installed_chart_hashes == canonical_chart_hashes
    assert installed_config_hashes == canonical_config_hashes
    assert installed_workflow_hashes == canonical_workflow_hashes
    assert sorted(p.name for p in data.iterdir()) == ["charts", "configs", "workflows"]

    paths = resolved["paths"]
    source_chart = _ASSET_DIRS["charts/cogniverse"]
    for base, device, set_values in (
        (paths["values_k3s"], paths["device"]["rocm"], _RENDER_VALUES),
        (paths["values_prod"], paths["device"]["cuda"], _PROD_RENDER_VALUES),
    ):
        packaged = [Path(base), Path(device), Path(paths["llm_serving"]["modal"])]
        packaged_helm_render = _helm_template(chart, packaged, set_values)
        source_render = _helm_template(
            source_chart, [source_chart / p.name for p in packaged], set_values
        )
        assert packaged_helm_render.returncode == 0, packaged_helm_render.stderr
        assert source_render.returncode == 0, source_render.stderr
        assert packaged_helm_render.stdout == source_render.stdout


@pytest.mark.integration
def test_wheel_from_unpacked_sdist_carries_the_checkout_wheel_assets(
    checkout_wheel: Path, tmp_path: Path
) -> None:
    source = _disposable_checkout(tmp_path / "checkout")
    # A tracked asset whose bytes exist only in this checkout: a build that read
    # the workspace instead of the sdist would carry the workspace's bytes.
    sentinel = "configs/config.json"
    modified = (tmp_path / "checkout" / sentinel).read_bytes() + b"\n"
    (tmp_path / "checkout" / sentinel).write_bytes(modified)
    _git(tmp_path / "checkout", "commit", "-q", "-am", "sentinel")
    expected = {**_canonical_asset_hashes(), sentinel: _sha256(modified)}
    assert expected != _canonical_asset_hashes()
    sdist = _build(source, tmp_path / "sdist", "sdist")
    disposable_checkout_wheel = _build(source, tmp_path / "checkout-wheel", "wheel")
    shutil.rmtree(tmp_path / "checkout")

    members = _sdist_members(sdist)
    assert members["hatch_build.py"] == (_CLI_ROOT / "hatch_build.py").read_bytes()
    assert members["pyproject.toml"] == (_CLI_ROOT / "pyproject.toml").read_bytes()
    assert _sdist_asset_hashes(sdist) == expected

    project = _unpack_sdist(sdist, tmp_path / "unpacked")
    wheel = _build(project, tmp_path / "wheel", "wheel")

    wheel_from_sdist_asset_hashes = _wheel_asset_hashes(wheel)
    wheel_from_checkout_asset_hashes = _wheel_asset_hashes(disposable_checkout_wheel)
    assert wheel_from_checkout_asset_hashes == expected
    assert wheel_from_sdist_asset_hashes == wheel_from_checkout_asset_hashes
    assert _wheel_asset_hashes(checkout_wheel) == _canonical_asset_hashes()


@pytest.mark.integration
@pytest.mark.parametrize("target", ["wheel", "sdist"])
@pytest.mark.parametrize("loss", ["deleted", "untracked"])
def test_checkout_build_fails_without_a_required_asset(
    tmp_path: Path, target: str, loss: str
) -> None:
    source = _disposable_checkout(tmp_path / "checkout")
    required = "charts/cogniverse/values.rocm.yaml"
    if loss == "deleted":
        (tmp_path / "checkout" / required).unlink()
    else:
        _git(tmp_path / "checkout", "rm", "-q", "--cached", required)

    build = _uv_build(source, tmp_path / "out", target)

    assert build.returncode == _BUILD_FAILURE_EXIT, build.stdout + build.stderr
    assert f"RuntimeError: {_MISSING_ASSETS}{required}\n" in build.stderr
    assert _artifacts(tmp_path / "out") == []


@pytest.mark.integration
def test_sdist_build_fails_without_a_required_asset(tmp_path: Path) -> None:
    source = _disposable_checkout(tmp_path / "checkout")
    sdist = _build(source, tmp_path / "sdist", "sdist")
    project = _unpack_sdist(sdist, tmp_path / "unpacked")
    required = "workflows/tenant-provisioning.yaml"
    (project / _DATA_PREFIX / required).unlink()

    build = _uv_build(project, tmp_path / "out", "wheel")

    assert build.returncode == _BUILD_FAILURE_EXIT, build.stdout + build.stderr
    assert f"RuntimeError: {_MISSING_ASSETS}{required}\n" in build.stderr
    assert _artifacts(tmp_path / "out") == []


@pytest.mark.integration
def test_build_outside_a_git_checkout_or_sdist_names_the_requirement(
    tmp_path: Path,
) -> None:
    source = _disposable_checkout(tmp_path / "checkout")
    shutil.rmtree(tmp_path / "checkout" / ".git")

    build = _uv_build(
        source, tmp_path / "out", "wheel", SETUPTOOLS_SCM_PRETEND_VERSION="1.2.3"
    )

    assert build.returncode == _BUILD_FAILURE_EXIT, build.stdout + build.stderr
    assert (
        "RuntimeError: cogniverse-cli builds its deployment assets from a git "
        "checkout or an sdist; git rev-parse --show-toplevel failed in "
        f"{source}: fatal: not a git repository"
    ) in build.stderr
    assert _artifacts(tmp_path / "out") == []


@pytest.mark.integration
def test_untracked_files_are_not_packaged(tmp_path: Path) -> None:
    source = _disposable_checkout(tmp_path / "checkout")
    for stray in (
        tmp_path / "checkout" / "charts" / "cogniverse" / "values.local.yaml",
        tmp_path / "checkout" / "configs" / "local-override.json",
        source / "cogniverse_cli" / "data" / "charts" / "cogniverse" / "stale.yaml",
    ):
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("stray: true\n")

    wheel = _build(source, tmp_path / "wheel", "wheel")
    sdist = _build(source, tmp_path / "sdist", "sdist")

    assert _wheel_asset_hashes(wheel) == _canonical_asset_hashes()
    assert _sdist_asset_hashes(sdist) == _canonical_asset_hashes()


@pytest.mark.integration
def test_parallel_builds_share_no_writable_staging(tmp_path: Path) -> None:
    source = _disposable_checkout(tmp_path / "checkout")
    project = _unpack_sdist(
        _build(source, tmp_path / "sdist", "sdist"), tmp_path / "unpacked"
    )
    sources_before = {
        "checkout": _source_snapshot(tmp_path / "checkout"),
        "unpacked": _source_snapshot(project),
    }
    jobs = [
        *((source, tmp_path / f"checkout-{i}", "wheel") for i in range(2)),
        *((project, tmp_path / f"unpacked-{i}", "wheel") for i in range(2)),
        *((source, tmp_path / f"sdist-then-wheel-{i}", None) for i in range(2)),
    ]
    start = threading.Barrier(len(jobs))
    results: list[subprocess.CompletedProcess | None] = [None] * len(jobs)

    def build(index: int, job_source: Path, out_dir: Path, target: str | None):
        start.wait()
        results[index] = _uv_build(job_source, out_dir, target)

    threads = [
        threading.Thread(target=build, args=(index, *job))
        for index, job in enumerate(jobs)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [r.returncode for r in results] == [0] * len(jobs), [
        r.stderr for r in results
    ]
    assert [
        _wheel_asset_hashes(next(out_dir.glob("*.whl"))) for _, out_dir, _ in jobs
    ] == [_canonical_asset_hashes()] * len(jobs)
    assert [
        _sdist_asset_hashes(next(out_dir.glob("*.tar.gz")))
        for _, out_dir, target in jobs
        if target is None
    ] == [_canonical_asset_hashes()] * 2
    assert [len(_artifacts(out_dir)) for _, out_dir, target in jobs] == [
        1,
        1,
        1,
        1,
        2,
        2,
    ]
    assert {
        "checkout": _source_snapshot(tmp_path / "checkout"),
        "unpacked": _source_snapshot(project),
    } == sources_before

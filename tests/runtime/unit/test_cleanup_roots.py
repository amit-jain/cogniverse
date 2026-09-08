"""Cleanup rejects unsafe roots before opening a backend or deleting files."""

import asyncio
import inspect
import os
import sys
import time
from pathlib import Path

import pytest

from cogniverse_runtime import optimization_cli as oc

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def cleanup_args(tmp_path):
    log_dir = tmp_path / "logs"
    temp_dir = tmp_path / "scratch"
    log_dir.mkdir()
    temp_dir.mkdir()
    return dict(
        tenant_id=None,
        log_dir=str(log_dir),
        temp_dir=str(temp_dir),
        log_retention_days=7,
        memory_retention_days=30,
        temp_retention_days=1,
        schemas_dir="configs/schemas",
        config_keep_versions=10,
    )


@pytest.mark.parametrize("parameter", ["log_dir", "temp_dir"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " ",
        "/",
        "/tmp",
        "/var/tmp",
        "home",
        "missing",
        "file",
        "interpreter",
        "resolved_interpreter",
        "interpreter_parent_symlink",
        "checkout",
        "checkout_parent",
        "symlink_tmp",
    ],
)
def test_run_cleanup_rejects_root(
    parameter, value, cleanup_args, tmp_path, monkeypatch
):
    if value == "home":
        value = str(Path.home())
    elif value == "missing":
        value = str(tmp_path / "missing")
    elif value == "file":
        file = tmp_path / "file"
        file.write_text("keep")
        value = str(file)
    elif value in ("interpreter", "resolved_interpreter"):
        executable = tmp_path / "venv" / "bin" / "python"
        executable.parent.mkdir(parents=True)
        executable.write_text("interpreter")
        if value == "resolved_interpreter":
            link = tmp_path / "python-link"
            link.symlink_to(executable)
            monkeypatch.setattr(sys, "executable", str(link))
        else:
            monkeypatch.setattr(sys, "executable", str(executable))
        value = str(executable.parent.parent)
    elif value == "interpreter_parent_symlink":
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").symlink_to(sys.executable)
        alias = tmp_path / "venv-alias"
        alias.symlink_to(venv, target_is_directory=True)
        monkeypatch.setattr(sys, "executable", str(alias / "bin" / "python"))
        value = str(venv)
    elif value in ("checkout", "checkout_parent"):
        repo = Path(oc.__file__).resolve().parents[3]
        value = str(repo if value == "checkout" else repo.parent)
    elif value == "symlink_tmp":
        link = tmp_path / "tmp-link"
        link.symlink_to("/tmp", target_is_directory=True)
        value = str(link)

    def refuse_backend():
        raise AssertionError("unsafe roots reached backend initialization")

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        refuse_backend,
    )
    cleanup_args[parameter] = value
    with pytest.raises(oc.CleanupRootError) as exc:
        asyncio.run(oc.run_cleanup(**cleanup_args))
    assert exc.value.parameter == parameter
    assert str(exc.value).startswith(f"{parameter}: ")
    assert (
        sorted(
            p.name
            for p in Path(
                cleanup_args["temp_dir" if parameter == "log_dir" else "log_dir"]
            ).iterdir()
        )
        == []
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "log_dir",
        "temp_dir",
        "log_retention_days",
        "memory_retention_days",
        "temp_retention_days",
    ],
)
def test_cleanup_parameters_have_no_defaults(parameter, cleanup_args):
    assert (
        inspect.signature(oc.run_cleanup).parameters[parameter].default
        is inspect.Parameter.empty
    )
    del cleanup_args[parameter]
    with pytest.raises(TypeError, match=parameter):
        asyncio.run(oc.run_cleanup(**cleanup_args))


@pytest.mark.parametrize("retention", [1, 7])
def test_prune_exact_old_file_and_report(tmp_path, retention):
    nested = tmp_path / "nested"
    nested.mkdir()
    old = nested / "old"
    fresh = nested / "fresh"
    old.write_text("old data")
    fresh.write_text("fresh data")
    age = time.time() - (retention + 1) * 86400
    os.utime(old, (age, age))
    assert oc._prune_aged_files(str(tmp_path), older_than_days=retention) == {
        "path": str(tmp_path),
        "scanned": 2,
        "deleted": 1,
        "errors": [],
    }
    assert sorted(p.name for p in nested.iterdir()) == ["fresh"]
    assert fresh.read_text() == "fresh data"


def test_prune_reports_unlink_failure(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.write_text("keep on failure")
    os.utime(old, (0, 0))

    def deny_unlink(self, *args, **kwargs):
        raise PermissionError("unlink denied")

    monkeypatch.setattr(Path, "unlink", deny_unlink)
    assert oc._prune_aged_files(str(tmp_path), older_than_days=1) == {
        "path": str(tmp_path),
        "scanned": 1,
        "deleted": 0,
        "errors": [f"{old}: unlink denied"],
    }
    assert old.read_text() == "keep on failure"


def test_concurrent_pruning_keeps_roots_separate(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    roots = [tmp_path / name for name in ("logs", "temp")]
    for root in roots:
        root.mkdir()
        (root / "old").write_text(root.name)
        (root / "fresh").write_text(root.name)
        os.utime(root / "old", (0, 0))
    barrier = Barrier(2)

    def sweep(root):
        barrier.wait(timeout=5)
        return oc._prune_aged_files(str(root), older_than_days=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(sweep, roots))
    assert reports == [
        {"path": str(root), "scanned": 2, "deleted": 1, "errors": []} for root in roots
    ]
    assert [[p.name for p in root.iterdir()] for root in roots] == [
        ["fresh"],
        ["fresh"],
    ]
    assert [(root / "fresh").read_text() for root in roots] == ["logs", "temp"]

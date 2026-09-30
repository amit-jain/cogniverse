"""arize-phoenix-client's pytest plugin stays out of every test session.

Its ``pytest11`` entry point imports ``phoenix/__init__.py``, which loads the
whole Phoenix server into the session heap.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_PLUGIN_NAME = "phoenix"


def _tracked_ini_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "pytest.ini", "*/pytest.ini"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO_ROOT / name for name in listed]


def _active_plugins(ini: Path, target: Path) -> dict[str, str]:
    output = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ini),
            "--rootdir",
            str(ini.parent),
            "--trace-config",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            str(target),
        ],
        cwd=ini.parent,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    table = output.split("active plugins:\n", 1)[1].split("rootdir:", 1)[0]
    rows = (line.split(":", 1) for line in table.splitlines() if " : " in line)
    return {name.strip(): value.strip() for name, value in rows}


def test_phoenix_plugin_is_installed_as_a_pytest11_entry_point():
    values = [ep.value for ep in entry_points(group="pytest11", name=_PLUGIN_NAME)]
    assert values == ["phoenix.client.pytest.plugin"]


def test_running_session_blocks_the_phoenix_plugin(pytestconfig):
    manager = pytestconfig.pluginmanager
    assert manager.is_blocked(_PLUGIN_NAME) is True
    assert manager.get_plugin(_PLUGIN_NAME) is None


def test_repo_has_the_pytest_ini_files_this_guard_checks():
    relative = sorted(str(p.relative_to(REPO_ROOT)) for p in _tracked_ini_files())
    assert relative == ["pytest.ini", "tests/ingestion/pytest.ini"]


@pytest.mark.parametrize(
    "ini", _tracked_ini_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_every_pytest_ini_keeps_the_phoenix_plugin_unloaded(ini, tmp_path):
    plugins = _active_plugins(ini, tmp_path)
    assert plugins[_PLUGIN_NAME] == "None"
    assert plugins["pytest_cov"] == str(Path(sys.modules["pytest_cov.plugin"].__file__))

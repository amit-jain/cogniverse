"""Unit tests for cogniverse_cli.config path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from cogniverse_cli.config import (
    get_chart_path,
    resolve_project_root,
)


def test_resolve_project_root_finds_workspace(tmp_path: Path) -> None:
    """A directory containing a pyproject.toml with [tool.uv.workspace]
    is correctly identified as the project root."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.uv.workspace]\nmembers = ["libs/*"]\n',
        encoding="utf-8",
    )
    # Starting from a nested subdirectory should still find the root.
    nested = tmp_path / "libs" / "cli" / "cogniverse_cli"
    nested.mkdir(parents=True)

    result = resolve_project_root(start=nested)

    assert result == tmp_path


def test_resolve_project_root_returns_none_when_missing(tmp_path: Path) -> None:
    """When no ancestor has a workspace pyproject.toml, return None."""
    # tmp_path has no pyproject.toml at all.
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    result = resolve_project_root(start=nested)

    assert result is None


def test_resolve_project_root_ignores_non_workspace_pyproject(
    tmp_path: Path,
) -> None:
    """A pyproject.toml that does *not* contain [tool.uv.workspace] is
    skipped — resolution continues upward."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "some-lib"\n',
        encoding="utf-8",
    )
    nested = tmp_path / "src"
    nested.mkdir()

    result = resolve_project_root(start=nested)

    assert result is None


def test_get_chart_path_dev_mode(tmp_path: Path) -> None:
    """In dev mode the chart directory under the project root is
    returned."""
    chart_dir = tmp_path / "charts" / "cogniverse"
    chart_dir.mkdir(parents=True)

    result = get_chart_path(project_root=tmp_path)

    assert result == chart_dir


def test_get_chart_path_raises_when_missing(tmp_path: Path) -> None:
    """FileNotFoundError is raised when the chart directory does not
    exist in the project tree or package data."""
    with pytest.raises(FileNotFoundError):
        get_chart_path(project_root=tmp_path)

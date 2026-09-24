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
        '[tool.uv.workspace]\nmembers = ["libs/*"]\n\n[project]\nname = "cogniverse"\n',
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


def test_resolve_project_root_skips_a_foreign_workspace(tmp_path: Path) -> None:
    """A uv workspace whose project is not cogniverse is not the project
    root, even with its own charts, workflows and configs; resolution
    continues upward."""
    foreign = tmp_path / "foreign"
    for name in ("charts/cogniverse", "workflows", "configs"):
        (foreign / name).mkdir(parents=True)
    (foreign / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = []\n\n[project]\nname = "demo"\n',
        encoding="utf-8",
    )

    assert resolve_project_root(start=foreign / "workflows") is None

    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["libs/*"]\n\n[project]\nname = "cogniverse"\n',
        encoding="utf-8",
    )

    assert resolve_project_root(start=foreign / "workflows") == tmp_path


def test_resolve_project_root_requires_the_workspace_table(tmp_path: Path) -> None:
    """A cogniverse project outside a uv workspace (an unpacked sdist of the
    root project, say) is not the monorepo root."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "cogniverse"\n', encoding="utf-8"
    )

    assert resolve_project_root(start=tmp_path) is None


def test_resolve_project_root_names_an_unparseable_pyproject(tmp_path: Path) -> None:
    """A malformed pyproject.toml on the way up fails with its path instead of
    being mistaken for, or silently skipped as, the project root."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.uv.workspace\n", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        resolve_project_root(start=tmp_path)

    assert str(raised.value).startswith(f"Cannot parse {pyproject}: ")


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

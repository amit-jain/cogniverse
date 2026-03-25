"""Unit tests for cogniverse_cli.images build and import utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from cogniverse_cli.images import build_images, has_workspace_source, import_images


class TestHasWorkspaceSource:
    """Tests for :func:`has_workspace_source`."""

    def test_has_workspace_source_true(self, tmp_path: Path) -> None:
        """Returns True when libs/runtime directory exists."""
        (tmp_path / "libs" / "runtime").mkdir(parents=True)

        assert has_workspace_source(tmp_path) is True

    def test_has_workspace_source_false(self, tmp_path: Path) -> None:
        """Returns False when libs/runtime directory is missing."""
        assert has_workspace_source(tmp_path) is False


class TestBuildImages:
    """Tests for :func:`build_images`."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_calls_docker_build(
        self, mock_run: object
    ) -> None:
        """Two docker build commands are issued, one per image."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        tags = build_images(Path("/fake/root"))

        assert len(tags) == 2
        assert mock_run.call_count == 2  # type: ignore[attr-defined]

        for call in mock_run.call_args_list:  # type: ignore[attr-defined]
            cmd = call[0][0]
            assert cmd[0] == "docker"
            assert cmd[1] == "build"


class TestImportImages:
    """Tests for :func:`import_images`."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_import_images_calls_k3d_import(self, mock_run: object) -> None:
        """The k3d image import command includes all tags and cluster name."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        import_images("cogniverse", ["img:a", "img:b"])

        call_args = mock_run.call_args  # type: ignore[attr-defined]
        cmd = call_args[0][0]
        assert cmd[:3] == ["k3d", "image", "import"]
        assert "img:a" in cmd
        assert "img:b" in cmd
        assert "-c" in cmd
        assert "cogniverse" in cmd

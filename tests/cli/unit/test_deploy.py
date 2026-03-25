"""Unit tests for cogniverse_cli.deploy Helm helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from cogniverse_cli.deploy import helm_install


class TestHelmInstall:
    """Tests for :func:`helm_install`."""

    @patch("cogniverse_cli.deploy.subprocess.run", return_value=MagicMock(returncode=0))
    @patch("cogniverse_cli.deploy.release_exists", return_value=False)
    def test_helm_install_new_release(
        self, mock_exists: object, mock_run: object
    ) -> None:
        """When no release exists the command uses 'install'."""
        chart = Path("/charts/cogniverse")
        values = Path("/charts/cogniverse/values.yaml")

        helm_install(chart, values, name="cogniverse", namespace="cogniverse")

        mock_run.assert_called_once()  # type: ignore[attr-defined]
        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        assert cmd[1] == "install"

    @patch("cogniverse_cli.deploy.subprocess.run", return_value=MagicMock(returncode=0))
    @patch("cogniverse_cli.deploy.release_exists", return_value=True)
    def test_helm_install_upgrade_existing(
        self, mock_exists: object, mock_run: object
    ) -> None:
        """When a release already exists the command uses 'upgrade'."""
        chart = Path("/charts/cogniverse")
        values = Path("/charts/cogniverse/values.yaml")

        helm_install(chart, values, name="cogniverse", namespace="cogniverse")

        mock_run.assert_called_once()  # type: ignore[attr-defined]
        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        assert cmd[1] == "upgrade"

    @patch("cogniverse_cli.deploy.subprocess.run", return_value=MagicMock(returncode=0))
    @patch("cogniverse_cli.deploy.release_exists", return_value=False)
    def test_helm_install_includes_set_overrides(
        self, mock_exists: object, mock_run: object
    ) -> None:
        """--set key=value pairs are appended to the command."""
        chart = Path("/charts/cogniverse")
        values = Path("/charts/cogniverse/values.yaml")

        helm_install(
            chart,
            values,
            set_values={"image.tag": "latest", "replicas": "3"},
        )

        mock_run.assert_called_once()  # type: ignore[attr-defined]
        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        assert "--set" in cmd
        assert "image.tag=latest" in cmd
        assert "replicas=3" in cmd

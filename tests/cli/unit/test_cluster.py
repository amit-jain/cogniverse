"""Unit tests for cogniverse_cli.cluster lifecycle utilities."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from cogniverse_cli.cluster import (
    DEFAULT_PORTS,
    check_prerequisites,
    cluster_exists,
    create_cluster,
    has_existing_k8s,
)


class TestCheckPrerequisites:
    """Tests for :func:`check_prerequisites`."""

    @patch("cogniverse_cli.cluster.shutil.which")
    def test_check_prerequisites_all_present(self, mock_which: object) -> None:
        """When all tools are on PATH, the missing list is empty."""
        mock_which.return_value = "/usr/local/bin/tool"  # type: ignore[attr-defined]

        result = check_prerequisites(require_k3d=True)

        assert result == []

    @patch("cogniverse_cli.cluster.shutil.which")
    def test_check_prerequisites_missing_k3d(self, mock_which: object) -> None:
        """When k3d is missing, it appears in the result list."""

        def _side_effect(name: str) -> str | None:
            if name == "k3d":
                return None
            return f"/usr/local/bin/{name}"

        mock_which.side_effect = _side_effect  # type: ignore[attr-defined]

        result = check_prerequisites(require_k3d=True)

        assert result == ["k3d"]


class TestClusterExists:
    """Tests for :func:`cluster_exists`."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_cluster_exists_true(self, mock_run: object) -> None:
        """Returns True when k3d reports the cluster exists."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        assert cluster_exists("cogniverse") is True

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_cluster_exists_false(self, mock_run: object) -> None:
        """Returns False when k3d reports the cluster does not exist."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=1
        )

        assert cluster_exists("cogniverse") is False


class TestCreateCluster:
    """Tests for :func:`create_cluster`."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_create_cluster_builds_correct_command(
        self, mock_run: object
    ) -> None:
        """All default ports produce -p flags in the subprocess command."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        create_cluster("cogniverse")

        call_args = mock_run.call_args  # type: ignore[attr-defined]
        cmd = call_args[0][0]  # positional arg 0 is the command list

        assert cmd[:4] == ["k3d", "cluster", "create", "cogniverse"]

        # Each default port should produce a -p flag
        port_flags = [
            cmd[i + 1]
            for i in range(len(cmd))
            if cmd[i] == "-p"
        ]
        assert len(port_flags) == len(DEFAULT_PORTS)
        for port in DEFAULT_PORTS:
            assert f"{port}:{port}@loadbalancer" in port_flags


class TestHasExistingK8s:
    """Tests for :func:`has_existing_k8s`."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_has_existing_k8s_true(self, mock_run: object) -> None:
        """Returns True when kubectl cluster-info succeeds."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        assert has_existing_k8s() is True

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_has_existing_k8s_false(self, mock_run: object) -> None:
        """Returns False when kubectl cluster-info fails with non-zero exit."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=1
        )

        assert has_existing_k8s() is False

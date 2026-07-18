"""Unit tests for cogniverse_cli.deploy Helm helpers."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
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


class TestSemverChartVersion:
    """PEP 440 git versions must translate to valid SemVer chart versions."""

    def test_dev_version_with_local_segment(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        assert (
            semver_chart_version("0.1.dev2395+g1998ce1b5.d20260717")
            == "0.1.0-dev.2395+g1998ce1b5.d20260717"
        )

    def test_three_part_dev_version(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        assert semver_chart_version("0.2.1.dev3+gabc123") == "0.2.1-dev.3+gabc123"

    def test_release_version_passes_through(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        assert semver_chart_version("0.1.0") == "0.1.0"

    def test_local_segment_underscores_sanitized(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        assert semver_chart_version("0.1.dev5+gab_cd") == "0.1.0-dev.5+gab-cd"

    def test_rc_prerelease(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        assert semver_chart_version("1.0.0rc1") == "1.0.0-rc.1"

    def test_alpha_prerelease(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        assert semver_chart_version("1.0.0a2") == "1.0.0-alpha.2"

    def test_beta_prerelease(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        assert semver_chart_version("1.0.0b1") == "1.0.0-beta.1"

    def test_rc_with_dev_and_local_segments(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        out = semver_chart_version("0.2.0rc1.dev5+gabc")
        assert out == "0.2.0-rc.1.dev.5+gabc"
        assert re.fullmatch(
            r"\d+\.\d+\.\d+"
            r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?",
            out,
        ), f"not valid SemVer: {out}"

    def test_post_release_refused(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        with pytest.raises(ValueError, match="post-release"):
            semver_chart_version("1.0.0.post1")

    def test_epoch_refused(self) -> None:
        from cogniverse_cli.deploy import semver_chart_version

        with pytest.raises(ValueError, match="epoch"):
            semver_chart_version("1!2.0.0")


class TestChartVersionStamping:
    """helm_install(chart_version=...) packages the chart with the git
    version so `helm list` shows dev provenance instead of the static
    Chart.yaml line."""

    @patch("cogniverse_cli.deploy.release_exists", return_value=True)
    def test_chart_version_packages_then_upgrades_the_tgz(
        self, mock_exists: object
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            result = MagicMock(returncode=0, stderr="")
            if cmd[:2] == ["helm", "package"]:
                result.stdout = (
                    "Successfully packaged chart and saved it to: "
                    "/tmp/x/cogniverse-0.1.0-dev.2395+g1998ce1b5.tgz\n"
                )
            return result

        with patch("cogniverse_cli.deploy.subprocess.run", side_effect=fake_run):
            helm_install(
                Path("/charts/cogniverse"),
                Path("/charts/cogniverse/values.yaml"),
                chart_version="0.1.dev2395+g1998ce1b5",
            )

        package_cmd = calls[0]
        assert package_cmd[:2] == ["helm", "package"]
        version_flag = package_cmd[package_cmd.index("--version") + 1]
        app_version_flag = package_cmd[package_cmd.index("--app-version") + 1]
        assert version_flag == "0.1.0-dev.2395+g1998ce1b5"
        assert app_version_flag == "0.1.dev2395+g1998ce1b5"

        upgrade_cmd = calls[1]
        assert upgrade_cmd[1] == "upgrade"
        assert upgrade_cmd[3] == "/tmp/x/cogniverse-0.1.0-dev.2395+g1998ce1b5.tgz"

    @patch("cogniverse_cli.deploy.subprocess.run", return_value=MagicMock(returncode=0))
    @patch("cogniverse_cli.deploy.release_exists", return_value=True)
    def test_without_chart_version_uses_the_chart_dir(
        self, mock_exists: object, mock_run: object
    ) -> None:
        helm_install(Path("/charts/cogniverse"), Path("/v.yaml"))
        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        assert cmd[3] == "/charts/cogniverse"

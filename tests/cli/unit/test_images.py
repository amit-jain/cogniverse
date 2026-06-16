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
    def test_build_images_calls_docker_build(self, mock_run: object) -> None:
        """One docker build per cogniverse-owned image: backend-specific
        runtime + dashboard (TORCH_BACKEND build-arg) plus the backend-agnostic
        GLiNER sidecar. ColPali, Whisper, and the LateOn/DenseOn text embedders
        are now served by vLLM (vllm/vllm-openai-cpu) and pulled directly by
        k3d, so no local build is needed for them."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        # Pin backend so the test doesn't depend on the host's GPU detection.
        tags = build_images(Path("/fake/root"), torch_backend="cpu")

        assert tags == [
            "cogniverse/runtime-cpu:dev",
            "cogniverse/dashboard-cpu:dev",
            "cogniverse/gliner:dev",
        ]
        assert mock_run.call_count == 3  # type: ignore[attr-defined]

        for call in mock_run.call_args_list:  # type: ignore[attr-defined]
            cmd = call[0][0]
            assert cmd[0] == "docker"
            assert cmd[1] == "build"

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_runtime_passes_torch_backend_arg(
        self, mock_run: object
    ) -> None:
        """Runtime + dashboard builds get the matching --build-arg
        TORCH_BACKEND=<name> so the Dockerfile picks the right wheel."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        build_images(Path("/fake/root"), torch_backend="rocm")

        runtime_cmd = mock_run.call_args_list[0][0][0]  # type: ignore[attr-defined]
        dashboard_cmd = mock_run.call_args_list[1][0][0]  # type: ignore[attr-defined]
        assert "--build-arg" in runtime_cmd
        assert "TORCH_BACKEND=rocm" in runtime_cmd
        assert "cogniverse/runtime-rocm:dev" in runtime_cmd
        assert "TORCH_BACKEND=rocm" in dashboard_cmd
        assert "cogniverse/dashboard-rocm:dev" in dashboard_cmd

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_builds_gliner_not_pylate(self, mock_run: object) -> None:
        """GLiNER (pullPolicy: Never in the chart) MUST be built+imported by
        ``up`` or its pod ErrImageNeverPulls on a fresh deploy. The retired
        pylate sidecar must never be built again. GLiNER takes no
        TORCH_BACKEND arg and builds from its own ``deploy/gliner`` context."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        built = build_images(Path("/fake/root"), torch_backend="cpu")

        assert built == [
            "cogniverse/runtime-cpu:dev",
            "cogniverse/dashboard-cpu:dev",
            "cogniverse/gliner:dev",
        ]
        assert mock_run.call_count == 3  # type: ignore[attr-defined]
        all_cmds = [
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
        ]
        # GLiNER is built from deploy/gliner/Dockerfile with NO TORCH_BACKEND.
        gliner_cmd = next(c for c in all_cmds if "cogniverse/gliner:dev" in c)
        assert "deploy/gliner/Dockerfile" in gliner_cmd
        assert "deploy/gliner" in gliner_cmd
        assert not any(a.startswith("TORCH_BACKEND=") for a in gliner_cmd)
        for cmd in all_cmds:
            assert "deploy/pylate/Dockerfile" not in cmd
            assert "cogniverse/pylate:dev" not in cmd


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

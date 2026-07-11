"""Unit tests for cogniverse_cli.images build and import utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import yaml
from cogniverse_cli.images import (
    build_images,
    enabled_sidecars,
    has_workspace_source,
    import_images,
    read_app_version,
)


def _make_project_root(
    tmp_path: Path,
    *,
    app_version: str = "2.0.0",
    videoprism: bool = False,
    clap_embed: bool = False,
    face_embed: bool = False,
) -> Path:
    """A project root with just the chart files build_images reads: Chart.yaml
    (appVersion → tag) and values.yaml (inference.<svc>.enabled → build set)."""
    chart_dir = tmp_path / "charts" / "cogniverse"
    chart_dir.mkdir(parents=True)
    (chart_dir / "Chart.yaml").write_text(
        f'version: {app_version}\nappVersion: "{app_version}"\n'
    )
    values = {
        "inference": {
            "videoprism_jax": {"enabled": videoprism},
            "clap_embed": {"enabled": clap_embed},
            "face_embed": {"enabled": face_embed},
        }
    }
    (chart_dir / "values.yaml").write_text(yaml.safe_dump(values))
    return tmp_path


def _completed(mock_run: object) -> None:
    mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
        args=[], returncode=0
    )


class TestHasWorkspaceSource:
    """Tests for :func:`has_workspace_source`."""

    def test_has_workspace_source_true(self, tmp_path: Path) -> None:
        """Returns True when libs/runtime directory exists."""
        (tmp_path / "libs" / "runtime").mkdir(parents=True)

        assert has_workspace_source(tmp_path) is True

    def test_has_workspace_source_false(self, tmp_path: Path) -> None:
        """Returns False when libs/runtime directory is missing."""
        assert has_workspace_source(tmp_path) is False


class TestReadAppVersion:
    """The chart appVersion is the single source of truth for image tags."""

    def test_reads_app_version_from_chart(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path, app_version="3.1.4")
        assert read_app_version(root) == "3.1.4"


class TestBuildImages:
    """Tests for :func:`build_images`."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_calls_docker_build(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """The default build (no sidecars enabled) is exactly three images:
        backend-specific runtime + dashboard plus the backend-agnostic GLiNER
        sidecar, all tagged ``<appVersion>-dev``. ColPali/Whisper/LateOn/DenseOn
        are served by vLLM and pulled by k3d, so no local build is needed."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        tags = build_images(root, torch_backend="cpu")

        assert tags == [
            "cogniverse/runtime-cpu:2.0.0-dev",
            "cogniverse/dashboard-cpu:2.0.0-dev",
            "cogniverse/gliner:2.0.0-dev",
        ]
        assert mock_run.call_count == 3  # type: ignore[attr-defined]
        for call in mock_run.call_args_list:  # type: ignore[attr-defined]
            cmd = call[0][0]
            assert cmd[0] == "docker"
            assert cmd[1] == "build"

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_runtime_passes_torch_backend_arg(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """Runtime + dashboard builds get the matching --build-arg
        TORCH_BACKEND=<name> so the Dockerfile picks the right wheel, and the
        tag carries the backend + the versioned -dev suffix."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        build_images(root, torch_backend="rocm")

        runtime_cmd = mock_run.call_args_list[0][0][0]  # type: ignore[attr-defined]
        dashboard_cmd = mock_run.call_args_list[1][0][0]  # type: ignore[attr-defined]
        assert "--build-arg" in runtime_cmd
        assert "TORCH_BACKEND=rocm" in runtime_cmd
        assert "cogniverse/runtime-rocm:2.0.0-dev" in runtime_cmd
        assert "TORCH_BACKEND=rocm" in dashboard_cmd
        assert "cogniverse/dashboard-rocm:2.0.0-dev" in dashboard_cmd

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_builds_gliner_not_pylate(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """GLiNER (pullPolicy: Never in the chart) MUST be built+imported by
        ``up`` or its pod ErrImageNeverPulls on a fresh deploy. The retired
        pylate sidecar must never be built again. GLiNER takes no TORCH_BACKEND
        arg and builds from its own ``deploy/gliner`` context."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        built = build_images(root, torch_backend="cpu")

        assert built == [
            "cogniverse/runtime-cpu:2.0.0-dev",
            "cogniverse/dashboard-cpu:2.0.0-dev",
            "cogniverse/gliner:2.0.0-dev",
        ]
        all_cmds = [
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
        ]
        gliner_cmd = next(c for c in all_cmds if "cogniverse/gliner:2.0.0-dev" in c)
        assert "deploy/gliner/Dockerfile" in gliner_cmd
        assert "deploy/gliner" in gliner_cmd
        assert not any(a.startswith("TORCH_BACKEND=") for a in gliner_cmd)
        for cmd in all_cmds:
            assert "deploy/pylate/Dockerfile" not in cmd
            assert "cogniverse/pylate" not in " ".join(cmd)

    @patch("cogniverse_cli.images.subprocess.run")
    def test_disabled_sidecars_are_not_built(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """With every optional sidecar disabled, the build set is only the core
        three — a default ``up`` stays fast."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        built = build_images(root, torch_backend="cpu")

        joined = " ".join(" ".join(c[0][0]) for c in mock_run.call_args_list)  # type: ignore[attr-defined]
        assert "cogniverse/face-embed" not in joined
        assert "cogniverse/clap-embed" not in joined
        assert "cogniverse/videoprism" not in joined
        assert len(built) == 3

    @patch("cogniverse_cli.images.subprocess.run")
    def test_overlay_enabling_face_embed_adds_its_build(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """Flipping face_embed on in a deploy overlay makes build_images add its
        image — proving 'enabled: true just works'. face-embed COPYs from libs/
        and deploy/, so its build context is the repo root, and it takes no
        TORCH_BACKEND arg."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)  # base: all sidecars disabled
        overlay = tmp_path / "values.dev.yaml"
        overlay.write_text(
            yaml.safe_dump({"inference": {"face_embed": {"enabled": True}}})
        )

        built = build_images(root, torch_backend="cpu", values_files=[overlay])

        assert built == [
            "cogniverse/runtime-cpu:2.0.0-dev",
            "cogniverse/dashboard-cpu:2.0.0-dev",
            "cogniverse/gliner:2.0.0-dev",
            "cogniverse/face-embed:2.0.0-dev",
        ]
        face_cmd = next(
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if "cogniverse/face-embed:2.0.0-dev" in call[0][0]
        )
        assert "deploy/face_embed/Dockerfile" in face_cmd
        assert face_cmd[-1] == "."  # repo-root context
        assert not any(a.startswith("TORCH_BACKEND=") for a in face_cmd)


class TestEnabledSidecars:
    """Tests for :func:`enabled_sidecars` — the merge that gates sidecar builds."""

    def test_none_enabled_by_default(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path)
        assert enabled_sidecars(root, None) == []

    def test_enabled_in_base_values(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path, face_embed=True)
        assert enabled_sidecars(root, None) == ["face_embed"]

    def test_overlay_merges_over_base_in_sidecar_order(self, tmp_path: Path) -> None:
        """Overlays deep-merge over the chart defaults; the result is returned in
        SIDECAR_BUILDS order regardless of overlay key order."""
        root = _make_project_root(tmp_path)
        overlay = tmp_path / "o.yaml"
        overlay.write_text(
            yaml.safe_dump(
                {
                    "inference": {
                        "clap_embed": {"enabled": True},
                        "videoprism_jax": {"enabled": True},
                    }
                }
            )
        )
        assert enabled_sidecars(root, [overlay]) == ["videoprism_jax", "clap_embed"]


class TestImportImages:
    """Tests for :func:`import_images`."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_import_images_calls_k3d_import(self, mock_run: object) -> None:
        """The k3d image import command includes all tags and cluster name."""
        _completed(mock_run)

        import_images("cogniverse", ["img:a", "img:b"])

        call_args = mock_run.call_args  # type: ignore[attr-defined]
        cmd = call_args[0][0]
        assert cmd[:3] == ["k3d", "image", "import"]
        assert "img:a" in cmd
        assert "img:b" in cmd
        assert "-c" in cmd
        assert "cogniverse" in cmd

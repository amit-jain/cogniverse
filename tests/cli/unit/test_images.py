"""Unit tests for cogniverse_cli.images build and import utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import yaml
from cogniverse_cli.images import (
    build_images,
    dev_image_set_values,
    enabled_sidecars,
    has_workspace_source,
    import_images,
    read_app_version,
)

# A setuptools-scm-style git version and its docker-tag sanitization (+ -> -).
# Passed explicitly so the tests don't need a real git checkout.
DEV_VERSION = "0.1.dev5+gabc1234"
DEV_TAG = "0.1.dev5-gabc1234"


def _make_project_root(
    tmp_path: Path,
    *,
    app_version: str = "0.1.0",
    videoprism: bool = False,
    clap_embed: bool = False,
    face_embed: bool = False,
) -> Path:
    """A project root with just the chart files images.py reads: Chart.yaml
    (appVersion) and values.yaml (inference.<svc>.enabled → build set)."""
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
    """Chart appVersion is the static release line (release image tags)."""

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
        sidecar, all tagged with the commit-unique git version (``+`` sanitized
        to ``-``). ColPali/Whisper/LateOn/DenseOn are served by vLLM."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        tags = build_images(root, torch_backend="cpu", version=DEV_VERSION)

        assert tags == [
            f"cogniverse/runtime-cpu:{DEV_TAG}",
            f"cogniverse/dashboard-cpu:{DEV_TAG}",
            f"cogniverse/gliner:{DEV_TAG}",
        ]
        assert mock_run.call_count == 3  # type: ignore[attr-defined]
        for call in mock_run.call_args_list:  # type: ignore[attr-defined]
            cmd = call[0][0]
            assert cmd[0] == "docker"
            assert cmd[1] == "build"

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_runtime_passes_torch_backend_and_version(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """Runtime + dashboard builds get the matching --build-arg
        TORCH_BACKEND=<name>, a tag carrying the git version, and the FULL git
        version fed to hatch-vcs inside the git-less docker context via
        SETUPTOOLS_SCM_PRETEND_VERSION (the tag sanitizes ``+``, the build-arg
        keeps it)."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        build_images(root, torch_backend="rocm", version=DEV_VERSION)

        runtime_cmd = mock_run.call_args_list[0][0][0]  # type: ignore[attr-defined]
        dashboard_cmd = mock_run.call_args_list[1][0][0]  # type: ignore[attr-defined]
        gliner_cmd = mock_run.call_args_list[2][0][0]  # type: ignore[attr-defined]
        assert "TORCH_BACKEND=rocm" in runtime_cmd
        assert f"cogniverse/runtime-rocm:{DEV_TAG}" in runtime_cmd
        assert f"SETUPTOOLS_SCM_PRETEND_VERSION={DEV_VERSION}" in runtime_cmd
        assert "TORCH_BACKEND=rocm" in dashboard_cmd
        assert f"cogniverse/dashboard-rocm:{DEV_TAG}" in dashboard_cmd
        assert f"SETUPTOOLS_SCM_PRETEND_VERSION={DEV_VERSION}" in dashboard_cmd
        # GLiNER + sidecars don't install the workspace, so no scm arg.
        assert not any("SETUPTOOLS_SCM_PRETEND_VERSION" in a for a in gliner_cmd)

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

        built = build_images(root, torch_backend="cpu", version=DEV_VERSION)

        assert built == [
            f"cogniverse/runtime-cpu:{DEV_TAG}",
            f"cogniverse/dashboard-cpu:{DEV_TAG}",
            f"cogniverse/gliner:{DEV_TAG}",
        ]
        all_cmds = [
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
        ]
        gliner_cmd = next(c for c in all_cmds if f"cogniverse/gliner:{DEV_TAG}" in c)
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

        built = build_images(root, torch_backend="cpu", version=DEV_VERSION)

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

        built = build_images(
            root, torch_backend="cpu", values_files=[overlay], version=DEV_VERSION
        )

        assert built == [
            f"cogniverse/runtime-cpu:{DEV_TAG}",
            f"cogniverse/dashboard-cpu:{DEV_TAG}",
            f"cogniverse/gliner:{DEV_TAG}",
            f"cogniverse/face-embed:{DEV_TAG}",
        ]
        face_cmd = next(
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if f"cogniverse/face-embed:{DEV_TAG}" in call[0][0]
        )
        assert "deploy/face_embed/Dockerfile" in face_cmd
        assert face_cmd[-1] == "."  # repo-root context
        assert not any(a.startswith("TORCH_BACKEND=") for a in face_cmd)


class TestDevImageSetValues:
    """The chart --set overrides that point first-party images at the built tag."""

    def test_maps_core_images_to_the_git_tag(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path)
        overrides = dev_image_set_values(root, torch_backend="cpu", version=DEV_VERSION)
        assert overrides == {
            "runtime.imagesByBackend.cpu.tag": DEV_TAG,
            "dashboard.imagesByBackend.cpu.tag": DEV_TAG,
            "inference.gliner.image.tag": DEV_TAG,
        }

    def test_backend_scopes_runtime_and_dashboard(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path)
        overrides = dev_image_set_values(
            root, torch_backend="rocm", version=DEV_VERSION
        )
        assert "runtime.imagesByBackend.rocm.tag" in overrides
        assert "runtime.imagesByBackend.cpu.tag" not in overrides

    def test_includes_enabled_sidecars_only(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path, face_embed=True)
        overrides = dev_image_set_values(root, torch_backend="cpu", version=DEV_VERSION)
        assert overrides["inference.face_embed.image.tag"] == DEV_TAG
        assert "inference.clap_embed.image.tag" not in overrides


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


class TestPruneSupersededImages:
    """After a deploy, image generations older than current + one previous
    are removed on the host and inside the k3d node — each `cogniverse up`
    otherwise leaves ~25GB of superseded tags behind. Node removal goes by
    image ID and only for IDs whose every tag is superseded: crictl rmi
    drops all of an ID's tags at once, and e.g. the gliner image shares one
    ID across every generation."""

    HOST_LISTING = "\n".join(
        [
            "cogniverse/runtime-rocm:0.1.dev2420-g813e8e5c8\taaa1",
            "cogniverse/runtime-rocm:0.1.dev2418-g999492e27\taaa2",
            "cogniverse/runtime-rocm:0.1.dev2397-g0f2366466\taaa3",
            "cogniverse/dashboard-rocm:0.1.dev2420-g813e8e5c8\tbbb1",
            "cogniverse/dashboard-rocm:0.1.dev2397-g0f2366466\tbbb3",
            "cogniverse/gliner:0.1.dev2420-g813e8e5c8\tccc1",
            "vespaengine/vespa:8.668.5\tddd1",
        ]
    )

    NODE_JSON = json.dumps(
        {
            "images": [
                {
                    "id": "sha-runtime-new",
                    "repoTags": [
                        "docker.io/cogniverse/runtime-rocm:0.1.dev2420-g813e8e5c8"
                    ],
                },
                {
                    "id": "sha-runtime-old",
                    "repoTags": [
                        "docker.io/cogniverse/runtime-rocm:0.1.dev2397-g0f2366466"
                    ],
                },
                {
                    "id": "sha-gliner-shared",
                    "repoTags": [
                        "docker.io/cogniverse/gliner:0.1.dev2397-g0f2366466",
                        "docker.io/cogniverse/gliner:0.1.dev2420-g813e8e5c8",
                    ],
                },
                {
                    "id": "sha-vespa",
                    "repoTags": ["docker.io/vespaengine/vespa:8.668.5"],
                },
            ]
        }
    )

    def _runner(self, calls):
        host_listing = self.HOST_LISTING
        node_json = self.NODE_JSON

        def run(cmd, **kwargs):
            calls.append(cmd)
            out = ""
            if cmd[:2] == ["docker", "images"]:
                out = host_listing
            elif "crictl" in cmd and "images" in cmd:
                out = node_json
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

        return run

    def test_removes_only_generations_older_than_current_plus_one(self):
        from cogniverse_cli.images import prune_superseded_images

        calls: list = []
        removed = prune_superseded_images(
            "0.1.dev2420+g813e8e5c8", runner=self._runner(calls)
        )

        rmi_cmds = [c for c in calls if c[:2] == ["docker", "rmi"]]
        removed_tags = {tag for c in rmi_cmds for tag in c[2:]}
        assert removed_tags == {
            "cogniverse/runtime-rocm:0.1.dev2397-g0f2366466",
            "cogniverse/dashboard-rocm:0.1.dev2397-g0f2366466",
        }
        assert set(removed) == removed_tags

    def test_node_prune_skips_ids_with_a_kept_tag(self):
        from cogniverse_cli.images import prune_superseded_images

        calls: list = []
        prune_superseded_images(
            "0.1.dev2420+g813e8e5c8",
            node_container="k3d-cogniverse-server-0",
            runner=self._runner(calls),
        )

        crictl_rmi = [c for c in calls if "crictl" in c and "rmi" in c]
        removed_ids = {arg for c in crictl_rmi for arg in c[c.index("rmi") + 1 :]}
        # runtime-old is superseded and uniquely tagged -> removed; the
        # gliner ID carries the CURRENT tag too -> untouchable; vespa is
        # not a cogniverse image.
        assert "sha-runtime-old" in removed_ids
        assert "sha-gliner-shared" not in removed_ids
        assert "sha-vespa" not in removed_ids
        assert "sha-runtime-new" not in removed_ids

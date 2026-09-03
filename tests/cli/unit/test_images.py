"""Unit tests for cogniverse_cli.images build and import utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from unittest.mock import call as mock_call

import cogniverse_cli.images as images_mod
import pytest
import yaml
from cogniverse_cli.images import (
    _read_third_party_images,
    build_images,
    detect_torch_backend,
    dev_image_set_values,
    enabled_sidecars,
    has_workspace_source,
    import_images,
    pull_and_import_third_party,
    read_app_version,
)

# A deploy-input-derived git version and its docker-tag sanitization (+ -> -).
# Passed explicitly so the tests don't need a real git checkout.
DEV_VERSION = "0.1.dev5+gabc1234"
DEV_TAG = "0.1.dev5-gabc1234"


def _make_project_root(
    tmp_path: Path,
    *,
    app_version: str = "0.1.0",
    clap_embed: bool = False,
    face_embed: bool = False,
    colbert_pylate: bool = False,
    code_colbert_pylate: bool = False,
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
            "clap_embed": {"enabled": clap_embed},
            "face_embed": {"enabled": face_embed},
            "colbert_pylate": {"enabled": colbert_pylate},
            "code_colbert_pylate": {"enabled": code_colbert_pylate},
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
        sidecar, all tagged with the deploy-input-derived git version (``+``
        sanitized to ``-``). ColPali/Whisper/LateOn/DenseOn are served by
        vLLM."""
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
        TORCH_BACKEND=<name>, a tag carrying the deploy-input-derived git
        version, and the FULL git version fed into the git-less docker context
        via SETUPTOOLS_SCM_PRETEND_VERSION (the tag sanitizes ``+``, the
        build-arg keeps it)."""
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
    def test_build_images_builds_gliner_without_backend_arg(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """GLiNER (pullPolicy: Never in the chart) MUST be built+imported by
        ``up`` or its pod ErrImageNeverPulls on a fresh deploy. GLiNER takes
        no TORCH_BACKEND arg and builds from the repository root so its
        canonical CLI server is available to the Dockerfile."""
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
        assert gliner_cmd == [
            "docker",
            "build",
            "-f",
            "deploy/gliner/Dockerfile",
            "-t",
            f"cogniverse/gliner:{DEV_TAG}",
            ".",
        ]
        for cmd in all_cmds:
            assert "deploy/pylate/Dockerfile" not in cmd
            assert "cogniverse/pylate" not in " ".join(cmd)

    @patch("cogniverse_cli.images.subprocess.run")
    def test_enabled_lateon_services_build_one_pylate_image_with_backend(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """Both LateOn services share the cogniverse/pylate image, so enabling
        both builds it exactly once, from the repository root (the canonical
        CLI PyLate server is COPY'd in) with the host-matching TORCH_BACKEND.
        Both chart entries still get the deploy-input-derived dev-tag
        override."""
        _completed(mock_run)
        root = _make_project_root(
            tmp_path, colbert_pylate=True, code_colbert_pylate=True
        )

        built = build_images(root, torch_backend="rocm", version=DEV_VERSION)

        assert built == [
            f"cogniverse/runtime-rocm:{DEV_TAG}",
            f"cogniverse/dashboard-rocm:{DEV_TAG}",
            f"cogniverse/gliner:{DEV_TAG}",
            f"cogniverse/pylate:{DEV_TAG}",
        ]
        pylate_cmds = [
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if f"cogniverse/pylate:{DEV_TAG}" in call[0][0]
        ]
        assert pylate_cmds == [
            [
                "docker",
                "build",
                "-f",
                "deploy/pylate/Dockerfile",
                "--build-arg",
                "TORCH_BACKEND=rocm",
                "-t",
                f"cogniverse/pylate:{DEV_TAG}",
                ".",
            ]
        ]

        overrides = dev_image_set_values(
            root, torch_backend="rocm", version=DEV_VERSION
        )
        assert overrides["inference.colbert_pylate.image.tag"] == DEV_TAG
        assert overrides["inference.code_colbert_pylate.image.tag"] == DEV_TAG

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


def test_release_gliner_build_includes_canonical_server() -> None:
    workflow_path = Path(__file__).parents[3] / ".github/workflows/release-images.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    image_matrix = workflow["jobs"]["build-push"]["strategy"]["matrix"]["include"]
    entries = {entry["repo"]: entry for entry in image_matrix}

    assert entries["gliner"] == {
        "repo": "gliner",
        "dockerfile": "deploy/gliner/Dockerfile",
        "context": ".",
        "backend": "",
    }
    assert "videoprism" not in entries


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
                        "face_embed": {"enabled": True},
                        "clap_embed": {"enabled": True},
                    }
                }
            )
        )
        assert enabled_sidecars(root, [overlay]) == ["clap_embed", "face_embed"]

    def test_external_url_excludes_the_sidecar_build(self, tmp_path: Path) -> None:
        """A Modal-hosted service deploys no local pod, so its sidecar image
        must not enter the build set."""
        root = _make_project_root(tmp_path, face_embed=True)
        overlay = tmp_path / "o.yaml"
        overlay.write_text(
            yaml.safe_dump(
                {
                    "inference": {
                        "face_embed": {
                            "externalUrl": (
                                "https://amit--cogniverse-face-embed.modal.run"
                            )
                        }
                    }
                }
            )
        )
        assert enabled_sidecars(root, [overlay]) == []


class TestImportImages:
    """Tests for :func:`import_images`."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_import_images_calls_k3d_import(self, mock_run: object) -> None:
        """Images are imported independently without a memory-heavy tools pod."""
        _completed(mock_run)

        import_images("cogniverse", ["img:a", "img:b"])

        assert mock_run.call_args_list == [  # type: ignore[attr-defined]
            mock_call(
                [
                    "k3d",
                    "image",
                    "import",
                    "--mode",
                    "direct",
                    "img:a",
                    "-c",
                    "cogniverse",
                ],
                check=True,
                timeout=1800,
            ),
            mock_call(
                [
                    "k3d",
                    "image",
                    "import",
                    "--mode",
                    "direct",
                    "img:b",
                    "-c",
                    "cogniverse",
                ],
                check=True,
                timeout=1800,
            ),
        ]

    @patch("cogniverse_cli.images.subprocess.run")
    def test_import_failure_names_image_and_stops_later_imports(
        self, mock_run: object
    ) -> None:
        failed_command = [
            "k3d",
            "image",
            "import",
            "--mode",
            "direct",
            "img:b",
            "-c",
            "cogniverse",
        ]
        mock_run.side_effect = [  # type: ignore[attr-defined]
            subprocess.CompletedProcess(args=[], returncode=0),
            subprocess.CalledProcessError(returncode=1, cmd=failed_command),
        ]

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            import_images("cogniverse", ["img:a", "img:b", "img:c"])

        assert exc_info.value.cmd == failed_command
        assert [call.args[0] for call in mock_run.call_args_list] == [  # type: ignore[attr-defined]
            [
                "k3d",
                "image",
                "import",
                "--mode",
                "direct",
                "img:a",
                "-c",
                "cogniverse",
            ],
            failed_command,
        ]


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


class TestDetectTorchBackend:
    """The backend ladder: env override -> nvidia-smi -> rocminfo(gfx) ->
    /sys/module/amdgpu -> cpu. Each rung is exercised in isolation because
    the dev host itself has real GPU tooling."""

    def _blank_slate(self, monkeypatch) -> MagicMock:
        """No env override, no GPU binaries, no amdgpu module. Returns the
        Path stand-in so a branch can flip ``/sys/module/amdgpu`` on."""
        monkeypatch.delenv("COGNIVERSE_TORCH_BACKEND", raising=False)
        monkeypatch.setattr(images_mod.shutil, "which", lambda name: None)
        fake_path = MagicMock()
        fake_path.return_value.exists.return_value = False
        monkeypatch.setattr(images_mod, "Path", fake_path)
        return fake_path

    def test_env_override_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("COGNIVERSE_TORCH_BACKEND", "rocm")
        assert detect_torch_backend() == "rocm"

    def test_nvidia_smi_success_is_cuda(self, monkeypatch) -> None:
        self._blank_slate(monkeypatch)
        monkeypatch.setattr(
            images_mod.shutil,
            "which",
            lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
        )
        monkeypatch.setattr(
            images_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0),
        )
        assert detect_torch_backend() == "cuda"

    def test_nvidia_smi_failure_falls_through_to_cpu(self, monkeypatch) -> None:
        self._blank_slate(monkeypatch)
        monkeypatch.setattr(
            images_mod.shutil,
            "which",
            lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
        )

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        monkeypatch.setattr(images_mod.subprocess, "run", boom)
        assert detect_torch_backend() == "cpu"

    def test_rocminfo_gfx_agent_is_rocm(self, monkeypatch) -> None:
        self._blank_slate(monkeypatch)
        monkeypatch.setattr(
            images_mod.shutil,
            "which",
            lambda name: "/usr/bin/rocminfo" if name == "rocminfo" else None,
        )
        monkeypatch.setattr(
            images_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0], 0, stdout="Name:      gfx1151\nMarketing Name: AMD\n"
            ),
        )
        assert detect_torch_backend() == "rocm"

    def test_rocminfo_without_gfx_falls_through_to_cpu(self, monkeypatch) -> None:
        self._blank_slate(monkeypatch)
        monkeypatch.setattr(
            images_mod.shutil,
            "which",
            lambda name: "/usr/bin/rocminfo" if name == "rocminfo" else None,
        )
        monkeypatch.setattr(
            images_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="no agents\n"),
        )
        assert detect_torch_backend() == "cpu"

    def test_amdgpu_module_present_is_rocm(self, monkeypatch) -> None:
        fake_path = self._blank_slate(monkeypatch)
        fake_path.return_value.exists.return_value = True
        assert detect_torch_backend() == "rocm"
        fake_path.assert_called_with("/sys/module/amdgpu")

    def test_no_gpu_evidence_is_cpu(self, monkeypatch) -> None:
        self._blank_slate(monkeypatch)
        assert detect_torch_backend() == "cpu"


class TestReadThirdPartyImages:
    """`_read_third_party_images` walks the values file the way the chart
    resolves images: vespa/phoenix, semantic-router, optional llm.builtin,
    then each enabled inference.<svc> including imagesByDevice; pullPolicy
    Never and enabled:false are skipped."""

    def _values_file(self, tmp_path: Path) -> Path:
        data = {
            "vespa": {"image": {"repository": "vespaengine/vespa", "tag": "8.1"}},
            "phoenix": {"image": {"repository": "arizephoenix/phoenix", "tag": "5.0"}},
            "llm": {
                "builtin": {
                    "image": {"repository": "vllm/vllm-openai-cpu", "tag": "0.6"}
                }
            },
            "semanticRouter": {"enabled": False},
            "inference": {
                # Locally-built image (pullPolicy Never) -> never pulled.
                "gliner": {
                    "enabled": True,
                    "image": {
                        "repository": "cogniverse/gliner",
                        "tag": "dev",
                        "pullPolicy": "Never",
                    },
                },
                # Device-specific image AND the base image are both pre-pulled.
                "clap_embed": {
                    "enabled": True,
                    "device": "rocm",
                    "imagesByDevice": {
                        "rocm": {"repository": "cogniverse/clap-rocm", "tag": "r1"},
                        "cpu": {"repository": "cogniverse/clap-cpu", "tag": "c1"},
                    },
                    "image": {"repository": "cogniverse/clap", "tag": "base"},
                },
                # Disabled -> skipped entirely.
                "face_embed": {
                    "enabled": False,
                    "image": {"repository": "cogniverse/face", "tag": "x"},
                },
            },
        }
        vf = tmp_path / "values.yaml"
        vf.write_text(yaml.safe_dump(data))
        return vf

    def test_resolves_core_device_and_skips_never_and_disabled(
        self, tmp_path: Path
    ) -> None:
        result = _read_third_party_images(self._values_file(tmp_path), skip_llm=False)
        assert result == [
            "vespaengine/vespa:8.1",
            "arizephoenix/phoenix:5.0",
            "vllm/vllm-openai-cpu:0.6",
            "cogniverse/clap-rocm:r1",
            "cogniverse/clap:base",
        ]
        # pullPolicy Never (gliner) and enabled:false (face_embed) never appear.
        assert "cogniverse/gliner:dev" not in result
        assert "cogniverse/face:x" not in result
        # The non-selected device variant (cpu) is not pulled.
        assert "cogniverse/clap-cpu:c1" not in result

    def test_skip_llm_omits_builtin_llm_image(self, tmp_path: Path) -> None:
        result = _read_third_party_images(self._values_file(tmp_path), skip_llm=True)
        assert result == [
            "vespaengine/vespa:8.1",
            "arizephoenix/phoenix:5.0",
            "cogniverse/clap-rocm:r1",
            "cogniverse/clap:base",
        ]

    def test_semantic_router_images_included_when_enabled(self, tmp_path: Path) -> None:
        vf = tmp_path / "sr.yaml"
        vf.write_text(
            yaml.safe_dump(
                {
                    "semanticRouter": {
                        "enabled": True,
                        "envoy": {
                            "image": {"repository": "envoyproxy/envoy", "tag": "1.29"}
                        },
                        "router": {
                            "image": {"repository": "cogniverse/sr", "tag": "2.0"}
                        },
                    }
                }
            )
        )
        assert _read_third_party_images(vf, skip_llm=True) == [
            "envoyproxy/envoy:1.29",
            "cogniverse/sr:2.0",
        ]

    def test_duplicate_images_are_deduplicated_first_wins(self, tmp_path: Path) -> None:
        vf = tmp_path / "dup.yaml"
        vf.write_text(
            yaml.safe_dump(
                {
                    "vespa": {"image": {"repository": "shared/img", "tag": "1"}},
                    "phoenix": {"image": {"repository": "shared/img", "tag": "1"}},
                    "semanticRouter": {"enabled": False},
                }
            )
        )
        assert _read_third_party_images(vf, skip_llm=True) == ["shared/img:1"]

    def test_missing_tag_defaults_to_latest(self, tmp_path: Path) -> None:
        vf = tmp_path / "notag.yaml"
        vf.write_text(
            yaml.safe_dump(
                {
                    "vespa": {"image": {"repository": "vespaengine/vespa"}},
                    "semanticRouter": {"enabled": False},
                }
            )
        )
        assert _read_third_party_images(vf, skip_llm=True) == [
            "vespaengine/vespa:latest"
        ]


class TestPullAndImportThirdParty:
    """`pull_and_import_third_party` docker-pulls each resolved image then
    imports them independently into k3d."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_pulls_and_imports_each_image_independently(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )
        vf = tmp_path / "values.yaml"
        vf.write_text(
            yaml.safe_dump(
                {
                    "vespa": {
                        "image": {"repository": "vespaengine/vespa", "tag": "8.1"}
                    },
                    "phoenix": {
                        "image": {"repository": "arizephoenix/phoenix", "tag": "5.0"}
                    },
                    "semanticRouter": {"enabled": False},
                }
            )
        )

        pull_and_import_third_party("cogniverse", vf, skip_llm=True)

        calls = [c.args[0] for c in mock_run.call_args_list]  # type: ignore[attr-defined]
        assert calls == [
            ["docker", "pull", "vespaengine/vespa:8.1"],
            ["docker", "pull", "arizephoenix/phoenix:5.0"],
            [
                "k3d",
                "image",
                "import",
                "--mode",
                "direct",
                "vespaengine/vespa:8.1",
                "-c",
                "cogniverse",
            ],
            [
                "k3d",
                "image",
                "import",
                "--mode",
                "direct",
                "arizephoenix/phoenix:5.0",
                "-c",
                "cogniverse",
            ],
        ]
        for call in mock_run.call_args_list:  # type: ignore[attr-defined]
            assert call.kwargs["check"] is True

    @patch("cogniverse_cli.images.subprocess.run")
    def test_pull_failure_names_image_and_stops_before_later_work(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        vf = tmp_path / "values.yaml"
        vf.write_text(
            yaml.safe_dump(
                {
                    "vespa": {"image": {"repository": "example/failing", "tag": "1"}},
                    "phoenix": {"image": {"repository": "example/later", "tag": "2"}},
                    "semanticRouter": {"enabled": False},
                }
            )
        )
        failed_command = ["docker", "pull", "example/failing:1"]
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=failed_command
        )  # type: ignore[attr-defined]

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            pull_and_import_third_party("cogniverse", vf, skip_llm=True)

        assert exc_info.value.cmd == failed_command
        assert [call.args[0] for call in mock_run.call_args_list] == [failed_command]  # type: ignore[attr-defined]

    @patch("cogniverse_cli.images.subprocess.run")
    def test_import_failure_names_image_and_stops_later_imports(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        vf = tmp_path / "values.yaml"
        vf.write_text(
            yaml.safe_dump(
                {
                    "vespa": {"image": {"repository": "example/first", "tag": "1"}},
                    "phoenix": {"image": {"repository": "example/failing", "tag": "2"}},
                    "semanticRouter": {
                        "envoy": {"image": {"repository": "example/later", "tag": "3"}},
                        "router": {"image": {}},
                    },
                }
            )
        )
        failed_command = [
            "k3d",
            "image",
            "import",
            "--mode",
            "direct",
            "example/failing:2",
            "-c",
            "cogniverse",
        ]
        mock_run.side_effect = [  # type: ignore[attr-defined]
            subprocess.CompletedProcess(args=[], returncode=0),
            subprocess.CompletedProcess(args=[], returncode=0),
            subprocess.CompletedProcess(args=[], returncode=0),
            subprocess.CompletedProcess(args=[], returncode=0),
            subprocess.CalledProcessError(returncode=1, cmd=failed_command),
        ]

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            pull_and_import_third_party("cogniverse", vf, skip_llm=True)

        assert exc_info.value.cmd == failed_command
        import_commands = [
            call.args[0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if call.args[0][:3] == ["k3d", "image", "import"]
        ]
        assert import_commands == [
            [
                "k3d",
                "image",
                "import",
                "--mode",
                "direct",
                "example/first:1",
                "-c",
                "cogniverse",
            ],
            failed_command,
        ]

    @patch("cogniverse_cli.images.subprocess.run")
    def test_no_images_pulls_nothing(self, mock_run: object, tmp_path: Path) -> None:
        vf = tmp_path / "empty.yaml"
        vf.write_text(yaml.safe_dump({"semanticRouter": {"enabled": False}}))

        pull_and_import_third_party("cogniverse", vf, skip_llm=True)

        mock_run.assert_not_called()  # type: ignore[attr-defined]


class TestDeviceOverlaySidecarTags:
    """Tag overrides are emitted per ENABLED sidecar, so whoever deploys must
    compute them from the same overlays it hands helm."""

    REPO_ROOT = Path(__file__).resolve().parents[3]

    def _chart(self, name: str) -> Path:
        path = self.REPO_ROOT / "charts" / "cogniverse" / name
        assert path.exists(), f"missing chart values file: {path}"
        return path

    def test_rocm_overlay_enables_a_sidecar_the_defaults_do_not(self) -> None:
        """The real chart layering the deploy relies on: the second LateOn
        service exists only once the device overlay is merged in."""
        defaults = enabled_sidecars(self.REPO_ROOT, None)
        with_rocm = enabled_sidecars(self.REPO_ROOT, [self._chart("values.rocm.yaml")])

        assert "code_colbert_pylate" not in defaults
        assert "colbert_pylate" in defaults
        assert with_rocm == ["colbert_pylate", "code_colbert_pylate"]

    def test_overrides_cover_every_sidecar_the_overlay_enables(self) -> None:
        overrides = dev_image_set_values(
            self.REPO_ROOT,
            torch_backend="rocm",
            values_files=[self._chart("values.rocm.yaml")],
            version=DEV_VERSION,
        )

        assert overrides["inference.colbert_pylate.image.tag"] == DEV_TAG
        assert overrides["inference.code_colbert_pylate.image.tag"] == DEV_TAG

    def test_omitting_the_overlay_leaves_its_sidecar_on_the_placeholder_tag(
        self,
    ) -> None:
        """Computing overrides from chart defaults while helm applies the
        overlay is the trap: the overlay-only service keeps the chart's static
        placeholder tag, which no build ever produces, so the pod cannot pull
        it under pullPolicy=Never.
        """
        overrides = dev_image_set_values(
            self.REPO_ROOT, torch_backend="rocm", version=DEV_VERSION
        )

        assert "inference.colbert_pylate.image.tag" in overrides
        assert "inference.code_colbert_pylate.image.tag" not in overrides


class TestFirstPartyImageCoverage:
    """A first-party image (a ``cogniverse/*`` repository) exists in no
    registry, so a chart-enabled service that the build never produced leaves
    its pod stuck on ErrImageNeverPull. These tests derive the required set
    from the chart the deploy actually renders, so enabling ANY future sidecar
    in values is covered without editing a list here.
    """

    REPO_ROOT = Path(__file__).resolve().parents[3]

    def _chart(self, name: str) -> Path:
        path = self.REPO_ROOT / "charts" / "cogniverse" / name
        assert path.exists(), f"missing chart values file: {path}"
        return path

    @pytest.mark.parametrize(
        "overlays",
        [
            (),
            ("values.k3s.yaml",),
            ("values.k3s.yaml", "values.rocm.yaml"),
            ("values.k3s.yaml", "values.cpu.yaml"),
            ("values.k3s.yaml", "values.cuda.yaml"),
        ],
    )
    def test_every_chart_enabled_first_party_service_is_buildable(
        self, overlays: tuple[str, ...]
    ) -> None:
        """Every enabled ``cogniverse/*`` inference service the deploy renders
        must have a build spec whose Dockerfile exists on disk."""
        from cogniverse_cli.images import LOCAL_IMAGE_BUILDS, first_party_services

        required = first_party_services(
            self.REPO_ROOT, [self._chart(name) for name in overlays]
        )

        missing = sorted(set(required) - set(LOCAL_IMAGE_BUILDS))
        assert missing == [], (
            f"chart enables first-party images with no build spec: {missing}"
        )
        for svc, repo in required.items():
            spec_repo, dockerfile, _ = LOCAL_IMAGE_BUILDS[svc]
            assert spec_repo == repo, f"{svc}: chart repo {repo} != build {spec_repo}"
            assert (self.REPO_ROOT / dockerfile).exists(), f"{svc}: {dockerfile}"

    def test_first_party_services_ignores_registry_backed_services(self) -> None:
        """vLLM-served services are pulled from a registry, never built."""
        from cogniverse_cli.images import first_party_services

        required = first_party_services(
            self.REPO_ROOT, [self._chart("values.k3s.yaml")]
        )

        assert "denseon" not in required
        assert "vllm_asr" not in required

    def test_external_url_excludes_the_first_party_image(self, tmp_path: Path) -> None:
        """A Modal-hosted first-party service renders no pod, so its image is
        not required for the deploy."""
        from cogniverse_cli.images import first_party_services

        root = _make_project_root(tmp_path, face_embed=True)
        assert first_party_services(root, None) == {}

        chart_dir = root / "charts" / "cogniverse"
        values = yaml.safe_load((chart_dir / "values.yaml").read_text())
        values["inference"]["face_embed"]["image"] = {
            "repository": "cogniverse/face-embed",
            "tag": "0.1.0",
        }
        (chart_dir / "values.yaml").write_text(yaml.safe_dump(values))
        assert first_party_services(root, None) == {
            "face_embed": "cogniverse/face-embed"
        }

        overlay = tmp_path / "o.yaml"
        overlay.write_text(
            yaml.safe_dump(
                {
                    "inference": {
                        "face_embed": {
                            "externalUrl": (
                                "https://amit--cogniverse-face-embed.modal.run"
                            )
                        }
                    }
                }
            )
        )
        assert first_party_services(root, [overlay]) == {}

    @patch("subprocess.run")
    def test_verification_fails_when_build_skipped_a_deploy_overlay(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """The reported failure, stated generically: images built from chart
        defaults do not satisfy a deploy whose overlay enables another
        first-party service. Verification must raise and name it."""
        from cogniverse_cli.images import (
            build_images,
            verify_local_images_cover_deploy,
        )

        _completed(mock_run)
        root = _make_project_root(tmp_path)
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(
            yaml.safe_dump(
                {
                    "inference": {
                        "future_embed": {
                            "enabled": True,
                            "image": {
                                "repository": "cogniverse/future-embed",
                                "pullPolicy": "Never",
                            },
                        }
                    }
                }
            )
        )

        built = build_images(root, torch_backend="cpu", version=DEV_VERSION)

        with pytest.raises(RuntimeError) as excinfo:
            verify_local_images_cover_deploy(
                root, [overlay], built_tags=built, version=DEV_VERSION
            )
        assert "future_embed" in str(excinfo.value)
        assert "cogniverse/future-embed" in str(excinfo.value)

    @patch("subprocess.run")
    def test_verification_passes_when_build_used_the_deploy_overlays(
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same overlay, built with the overlay: nothing is missing."""
        from cogniverse_cli.images import (
            LOCAL_IMAGE_BUILDS,
            build_images,
            verify_local_images_cover_deploy,
        )

        _completed(mock_run)
        monkeypatch.setitem(
            LOCAL_IMAGE_BUILDS,
            "future_embed",
            ("cogniverse/future-embed", "deploy/gliner/Dockerfile", "."),
        )
        root = _make_project_root(tmp_path)
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(
            yaml.safe_dump(
                {
                    "inference": {
                        "future_embed": {
                            "enabled": True,
                            "image": {
                                "repository": "cogniverse/future-embed",
                                "pullPolicy": "Never",
                            },
                        }
                    }
                }
            )
        )

        built = build_images(
            root, torch_backend="cpu", values_files=[overlay], version=DEV_VERSION
        )

        assert f"cogniverse/future-embed:{DEV_TAG}" in built
        verify_local_images_cover_deploy(
            root, [overlay], built_tags=built, version=DEV_VERSION
        )

    @patch("subprocess.run")
    def test_build_raises_for_an_enabled_service_with_no_build_spec(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Enabling a first-party service nobody taught the builder about is a
        drift error at build time, not an ErrImageNeverPull at deploy time."""
        from cogniverse_cli.images import build_images

        _completed(mock_run)
        root = _make_project_root(tmp_path)
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(
            yaml.safe_dump(
                {
                    "inference": {
                        "unknown_embed": {
                            "enabled": True,
                            "image": {"repository": "cogniverse/unknown-embed"},
                        }
                    }
                }
            )
        )

        with pytest.raises(RuntimeError, match="unknown_embed"):
            build_images(
                root, torch_backend="cpu", values_files=[overlay], version=DEV_VERSION
            )

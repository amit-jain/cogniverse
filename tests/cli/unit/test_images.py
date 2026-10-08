"""Unit tests for cogniverse_cli.images build and import utilities."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    dev_version,
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
DEV_VERSIONS = {
    "runtime": "0.1.dev11+gaaa111aaa",
    "dashboard": "0.1.dev12+gbbb222bbb",
    "web": "0.1.dev19+giii999iii",
    "pylate": "0.1.dev13+gccc333ccc",
    "gliner": "0.1.dev14+gddd444ddd",
    "clap_embed": "0.1.dev15+geee555eee",
    "face_embed": "0.1.dev16+gfff666fff",
    "video_embed": "0.1.dev17+gggg777ggg",
    "vllm_audio": "0.1.dev18+ghhh888hhh",
}
DEV_TAGS = {image: version.replace("+", "-") for image, version in DEV_VERSIONS.items()}
UNIFORM_DEV_VERSIONS = dict.fromkeys(DEV_VERSIONS, DEV_VERSION)


def _make_project_root(
    tmp_path: Path,
    *,
    app_version: str = "0.1.0",
    clap_embed: bool = False,
    face_embed: bool = False,
    colbert_pylate: bool = False,
    code_colbert_pylate: bool = False,
    web: bool = True,
    dashboard: bool = False,
) -> Path:
    """A project root with just the chart files images.py reads: Chart.yaml
    (appVersion) and values.yaml (web/dashboard and inference.<svc> enabled
    flags → build set). The UI defaults match the shipped chart."""
    chart_dir = tmp_path / "charts" / "cogniverse"
    chart_dir.mkdir(parents=True)
    (chart_dir / "Chart.yaml").write_text(
        f'version: {app_version}\nappVersion: "{app_version}"\n'
    )
    values = {
        "web": {"enabled": web},
        "dashboard": {"enabled": dashboard},
        "inference": {
            "clap_embed": {"enabled": clap_embed},
            "face_embed": {"enabled": face_embed},
            "colbert_pylate": {"enabled": colbert_pylate},
            "code_colbert_pylate": {"enabled": code_colbert_pylate},
        },
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


class TestPerImageDevVersion:
    """Each image tag follows only the repository inputs copied into it."""

    ALL_IMAGE_FAMILIES = {
        "runtime",
        "dashboard",
        "web",
        "pylate",
        "gliner",
        "clap_embed",
        "face_embed",
        "video_embed",
        "vllm_audio",
    }
    INPUT_CASES = [
        (
            "libs/runtime/Dockerfile",
            "libs/runtime/Dockerfile",
            "FROM scratch\n",
            {"runtime"},
        ),
        (
            "pyproject.toml",
            "pyproject.toml",
            "[project]\nname = 'demo'\nversion = '0.1.0'\n# changed\n",
            {"dashboard", "runtime"},
        ),
        ("uv.lock", "uv.lock", "lock-version = 2\n", {"dashboard", "runtime"}),
        (
            "libs/sdk",
            "libs/sdk/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        (
            "libs/foundation",
            "libs/foundation/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        (
            "libs/evaluation",
            "libs/evaluation/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        (
            "libs/core",
            "libs/core/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        (
            "libs/synthetic",
            "libs/synthetic/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        (
            "libs/vespa",
            "libs/vespa/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        (
            "libs/agents",
            "libs/agents/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        (
            "libs/telemetry-phoenix",
            "libs/telemetry-phoenix/input.py",
            "VALUE = 'changed'\n",
            {"dashboard", "runtime"},
        ),
        ("libs/runtime", "libs/runtime/input.py", "VALUE = 'changed'\n", {"runtime"}),
        (
            "configs/schemas",
            "configs/schemas/input.json",
            "{}\n",
            {"dashboard", "runtime"},
        ),
        (
            "configs/config.json",
            "configs/config.json",
            '{"changed": true}\n',
            {"dashboard", "runtime"},
        ),
        (
            "configs/agent_policies",
            "configs/agent_policies/input.yaml",
            "egress: []\n",
            {"runtime"},
        ),
        (
            ".dockerignore",
            ".dockerignore",
            "tests/\nsrc/\nscripts/run_*.py\n*.md\n# changed\n",
            ALL_IMAGE_FAMILIES - {"web"},
        ),
        (
            "libs/dashboard/Dockerfile",
            "libs/dashboard/Dockerfile",
            "FROM scratch\n",
            {"dashboard"},
        ),
        (
            "libs/dashboard",
            "libs/dashboard/input.py",
            "VALUE = 'changed'\n",
            {"dashboard"},
        ),
        ("scripts", "scripts/dashboard_tab.py", "VALUE = 'changed'\n", {"dashboard"}),
        (
            "clients/web/Dockerfile",
            "clients/web/Dockerfile",
            "FROM node:22.22.0-bookworm-slim\n",
            {"web"},
        ),
        (
            "clients/web",
            "clients/web/src/server/app.ts",
            "export const changed = true;\n",
            {"web"},
        ),
        (
            "deploy/pylate/Dockerfile",
            "deploy/pylate/Dockerfile",
            "FROM busybox\n",
            {"pylate"},
        ),
        (
            "libs/cli/cogniverse_cli/modal_inference/servers/pylate.py",
            "libs/cli/cogniverse_cli/modal_inference/servers/pylate.py",
            "VALUE = 'changed'\n",
            {"pylate"},
        ),
        (
            "deploy/gliner/Dockerfile",
            "deploy/gliner/Dockerfile",
            "FROM busybox\n",
            {"gliner"},
        ),
        (
            "libs/cli/cogniverse_cli/modal_inference/servers/gliner.py",
            "libs/cli/cogniverse_cli/modal_inference/servers/gliner.py",
            "VALUE = 'changed'\n",
            {"gliner"},
        ),
        (
            "deploy/clap_embed/Dockerfile",
            "deploy/clap_embed/Dockerfile",
            "FROM busybox\n",
            {"clap_embed"},
        ),
        (
            "deploy/vllm_audio/Dockerfile",
            "deploy/vllm_audio/Dockerfile",
            "FROM busybox\n",
            {"vllm_audio"},
        ),
        (
            "deploy/clap_embed/requirements.txt",
            "deploy/clap_embed/requirements.txt",
            "package==1\n",
            {"clap_embed"},
        ),
        (
            "libs/cli/cogniverse_cli/modal_inference/servers/clap.py",
            "libs/cli/cogniverse_cli/modal_inference/servers/clap.py",
            "VALUE = 'changed'\n",
            {"clap_embed"},
        ),
        (
            "deploy/face_embed/Dockerfile",
            "deploy/face_embed/Dockerfile",
            "FROM busybox\n",
            {"face_embed"},
        ),
        (
            "deploy/face_embed/requirements.txt",
            "deploy/face_embed/requirements.txt",
            "package==1\n",
            {"face_embed"},
        ),
        (
            "libs/cli/cogniverse_cli/modal_inference/servers/face.py",
            "libs/cli/cogniverse_cli/modal_inference/servers/face.py",
            "VALUE = 'changed'\n",
            {"face_embed"},
        ),
        (
            "deploy/video_embed/Dockerfile",
            "deploy/video_embed/Dockerfile",
            "FROM busybox\n",
            {"video_embed"},
        ),
        (
            "deploy/video_embed/requirements.txt",
            "deploy/video_embed/requirements.txt",
            "package==1\n",
            {"video_embed"},
        ),
        (
            "libs/cli/cogniverse_cli/modal_inference/servers/video_embed.py",
            "libs/cli/cogniverse_cli/modal_inference/servers/video_embed.py",
            "VALUE = 'changed'\n",
            {"video_embed"},
        ),
    ]

    @staticmethod
    def _git(repo_root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    @classmethod
    def _seed_git_repo(cls, tmp_path: Path) -> Path:
        repo_root = tmp_path / "repo"
        files = {
            "pyproject.toml": "[project]\nname = 'demo'\nversion = '0.1.0'\n",
            ".dockerignore": "tests/\nsrc/\nscripts/run_*.py\n*.md\n",
            "clients/web/Dockerfile": "FROM scratch\n",
            "clients/web/.dockerignore": "/node_modules\n/dist\n/tests\n",
            "clients/web/src/server/app.ts": "export const changed = false;\n",
            "clients/web/tests/server.test.ts": "export {};\n",
            "libs/core/module.py": "CORE = 'base'\n",
            "libs/dashboard/module.py": "DASHBOARD = 'base'\n",
            "libs/runtime/module.py": "RUNTIME = 'base'\n",
            "scripts/run_ignored.py": "VALUE = 'base'\n",
            "deploy/pylate/Dockerfile": "FROM scratch\n",
            "deploy/gliner/Dockerfile": "FROM scratch\n",
            "deploy/clap_embed/Dockerfile": "FROM scratch\n",
            "deploy/face_embed/Dockerfile": "FROM scratch\n",
            "deploy/video_embed/Dockerfile": "FROM scratch\n",
            "tests/test_only.py": "VALUE = 'base'\n",
        }
        for relative_path, content in files.items():
            path = repo_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        cls._git(repo_root, "init")
        cls._git(repo_root, "config", "user.email", "tests@example.com")
        cls._git(repo_root, "config", "user.name", "Tests")
        cls._git(repo_root, "add", "-A")
        cls._git(repo_root, "commit", "-m", "base")
        return repo_root

    @classmethod
    def _commit(cls, repo_root: Path, relative_path: str, content: str) -> None:
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        cls._git(repo_root, "add", "-A")
        cls._git(repo_root, "commit", "-m", f"change {relative_path}")

    @staticmethod
    def _versions(repo_root: Path) -> dict[str, str]:
        return {
            image: dev_version(repo_root, image)
            for image in images_mod.IMAGE_INPUT_PATHS
        }

    @staticmethod
    def _changed_images(before: dict[str, str], after: dict[str, str]) -> set[str]:
        return {image for image in before if before[image] != after[image]}

    def test_tests_only_commit_changes_no_image_tag(self, tmp_path: Path) -> None:
        repo_root = self._seed_git_repo(tmp_path)
        before = self._versions(repo_root)

        self._commit(repo_root, "tests/test_only.py", "VALUE = 'changed'\n")

        assert self._versions(repo_root) == before

    def test_dashboard_commit_changes_only_dashboard_tag(self, tmp_path: Path) -> None:
        repo_root = self._seed_git_repo(tmp_path)
        before = self._versions(repo_root)

        self._commit(repo_root, "libs/dashboard/module.py", "DASHBOARD = 'changed'\n")
        after = self._versions(repo_root)

        assert self._changed_images(before, after) == {"dashboard"}

    def test_web_tag_follows_its_own_context_ignore_rules(self, tmp_path: Path) -> None:
        """The web image builds from clients/web, so its source counts even
        where the root ignore rules (``src/``) would drop it, and its own
        ignore rules drop its tests and its ignore file still counts."""
        repo_root = self._seed_git_repo(tmp_path)
        before = self._versions(repo_root)

        self._commit(
            repo_root, "clients/web/tests/server.test.ts", "export const x = 1;\n"
        )
        assert self._versions(repo_root) == before

        self._commit(
            repo_root, "clients/web/src/client/App.tsx", "export const y = 1;\n"
        )
        after_source = self._versions(repo_root)
        assert self._changed_images(before, after_source) == {"web"}

        self._commit(repo_root, "clients/web/.dockerignore", "/node_modules\n/dist\n")
        assert self._changed_images(after_source, self._versions(repo_root)) == {"web"}

    def test_dockerignored_copy_source_commit_changes_no_image_tag(
        self, tmp_path: Path
    ) -> None:
        repo_root = self._seed_git_repo(tmp_path)
        before = self._versions(repo_root)

        self._commit(repo_root, "scripts/run_ignored.py", "VALUE = 'ignored change'\n")

        assert self._versions(repo_root) == before

    def test_core_commit_changes_both_app_image_tags(self, tmp_path: Path) -> None:
        repo_root = self._seed_git_repo(tmp_path)
        before = self._versions(repo_root)

        self._commit(repo_root, "libs/core/module.py", "CORE = 'changed'\n")
        after = self._versions(repo_root)

        assert self._changed_images(before, after) == {"dashboard", "runtime"}

    def test_pylate_deploy_commit_changes_only_pylate_tag(self, tmp_path: Path) -> None:
        repo_root = self._seed_git_repo(tmp_path)
        before = self._versions(repo_root)

        self._commit(repo_root, "deploy/pylate/Dockerfile", "FROM python:3.12\n")
        after = self._versions(repo_root)

        assert self._changed_images(before, after) == {"pylate"}

    def test_input_case_table_covers_every_declared_path(self) -> None:
        assert {case[0] for case in self.INPUT_CASES} == set(
            images_mod.DEPLOY_INPUT_PATHS
        )

    @pytest.mark.parametrize(
        ("declared_path", "relative_path", "content", "expected_changed_images"),
        INPUT_CASES,
    )
    def test_each_declared_input_changes_exact_image_tags(
        self,
        tmp_path: Path,
        declared_path: str,
        relative_path: str,
        content: str,
        expected_changed_images: set[str],
    ) -> None:
        repo_root = self._seed_git_repo(tmp_path)
        before = self._versions(repo_root)

        self._commit(repo_root, relative_path, content)
        after = self._versions(repo_root)

        assert declared_path in images_mod.DEPLOY_INPUT_PATHS
        assert self._changed_images(before, after) == expected_changed_images


class TestImageInputDeclarations:
    """Declared inputs cannot omit a source copied from a local build context."""

    REPO_ROOT = Path(__file__).resolve().parents[3]

    @staticmethod
    def _local_copy_sources(dockerfile: Path) -> list[str]:
        logical_lines: list[str] = []
        pending = ""
        for raw_line in dockerfile.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pending = f"{pending} {stripped}".strip()
            if pending.endswith("\\"):
                pending = pending[:-1].rstrip()
                continue
            logical_lines.append(pending)
            pending = ""

        sources: list[str] = []
        for line in logical_lines:
            tokens = shlex.split(line)
            if not tokens or tokens[0].upper() not in {"ADD", "COPY"}:
                continue
            if any(token.startswith("--from=") for token in tokens[1:]):
                continue
            operands = [token for token in tokens[1:] if not token.startswith("--")]
            sources.extend(operands[:-1])
        return sources

    def test_every_dockerfile_copy_source_is_covered_by_its_image_inputs(
        self,
    ) -> None:
        assert set(images_mod.IMAGE_DOCKERFILES) == set(images_mod.IMAGE_INPUT_PATHS)
        for image, dockerfile_path in images_mod.IMAGE_DOCKERFILES.items():
            declared = images_mod.IMAGE_INPUT_PATHS[image]
            context = images_mod.image_build_context(image)
            assert dockerfile_path in declared, f"{image} omits its Dockerfile"
            ignore_file = images_mod.image_dockerignore(image)
            assert (self.REPO_ROOT / ignore_file).is_file(), (
                f"{image} builds from {context} without {ignore_file}"
            )
            assert any(
                ignore_file == path or ignore_file.startswith(f"{path}/")
                for path in declared
            ), f"{image} omits {ignore_file}"
            for copied in self._local_copy_sources(self.REPO_ROOT / dockerfile_path):
                source = copied if context == "." else f"{context}/{copied}"
                covered = any(
                    source == path or source.startswith(f"{path.rstrip('/')}/")
                    for path in declared
                )
                assert covered, (
                    f"{image} input set omits {source!r} copied by {dockerfile_path}"
                )

    def test_the_runtime_image_ships_the_sandbox_policy_directory(self) -> None:
        """The runtime resolves agent policies relative to its working
        directory, so the image must copy the directory to that same path or
        every agent's egress policy is absent in the deployed runtime."""
        from cogniverse_runtime.sandbox_manager import _DEFAULT_POLICY_DIR

        dockerfile = self.REPO_ROOT / images_mod.IMAGE_DOCKERFILES["runtime"]
        lines = [line.strip() for line in dockerfile.read_text().splitlines()]
        final_stage = lines[
            max(i for i, line in enumerate(lines) if line.upper().startswith("FROM ")) :
        ]
        workdirs = [
            line.split()[1] for line in final_stage if line.startswith("WORKDIR ")
        ]
        assert workdirs == ["/app"]
        policy_copies = [
            shlex.split(line)[-2:]
            for line in final_stage
            if line.startswith("COPY ")
            and str(_DEFAULT_POLICY_DIR) in shlex.split(line)[-2]
        ]
        assert policy_copies == [[str(_DEFAULT_POLICY_DIR), f"./{_DEFAULT_POLICY_DIR}"]]
        assert sorted(
            path.name for path in (self.REPO_ROOT / _DEFAULT_POLICY_DIR).glob("*.yaml")
        ) == [
            "coding_agent.yaml",
            "orchestrator_agent.yaml",
            "routing_agent.yaml",
            "search_agent.yaml",
            "summarizer_agent.yaml",
        ]


class TestDeploymentImageIdentity:
    def test_identity_records_ordered_tags_values_and_set_overrides(
        self, tmp_path: Path
    ) -> None:
        root = _make_project_root(
            tmp_path,
            clap_embed=True,
            colbert_pylate=True,
            code_colbert_pylate=True,
        )
        values_file = root / "charts" / "cogniverse" / "values.yaml"
        set_overrides = {
            "runtime.backend": "rocm",
            "dashboard.backend": "rocm",
        }

        identity = images_mod.dev_deployment_identity(
            root,
            torch_backend="rocm",
            values_files=[values_file],
            set_overrides=set_overrides,
            versions=DEV_VERSIONS,
        )

        assert identity == {
            "backend": "rocm",
            "values_files": ("charts/cogniverse/values.yaml",),
            "set_overrides": {
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
            },
            "image_tags": (
                f"cogniverse/runtime-rocm:{DEV_TAGS['runtime']}",
                f"cogniverse/web:{DEV_TAGS['web']}",
                f"cogniverse/gliner:{DEV_TAGS['gliner']}",
                f"cogniverse/clap-embed:{DEV_TAGS['clap_embed']}",
                f"cogniverse/pylate:{DEV_TAGS['pylate']}-rocm",
            ),
            "chart_digest": (
                "sha256:d7bd3773fdbb4c2713ef1d2f6c422c45fa783a83dbdfdca464ec4d1fccc4232b"
            ),
        }

    def test_chart_content_changes_identity_without_changing_image_tags(
        self, tmp_path: Path
    ) -> None:
        root = _make_project_root(tmp_path)
        values_file = root / "charts" / "cogniverse" / "values.yaml"
        kwargs = {
            "torch_backend": "rocm",
            "values_files": [values_file],
            "set_overrides": {"runtime.backend": "rocm"},
            "versions": DEV_VERSIONS,
        }
        before = images_mod.dev_deployment_identity(root, **kwargs)

        template = root / "charts" / "cogniverse" / "templates" / "runtime.yaml"
        template.parent.mkdir()
        template.write_text("kind: Deployment\n")
        after = images_mod.dev_deployment_identity(root, **kwargs)

        assert after["image_tags"] == before["image_tags"]
        assert after["chart_digest"] != before["chart_digest"]


class TestBuildImages:
    """Tests for :func:`build_images`."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_uses_each_images_own_version(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        tags = build_images(root, torch_backend="rocm", versions=DEV_VERSIONS)

        assert tags == [
            f"cogniverse/runtime-rocm:{DEV_TAGS['runtime']}",
            f"cogniverse/web:{DEV_TAGS['web']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
        ]
        build_commands = [
            call.args[0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if call.args[0][:2] == ["docker", "build"]
        ]
        assert (
            f"SETUPTOOLS_SCM_PRETEND_VERSION={DEV_VERSIONS['runtime']}"
            in (build_commands[0])
        )
        assert build_commands[1] == [
            "docker",
            "build",
            "-f",
            "clients/web/Dockerfile",
            "-t",
            f"cogniverse/web:{DEV_TAGS['web']}",
            "clients/web",
        ]

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_skips_host_present_tag_but_returns_complete_set(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        root = _make_project_root(tmp_path)
        runtime_tag = f"cogniverse/runtime-cpu:{DEV_TAGS['runtime']}"
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout=f"{runtime_tag}\n", stderr=""
            ),
            subprocess.CompletedProcess(args=[], returncode=0),
            subprocess.CompletedProcess(args=[], returncode=0),
        ]

        tags = build_images(root, torch_backend="cpu", versions=DEV_VERSIONS)

        assert tags == [
            runtime_tag,
            f"cogniverse/web:{DEV_TAGS['web']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
        ]
        build_commands = [
            call.args[0]
            for call in mock_run.call_args_list
            if call.args[0][:2] == ["docker", "build"]
        ]
        assert build_commands == [
            [
                "docker",
                "build",
                "-f",
                "clients/web/Dockerfile",
                "-t",
                f"cogniverse/web:{DEV_TAGS['web']}",
                "clients/web",
            ],
            [
                "docker",
                "build",
                "-f",
                "deploy/gliner/Dockerfile",
                "-t",
                f"cogniverse/gliner:{DEV_TAGS['gliner']}",
                ".",
            ],
        ]

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_stops_when_host_inventory_fails(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        root = _make_project_root(tmp_path)
        inventory_command = [
            "docker",
            "image",
            "ls",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ]
        mock_run.side_effect = subprocess.CalledProcessError(1, inventory_command)

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            build_images(root, torch_backend="cpu", versions=DEV_VERSIONS)

        assert exc_info.value.cmd == inventory_command
        assert [call.args[0] for call in mock_run.call_args_list] == [inventory_command]

    def test_dev_image_tags_are_ordered_and_dedupe_shared_pylate(
        self, tmp_path: Path
    ) -> None:
        root = _make_project_root(
            tmp_path,
            clap_embed=True,
            colbert_pylate=True,
            code_colbert_pylate=True,
        )

        tags = images_mod.dev_image_tags(
            root,
            torch_backend="rocm",
            values_files=None,
            versions=DEV_VERSIONS,
        )

        assert tags == (
            f"cogniverse/runtime-rocm:{DEV_TAGS['runtime']}",
            f"cogniverse/web:{DEV_TAGS['web']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
            f"cogniverse/clap-embed:{DEV_TAGS['clap_embed']}",
            f"cogniverse/pylate:{DEV_TAGS['pylate']}-rocm",
        )

    def test_pylate_tag_distinguishes_torch_backend(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path, colbert_pylate=True)

        cpu_tags = images_mod.dev_image_tags(
            root, torch_backend="cpu", versions=DEV_VERSIONS
        )
        rocm_tags = images_mod.dev_image_tags(
            root, torch_backend="rocm", versions=DEV_VERSIONS
        )

        assert cpu_tags[-1] == f"cogniverse/pylate:{DEV_TAGS['pylate']}-cpu"
        assert rocm_tags[-1] == f"cogniverse/pylate:{DEV_TAGS['pylate']}-rocm"

    def test_concurrent_builds_share_one_cold_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _make_project_root(tmp_path)
        first_inventory = threading.Event()
        second_inventory = threading.Event()
        release_first = threading.Event()
        state_lock = threading.Lock()
        host_tags: set[str] = set()
        build_counts: Counter[str] = Counter()
        inventory_count = 0

        def run(command, **kwargs):
            nonlocal inventory_count
            if command[:3] == ["docker", "image", "ls"]:
                with state_lock:
                    inventory_count += 1
                    call_number = inventory_count
                    snapshot = "\n".join(sorted(host_tags))
                if call_number == 1:
                    first_inventory.set()
                    release_first.wait(timeout=2)
                else:
                    second_inventory.set()
                return subprocess.CompletedProcess(command, 0, stdout=snapshot)
            if command[:2] == ["docker", "build"]:
                tag = command[command.index("-t") + 1]
                with state_lock:
                    build_counts[tag] += 1
                    host_tags.add(tag)
                return subprocess.CompletedProcess(command, 0)
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr(images_mod.subprocess, "run", run)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                build_images, root, torch_backend="cpu", versions=DEV_VERSIONS
            )
            assert first_inventory.wait(timeout=1) is True
            second = executor.submit(
                build_images, root, torch_backend="cpu", versions=DEV_VERSIONS
            )
            overlapped_inventory = second_inventory.wait(timeout=0.2)
            release_first.set()
            results = [first.result(timeout=3), second.result(timeout=3)]

        expected_tags = [
            f"cogniverse/runtime-cpu:{DEV_TAGS['runtime']}",
            f"cogniverse/web:{DEV_TAGS['web']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
        ]
        assert overlapped_inventory is False
        assert results == [expected_tags, expected_tags]
        assert build_counts == Counter({tag: 1 for tag in expected_tags})

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_calls_docker_build(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """The default build (no sidecars enabled) is exactly three images:
        the backend-specific runtime, the web client and the backend-agnostic
        GLiNER sidecar, all tagged with the deploy-input-derived git version (``+``
        sanitized to ``-``). ColPali/Whisper/LateOn/DenseOn are served by
        vLLM."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        tags = build_images(root, torch_backend="cpu", versions=UNIFORM_DEV_VERSIONS)

        assert tags == [
            f"cogniverse/runtime-cpu:{DEV_TAG}",
            f"cogniverse/web:{DEV_TAG}",
            f"cogniverse/gliner:{DEV_TAG}",
        ]
        assert mock_run.call_count == 4  # type: ignore[attr-defined]
        for call in mock_run.call_args_list[1:]:  # type: ignore[attr-defined]
            cmd = call[0][0]
            assert cmd[0] == "docker"
            assert cmd[1] == "build"

    @patch("cogniverse_cli.images.subprocess.run")
    def test_build_images_runtime_passes_torch_backend_and_version(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """The runtime build gets the matching --build-arg
        TORCH_BACKEND=<name>, a tag carrying the deploy-input-derived git
        version, and the FULL git version fed into the git-less docker context
        via SETUPTOOLS_SCM_PRETEND_VERSION (the tag sanitizes ``+``, the
        build-arg keeps it)."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        build_images(root, torch_backend="rocm", versions=UNIFORM_DEV_VERSIONS)

        build_commands = [
            call.args[0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if call.args[0][:2] == ["docker", "build"]
        ]
        runtime_cmd, web_cmd, gliner_cmd = build_commands
        assert "TORCH_BACKEND=rocm" in runtime_cmd
        assert f"cogniverse/runtime-rocm:{DEV_TAG}" in runtime_cmd
        assert f"SETUPTOOLS_SCM_PRETEND_VERSION={DEV_VERSION}" in runtime_cmd
        # The web image installs no Python workspace and no torch: no args.
        assert web_cmd == [
            "docker",
            "build",
            "-f",
            "clients/web/Dockerfile",
            "-t",
            f"cogniverse/web:{DEV_TAG}",
            "clients/web",
        ]
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

        built = build_images(root, torch_backend="cpu", versions=UNIFORM_DEV_VERSIONS)

        assert built == [
            f"cogniverse/runtime-cpu:{DEV_TAG}",
            f"cogniverse/web:{DEV_TAG}",
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

        built = build_images(root, torch_backend="rocm", versions=UNIFORM_DEV_VERSIONS)

        assert built == [
            f"cogniverse/runtime-rocm:{DEV_TAG}",
            f"cogniverse/web:{DEV_TAG}",
            f"cogniverse/gliner:{DEV_TAG}",
            f"cogniverse/pylate:{DEV_TAG}-rocm",
        ]
        pylate_cmds = [
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if f"cogniverse/pylate:{DEV_TAG}-rocm" in call[0][0]
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
                f"cogniverse/pylate:{DEV_TAG}-rocm",
                ".",
            ]
        ]

        overrides = dev_image_set_values(
            root, torch_backend="rocm", versions=UNIFORM_DEV_VERSIONS
        )
        assert overrides["inference.colbert_pylate.image.tag"] == f"{DEV_TAG}-rocm"
        assert overrides["inference.code_colbert_pylate.image.tag"] == f"{DEV_TAG}-rocm"

    @patch("cogniverse_cli.images.subprocess.run")
    def test_disabled_sidecars_are_not_built(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """With every optional sidecar disabled, the build set is only the core
        three — a default ``up`` stays fast."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)

        built = build_images(root, torch_backend="cpu", versions=UNIFORM_DEV_VERSIONS)

        joined = " ".join(" ".join(c[0][0]) for c in mock_run.call_args_list)  # type: ignore[attr-defined]
        assert "cogniverse/face-embed" not in joined
        assert "cogniverse/clap-embed" not in joined
        assert len(built) == 3

    @patch("cogniverse_cli.images.subprocess.run")
    def test_ui_images_follow_their_enabled_flags(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """An overlay that turns the dashboard on and the web client off builds
        the dashboard for the host backend, with the workspace build args, and
        no web image."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)
        overlay = tmp_path / "values.ui.yaml"
        overlay.write_text(
            yaml.safe_dump({"web": {"enabled": False}, "dashboard": {"enabled": True}})
        )

        built = build_images(
            root, torch_backend="rocm", values_files=[overlay], versions=DEV_VERSIONS
        )

        assert built == [
            f"cogniverse/runtime-rocm:{DEV_TAGS['runtime']}",
            f"cogniverse/dashboard-rocm:{DEV_TAGS['dashboard']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
        ]
        dashboard_cmd = next(
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if f"cogniverse/dashboard-rocm:{DEV_TAGS['dashboard']}" in call[0][0]
        )
        assert dashboard_cmd == [
            "docker",
            "build",
            "-f",
            "libs/dashboard/Dockerfile",
            "--build-arg",
            "TORCH_BACKEND=rocm",
            "--build-arg",
            f"SETUPTOOLS_SCM_PRETEND_VERSION={DEV_VERSIONS['dashboard']}",
            "-t",
            f"cogniverse/dashboard-rocm:{DEV_TAGS['dashboard']}",
            ".",
        ]

    def test_verification_names_a_missing_web_image(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path)
        runtime_tag = f"cogniverse/runtime-cpu:{DEV_TAGS['runtime']}"

        with pytest.raises(RuntimeError) as excinfo:
            images_mod.verify_local_images_cover_deploy(
                root,
                None,
                built_tags=[runtime_tag],
                versions=DEV_VERSIONS,
                torch_backend="cpu",
            )

        assert str(excinfo.value) == (
            "Deploy enables first-party images that were not built: web -> "
            f"cogniverse/web:{DEV_TAGS['web']}. "
            "Build with the same values files helm receives."
        )

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
            root,
            torch_backend="cpu",
            values_files=[overlay],
            versions=UNIFORM_DEV_VERSIONS,
        )

        assert built == [
            f"cogniverse/runtime-cpu:{DEV_TAG}",
            f"cogniverse/web:{DEV_TAG}",
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

    @patch("cogniverse_cli.images.subprocess.run")
    def test_overlay_enabling_video_embed_adds_its_build(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        """video-embed COPYs its server from libs/ and its requirements from
        deploy/, so it builds from the repo root with no TORCH_BACKEND arg and
        is tagged with its own image family's version."""
        _completed(mock_run)
        root = _make_project_root(tmp_path)
        overlay = tmp_path / "values.dev.yaml"
        overlay.write_text(
            yaml.safe_dump({"inference": {"video_embed": {"enabled": True}}})
        )

        built = build_images(
            root,
            torch_backend="cpu",
            values_files=[overlay],
            versions=DEV_VERSIONS,
        )

        assert built == [
            f"cogniverse/runtime-cpu:{DEV_TAGS['runtime']}",
            f"cogniverse/web:{DEV_TAGS['web']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
            f"cogniverse/video-embed:{DEV_TAGS['video_embed']}",
        ]
        video_cmd = next(
            call[0][0]
            for call in mock_run.call_args_list  # type: ignore[attr-defined]
            if f"cogniverse/video-embed:{DEV_TAGS['video_embed']}" in call[0][0]
        )
        assert video_cmd[video_cmd.index("-f") + 1] == "deploy/video_embed/Dockerfile"
        assert video_cmd[-1] == "."
        assert not any(a.startswith("TORCH_BACKEND=") for a in video_cmd)


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


def test_release_builds_the_web_image_from_its_own_context() -> None:
    """The release publishes the image ``cogniverse up`` builds for the web
    client, from the same Dockerfile and build context, and every release
    entry builds from the context the CLI uses for its Dockerfile."""
    workflow_path = Path(__file__).parents[3] / ".github/workflows/release-images.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    image_matrix = workflow["jobs"]["build-push"]["strategy"]["matrix"]["include"]
    entries = {entry["repo"]: entry for entry in image_matrix}

    assert entries["web"] == {
        "repo": "web",
        "dockerfile": "clients/web/Dockerfile",
        "context": "clients/web",
        "backend": "",
    }
    assert images_mod.WEB_REPO == "cogniverse/web"
    family_by_dockerfile = {
        dockerfile: image for image, dockerfile in images_mod.IMAGE_DOCKERFILES.items()
    }
    assert {entry["repo"]: entry["context"] for entry in image_matrix} == {
        entry["repo"]: images_mod.image_build_context(
            family_by_dockerfile[entry["dockerfile"]]
        )
        for entry in image_matrix
    }


def _with_vllm_asr(root: Path, device: str, repository: str) -> Path:
    chart_dir = root / "charts" / "cogniverse"
    values = yaml.safe_load((chart_dir / "values.yaml").read_text())
    values["inference"]["vllm_asr"] = {
        "enabled": True,
        "device": device,
        "image": {"repository": repository, "tag": "0.1.0"},
    }
    (chart_dir / "values.yaml").write_text(yaml.safe_dump(values))
    return root


class TestDeviceImageBuilds:
    """The transcription image derives from its device's vLLM image, so the
    service's device picks both the repository and the build argument."""

    @patch("cogniverse_cli.images.subprocess.run")
    def test_rocm_transcription_builds_the_rocm_audio_image(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        _completed(mock_run)
        root = _with_vllm_asr(
            _make_project_root(tmp_path), "rocm", "cogniverse/vllm-audio-rocm"
        )

        built = build_images(root, torch_backend="cpu", versions=DEV_VERSIONS)

        audio_tag = f"cogniverse/vllm-audio-rocm:{DEV_TAGS['vllm_audio']}"
        assert built == [
            f"cogniverse/runtime-cpu:{DEV_TAGS['runtime']}",
            f"cogniverse/web:{DEV_TAGS['web']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
            audio_tag,
        ]
        audio_cmds = [
            call[0][0] for call in mock_run.call_args_list if audio_tag in call[0][0]
        ]
        assert audio_cmds == [
            [
                "docker",
                "build",
                "-f",
                "deploy/vllm_audio/Dockerfile",
                "--build-arg",
                "TORCH_BACKEND=rocm",
                "-t",
                audio_tag,
                ".",
            ]
        ]
        assert dev_image_set_values(
            root, torch_backend="cpu", versions=DEV_VERSIONS
        ) == {
            "runtime.imagesByBackend.cpu.tag": DEV_TAGS["runtime"],
            "web.image.tag": DEV_TAGS["web"],
            "inference.gliner.image.tag": DEV_TAGS["gliner"],
            "inference.vllm_asr.imagesByDevice.rocm.tag": DEV_TAGS["vllm_audio"],
        }

    def test_another_devices_audio_image_is_refused(self, tmp_path: Path) -> None:
        from cogniverse_cli.images import device_image_services

        root = _with_vllm_asr(
            _make_project_root(tmp_path), "rocm", "cogniverse/vllm-audio-cpu"
        )
        with pytest.raises(RuntimeError) as excinfo:
            device_image_services(root, None)
        assert str(excinfo.value) == (
            "inference.vllm_asr runs on device 'rocm' but resolves image "
            "'cogniverse/vllm-audio-cpu'; the rocm build is "
            "'cogniverse/vllm-audio-rocm'."
        )

    @patch("cogniverse_cli.images.subprocess.run")
    def test_upstream_image_is_not_built(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        _completed(mock_run)
        root = _with_vllm_asr(_make_project_root(tmp_path), "cuda", "vllm/vllm-openai")

        built = build_images(root, torch_backend="cpu", versions=DEV_VERSIONS)

        assert built == [
            f"cogniverse/runtime-cpu:{DEV_TAGS['runtime']}",
            f"cogniverse/web:{DEV_TAGS['web']}",
            f"cogniverse/gliner:{DEV_TAGS['gliner']}",
        ]

    def test_verification_names_a_missing_audio_image(self, tmp_path: Path) -> None:
        from cogniverse_cli.images import verify_local_images_cover_deploy

        root = _with_vllm_asr(
            _make_project_root(tmp_path), "cuda", "cogniverse/vllm-audio-cuda"
        )
        app_tags = [
            f"cogniverse/runtime-cpu:{DEV_TAGS['runtime']}",
            f"cogniverse/web:{DEV_TAGS['web']}",
        ]
        with pytest.raises(RuntimeError) as excinfo:
            verify_local_images_cover_deploy(
                root,
                None,
                built_tags=[
                    *app_tags,
                    f"cogniverse/vllm-audio-cpu:{DEV_TAGS['vllm_audio']}",
                ],
                versions=DEV_VERSIONS,
                torch_backend="cpu",
            )
        assert str(excinfo.value) == (
            "Deploy enables first-party images that were not built: vllm_asr -> "
            f"cogniverse/vllm-audio-cuda:{DEV_TAGS['vllm_audio']}. "
            "Build with the same values files helm receives."
        )
        verify_local_images_cover_deploy(
            root,
            None,
            built_tags=[
                *app_tags,
                f"cogniverse/vllm-audio-cuda:{DEV_TAGS['vllm_audio']}",
            ],
            versions=DEV_VERSIONS,
            torch_backend="cpu",
        )


class TestDevImageSetValues:
    """The chart --set overrides that point first-party images at the built tag."""

    def test_maps_core_images_to_the_git_tag(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path)
        overrides = dev_image_set_values(
            root, torch_backend="cpu", versions=DEV_VERSIONS
        )
        assert overrides == {
            "runtime.imagesByBackend.cpu.tag": DEV_TAGS["runtime"],
            "web.image.tag": DEV_TAGS["web"],
            "inference.gliner.image.tag": DEV_TAGS["gliner"],
        }

    def test_enabled_dashboard_gets_its_backend_tag(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path, web=False, dashboard=True)
        overrides = dev_image_set_values(
            root, torch_backend="rocm", versions=DEV_VERSIONS
        )
        assert overrides == {
            "runtime.imagesByBackend.rocm.tag": DEV_TAGS["runtime"],
            "dashboard.imagesByBackend.rocm.tag": DEV_TAGS["dashboard"],
            "inference.gliner.image.tag": DEV_TAGS["gliner"],
        }

    def test_backend_scopes_runtime_and_dashboard(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path)
        overrides = dev_image_set_values(
            root, torch_backend="rocm", versions=DEV_VERSIONS
        )
        assert "runtime.imagesByBackend.rocm.tag" in overrides
        assert "runtime.imagesByBackend.cpu.tag" not in overrides

    def test_includes_enabled_sidecars_only(self, tmp_path: Path) -> None:
        root = _make_project_root(tmp_path, face_embed=True)
        overrides = dev_image_set_values(
            root, torch_backend="cpu", versions=DEV_VERSIONS
        )
        assert overrides["inference.face_embed.image.tag"] == DEV_TAGS["face_embed"]
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
    def test_import_images_skips_tags_already_present_on_node(
        self, mock_run: MagicMock
    ) -> None:
        node_images = json.dumps(
            {
                "images": [
                    {
                        "id": "sha-present",
                        "repoTags": ["docker.io/img:a"],
                    }
                ]
            }
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout=node_images, stderr=""
            ),
            subprocess.CompletedProcess(args=[], returncode=0),
        ]

        import_images("cogniverse", ["img:a", "img:b"])

        assert [call.args[0] for call in mock_run.call_args_list] == [
            [
                "docker",
                "exec",
                "k3d-cogniverse-server-0",
                "crictl",
                "images",
                "-o",
                "json",
            ],
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
        ]

    @patch("cogniverse_cli.images.subprocess.run")
    def test_import_images_stops_when_node_inventory_fails(
        self, mock_run: MagicMock
    ) -> None:
        inventory_command = [
            "docker",
            "exec",
            "k3d-cogniverse-server-0",
            "crictl",
            "images",
            "-o",
            "json",
        ]
        mock_run.side_effect = subprocess.CalledProcessError(1, inventory_command)

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            import_images("cogniverse", ["img:a"])

        assert exc_info.value.cmd == inventory_command
        assert [call.args[0] for call in mock_run.call_args_list] == [inventory_command]

    def test_concurrent_imports_share_one_cold_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first_inventory = threading.Event()
        second_inventory = threading.Event()
        release_first = threading.Event()
        state_lock = threading.Lock()
        node_tags: set[str] = set()
        import_counts: Counter[str] = Counter()
        inventory_count = 0

        def run(command, **kwargs):
            nonlocal inventory_count
            if command[:3] == ["docker", "exec", "k3d-cogniverse-server-0"]:
                with state_lock:
                    inventory_count += 1
                    call_number = inventory_count
                    snapshot = json.dumps(
                        {
                            "images": [
                                {"id": tag, "repoTags": [f"docker.io/{tag}"]}
                                for tag in sorted(node_tags)
                            ]
                        }
                    )
                if call_number == 1:
                    first_inventory.set()
                    release_first.wait(timeout=2)
                else:
                    second_inventory.set()
                return subprocess.CompletedProcess(command, 0, stdout=snapshot)
            if command[:3] == ["k3d", "image", "import"]:
                tag = command[5]
                with state_lock:
                    import_counts[tag] += 1
                    node_tags.add(tag)
                return subprocess.CompletedProcess(command, 0)
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr(images_mod.subprocess, "run", run)
        tags = ["img:a", "img:b"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(import_images, "cogniverse", tags)
            assert first_inventory.wait(timeout=1) is True
            second = executor.submit(import_images, "cogniverse", tags)
            overlapped_inventory = second_inventory.wait(timeout=0.2)
            release_first.set()
            first.result(timeout=3)
            second.result(timeout=3)

        assert overlapped_inventory is False
        assert import_counts == Counter({"img:a": 1, "img:b": 1})

    @patch("cogniverse_cli.images.subprocess.run")
    def test_import_images_calls_k3d_import(self, mock_run: object) -> None:
        """Images are imported independently without a memory-heavy tools pod."""
        _completed(mock_run)

        import_images("cogniverse", ["img:a", "img:b"])

        assert mock_run.call_args_list == [  # type: ignore[attr-defined]
            mock_call(
                [
                    "docker",
                    "exec",
                    "k3d-cogniverse-server-0",
                    "crictl",
                    "images",
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ),
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
            subprocess.CompletedProcess(args=[], returncode=0),
            subprocess.CalledProcessError(returncode=1, cmd=failed_command),
        ]

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            import_images("cogniverse", ["img:a", "img:b", "img:c"])

        assert exc_info.value.cmd == failed_command
        assert [call.args[0] for call in mock_run.call_args_list] == [  # type: ignore[attr-defined]
            [
                "docker",
                "exec",
                "k3d-cogniverse-server-0",
                "crictl",
                "images",
                "-o",
                "json",
            ],
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

    CURRENT_TAGS = [
        "cogniverse/runtime-rocm:0.1.dev2420-g813e8e5c8",
        "cogniverse/dashboard-rocm:0.1.dev2420-g813e8e5c8",
        "cogniverse/gliner:0.1.dev2420-g813e8e5c8",
    ]

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
        removed = prune_superseded_images(self.CURRENT_TAGS, runner=self._runner(calls))

        rmi_cmds = [c for c in calls if c[:2] == ["docker", "rmi"]]
        removed_tags = {tag for c in rmi_cmds for tag in c[2:]}
        assert removed_tags == {
            "cogniverse/runtime-rocm:0.1.dev2397-g0f2366466",
        }
        assert set(removed) == removed_tags

    def test_node_prune_skips_ids_with_a_kept_tag(self):
        from cogniverse_cli.images import prune_superseded_images

        calls: list = []
        prune_superseded_images(
            self.CURRENT_TAGS,
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

    def test_each_repository_keeps_its_own_current_generation(self):
        from cogniverse_cli.images import prune_superseded_images

        calls: list = []
        current_tags = [
            "cogniverse/runtime-rocm:0.1.dev2420-g813e8e5c8",
            "cogniverse/dashboard-rocm:0.1.dev2397-g0f2366466",
            "cogniverse/gliner:0.1.dev2420-g813e8e5c8",
        ]

        removed = prune_superseded_images(current_tags, runner=self._runner(calls))

        assert removed == ["cogniverse/runtime-rocm:0.1.dev2397-g0f2366466"]


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

    @pytest.mark.parametrize(
        "device_overlay", ["values.cpu.yaml", "values.rocm.yaml", "values.cuda.yaml"]
    )
    def test_dev_deploy_pulls_no_locally_built_image(self, device_overlay: str) -> None:
        chart = Path(__file__).resolve().parents[3] / "charts" / "cogniverse"
        images = _read_third_party_images(
            [chart / "values.yaml", chart / "values.k3s.yaml", chart / device_overlay]
        )
        assert [image for image in images if image.startswith("cogniverse/")] == []

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
        result = _read_third_party_images([self._values_file(tmp_path)], skip_llm=False)
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
        result = _read_third_party_images([self._values_file(tmp_path)], skip_llm=True)
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
        assert _read_third_party_images([vf], skip_llm=True) == [
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
        assert _read_third_party_images([vf], skip_llm=True) == ["shared/img:1"]

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
        assert _read_third_party_images([vf], skip_llm=True) == [
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

        pull_and_import_third_party("cogniverse", [vf], skip_llm=True)

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
            pull_and_import_third_party("cogniverse", [vf], skip_llm=True)

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
            pull_and_import_third_party("cogniverse", [vf], skip_llm=True)

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

        pull_and_import_third_party("cogniverse", [vf], skip_llm=True)

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
            versions=DEV_VERSIONS,
        )

        assert overrides["inference.colbert_pylate.image.tag"] == (
            f"{DEV_TAGS['pylate']}-rocm"
        )
        assert (
            overrides["inference.code_colbert_pylate.image.tag"]
            == f"{DEV_TAGS['pylate']}-rocm"
        )

    def test_omitting_the_overlay_leaves_its_sidecar_on_the_placeholder_tag(
        self,
    ) -> None:
        """Computing overrides from chart defaults while helm applies the
        overlay is the trap: the overlay-only service keeps the chart's static
        placeholder tag, which no build ever produces, so the pod cannot pull
        it under pullPolicy=Never.
        """
        overrides = dev_image_set_values(
            self.REPO_ROOT, torch_backend="rocm", versions=DEV_VERSIONS
        )

        assert "inference.colbert_pylate.image.tag" in overrides
        assert "inference.code_colbert_pylate.image.tag" not in overrides

    def test_k3s_overlay_enables_the_video_embed_build(self) -> None:
        """The local stack's text-to-video retrieval runs through the X-CLIP
        sidecar, so the overlay `cogniverse up` deploys must put its image in
        the build set, tagged with the image family's own version, on the
        repository the chart renders."""
        from cogniverse_cli.images import LOCAL_IMAGE_BUILDS, first_party_services

        k3s = [self._chart("values.k3s.yaml")]

        assert enabled_sidecars(self.REPO_ROOT, k3s) == [
            "clap_embed",
            "video_embed",
            "colbert_pylate",
        ]
        overrides = dev_image_set_values(
            self.REPO_ROOT,
            torch_backend="cpu",
            values_files=k3s,
            versions=DEV_VERSIONS,
        )
        assert overrides["inference.video_embed.image.tag"] == DEV_TAGS["video_embed"]
        assert (
            first_party_services(self.REPO_ROOT, k3s)["video_embed"]
            == LOCAL_IMAGE_BUILDS["video_embed"][0]
        )


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
        from cogniverse_cli.images import (
            DEVICE_IMAGE_BUILDS,
            LOCAL_IMAGE_BUILDS,
            device_image_services,
            first_party_services,
        )

        values_files = [self._chart(name) for name in overlays]
        required = first_party_services(self.REPO_ROOT, values_files)
        devices = device_image_services(self.REPO_ROOT, values_files)

        missing = sorted(
            set(required) - set(LOCAL_IMAGE_BUILDS) - set(DEVICE_IMAGE_BUILDS)
        )
        assert missing == [], (
            f"chart enables first-party images with no build spec: {missing}"
        )
        for svc, repo in required.items():
            if svc in DEVICE_IMAGE_BUILDS:
                repos, dockerfile, _ = DEVICE_IMAGE_BUILDS[svc]
                spec_repo = repos[devices[svc]]
            else:
                spec_repo, dockerfile, _ = LOCAL_IMAGE_BUILDS[svc]
            assert spec_repo == repo, f"{svc}: chart repo {repo} != build {spec_repo}"
            assert (self.REPO_ROOT / dockerfile).exists(), f"{svc}: {dockerfile}"

    def test_first_party_services_ignores_registry_backed_services(self) -> None:
        """Stock vLLM services are pulled from a registry, never built; the
        transcription service runs the derived audio image for its device."""
        from cogniverse_cli.images import first_party_services

        required = first_party_services(
            self.REPO_ROOT, [self._chart("values.k3s.yaml")]
        )

        assert "denseon" not in required
        assert required["vllm_asr"] == "cogniverse/vllm-audio-cpu"

    @pytest.mark.parametrize(
        ("overlay", "device"),
        [
            (None, "cpu"),
            ("values.cpu.yaml", "cpu"),
            ("values.rocm.yaml", "rocm"),
            ("values.cuda.yaml", "cuda"),
        ],
    )
    def test_each_device_overlay_builds_the_audio_image_for_its_device(
        self, overlay: str | None, device: str
    ) -> None:
        from cogniverse_cli.images import dev_image_set_values, dev_image_tags

        values_files = [self._chart("values.k3s.yaml")]
        if overlay:
            values_files.append(self._chart(overlay))
        tags = dev_image_tags(
            self.REPO_ROOT,
            torch_backend="cpu",
            values_files=values_files,
            versions=DEV_VERSIONS,
        )
        audio = [tag for tag in tags if tag.startswith("cogniverse/vllm-audio")]
        assert audio == [f"cogniverse/vllm-audio-{device}:{DEV_TAGS['vllm_audio']}"]
        overrides = dev_image_set_values(
            self.REPO_ROOT,
            torch_backend="cpu",
            values_files=values_files,
            versions=DEV_VERSIONS,
        )
        assert {k: v for k, v in overrides.items() if "vllm_asr" in k} == {
            f"inference.vllm_asr.imagesByDevice.{device}.tag": DEV_TAGS["vllm_audio"]
        }

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

    def test_verification_rejects_pylate_tag_for_another_backend(
        self, tmp_path: Path
    ) -> None:
        from cogniverse_cli.images import verify_local_images_cover_deploy

        root = _make_project_root(tmp_path, colbert_pylate=True)
        values_file = root / "charts" / "cogniverse" / "values.yaml"
        values = yaml.safe_load(values_file.read_text())
        values["inference"]["colbert_pylate"]["image"] = {
            "repository": "cogniverse/pylate"
        }
        values_file.write_text(yaml.safe_dump(values))
        built_tags = [
            f"cogniverse/pylate:{DEV_TAGS['pylate']}-cpu",
        ]

        with pytest.raises(RuntimeError) as exc_info:
            verify_local_images_cover_deploy(
                root,
                None,
                built_tags=built_tags,
                versions=DEV_VERSIONS,
                torch_backend="rocm",
            )

        assert f"cogniverse/pylate:{DEV_TAGS['pylate']}-rocm" in str(exc_info.value)

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

        built = build_images(root, torch_backend="cpu", versions=UNIFORM_DEV_VERSIONS)

        with pytest.raises(RuntimeError) as excinfo:
            verify_local_images_cover_deploy(
                root,
                [overlay],
                built_tags=built,
                versions=UNIFORM_DEV_VERSIONS,
                torch_backend="cpu",
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
            root,
            torch_backend="cpu",
            values_files=[overlay],
            versions=UNIFORM_DEV_VERSIONS,
        )

        assert f"cogniverse/future-embed:{DEV_TAG}" in built
        verify_local_images_cover_deploy(
            root,
            [overlay],
            built_tags=built,
            versions=UNIFORM_DEV_VERSIONS,
            torch_backend="cpu",
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
                root,
                torch_backend="cpu",
                values_files=[overlay],
                versions=UNIFORM_DEV_VERSIONS,
            )


class TestMinioImagesArePrePulled:
    """Both MinIO images the chart runs are pinned in the chart values and land
    in the set the deploy pre-pulls and imports, so no step that runs ``mc`` —
    bucket bootstrap, backup upload, report upload, the e2e bucket probe —
    depends on a registry being reachable when it runs."""

    REPO_ROOT = Path(__file__).resolve().parents[3]
    CHART = REPO_ROOT / "charts" / "cogniverse"

    def _chart_minio(self) -> dict:
        return yaml.safe_load((self.CHART / "values.yaml").read_text())["minio"]

    def test_pre_pull_set_carries_both_pinned_minio_images(self) -> None:
        minio = self._chart_minio()
        images = _read_third_party_images(
            [self.CHART / "values.yaml", self.CHART / "values.k3s.yaml"],
            skip_llm=True,
        )

        repositories = (minio["image"]["repository"], minio["mcImage"]["repository"])
        assert [image for image in images if image.split(":")[0] in repositories] == [
            f"{minio['image']['repository']}:{minio['image']['tag']}",
            f"{minio['mcImage']['repository']}:{minio['mcImage']['tag']}",
        ]

    def test_pinned_minio_tags_do_not_float(self) -> None:
        """A floating tag defaults the kubelet to ``imagePullPolicy: Always``,
        so the node's cached copy is ignored and every run needs the registry."""
        minio = self._chart_minio()

        assert [minio["image"]["tag"], minio["mcImage"]["tag"]] != ["latest"] * 2
        assert [minio["image"]["pullPolicy"], minio["mcImage"]["pullPolicy"]] == [
            "IfNotPresent",
            "IfNotPresent",
        ]

    def test_no_template_restates_a_minio_image_reference(self) -> None:
        """Every MinIO image in the chart resolves through ``minio.*Image``; a
        literal in a template drifts from the reference the deploy imports."""
        offenders = [
            f"{path.relative_to(self.REPO_ROOT)}:{number}: {line.strip()}"
            for path in (self.CHART / "templates").rglob("*.yaml")
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if re.search(r'image:\s*"?[\w./-]*(minio|pgsty)/', line)
        ]

        assert offenders == []

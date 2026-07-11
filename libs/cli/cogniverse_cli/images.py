"""Container image build and import into k3d."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml

# First-party image repositories keyed by torch backend. Runtime + dashboard
# ship one image per backend; each bakes in the matching torch wheel —
# runtime-cpu carries torch+cpu, -cuda torch+cu128, -rocm torch+rocm6.4. Tags
# derive from the chart appVersion: dev builds are ``<appVersion>-dev``.
RUNTIME_REPOS_BY_BACKEND = {
    "cpu": "cogniverse/runtime-cpu",
    "cuda": "cogniverse/runtime-cuda",
    "rocm": "cogniverse/runtime-rocm",
}
DASHBOARD_REPOS_BY_BACKEND = {
    "cpu": "cogniverse/dashboard-cpu",
    "cuda": "cogniverse/dashboard-cuda",
    "rocm": "cogniverse/dashboard-rocm",
}
# GLiNER sidecar — backend-agnostic CPU-only NER server (deploy/gliner). Its
# chart image uses pullPolicy: Never, so k3d must have it built+imported or
# the pod ErrImageNeverPulls on a fresh deploy. One image, all backends.
GLINER_REPO = "cogniverse/gliner"
# Optional embedder sidecars — each backs a real opt-in feature (VideoPrism
# embeddings, acoustic search, face re-ID). Built only when their
# inference.<svc>.enabled resolves true in the deploy values, so a default
# build stays fast but flipping one on "just works". clap/face COPY from libs/
# and deploy/, so their build context is the repo root; videoprism is
# self-contained in its own directory. Keyed by inference service name.
SIDECAR_BUILDS = {
    "videoprism_jax": (
        "cogniverse/videoprism",
        "deploy/videoprism/Dockerfile",
        "deploy/videoprism",
    ),
    "clap_embed": ("cogniverse/clap-embed", "deploy/clap_embed/Dockerfile", "."),
    "face_embed": ("cogniverse/face-embed", "deploy/face_embed/Dockerfile", "."),
}
# colpali, whisper, and the LateOn/DenseOn text embedders are no longer
# built by us — vLLM serves them all:
# TomoroAI/tomoro-colqwen3-embed-4b via inference.vllm_colpali (vllm/vllm-openai-cpu)
# openai/whisper-large-v3-turbo via inference.vllm_asr (vllm/vllm-openai-cpu)
# lightonai/LateOn + lightonai/DenseOn via inference.colbert_pylate / denseon
# (vllm_token_embed / vllm_embed engines on vllm/vllm-openai-cpu)
# Operators pull vllm/vllm-openai-cpu (or per-device variants) directly.


def detect_torch_backend() -> str:
    """Return the torch backend matching the local host.

    Detection ladder (same shape as ``scripts/install_with_gpu.sh``):

    1. ``COGNIVERSE_TORCH_BACKEND`` env override.
    2. ``nvidia-smi`` reachable → cuda.
    3. ``rocminfo`` reports a ``gfx`` agent → rocm. Requires the calling
       user to have ``/dev/kfd`` access (render group); without that
       rocminfo falls through to (4).
    4. ``/sys/module/amdgpu`` loaded → rocm. The kernel module is
       enough evidence to install rocm wheels at build time — runtime
       GPU access is a separate concern and not all build paths need
       it. Catches the ROCm-host-but-no-render-group case.
    5. fallback → cpu.
    """
    explicit = os.environ.get("COGNIVERSE_TORCH_BACKEND")
    if explicit:
        return explicit

    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return "cuda"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    if shutil.which("rocminfo"):
        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Name:" in result.stdout and "gfx" in result.stdout:
                return "rocm"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    if Path("/sys/module/amdgpu").exists():
        return "rocm"

    return "cpu"


def has_workspace_source(project_root: Path) -> bool:
    """Check if workspace source is available for building images."""
    return (project_root / "libs" / "runtime").is_dir()


def read_app_version(project_root: Path) -> str:
    """Chart ``appVersion`` — the static release line (what the base
    ``values.yaml`` tags release images with)."""
    chart = project_root / "charts" / "cogniverse" / "Chart.yaml"
    data = yaml.safe_load(chart.read_text())
    return str(data["appVersion"])


def dev_version(project_root: Path) -> str:
    """Git-derived version (setuptools-scm) — the identical value hatch-vcs
    stamps on the Python wheels, so a local dev image and a local ``uv build``
    carry the same commit-unique version. Requires a real git checkout, which
    ``cogniverse up`` always has."""
    from setuptools_scm import get_version

    return get_version(root=str(project_root))


def _docker_tag(version: str) -> str:
    # Docker tags can't contain '+'; the git version already marks a dev build
    # (e.g. 0.1.dev2137-g<sha>), so there is no separate -dev suffix.
    return version.replace("+", "-")


def _dev_tag(repo: str, version: str) -> str:
    return f"{repo}:{_docker_tag(version)}"


def _deep_merge(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def enabled_sidecars(project_root: Path, values_files: list[Path] | None) -> list[str]:
    """Sidecar services whose ``inference.<svc>.enabled`` resolves true after
    merging the chart defaults with the deploy overlays helm will apply, in
    ``SIDECAR_BUILDS`` order."""
    merged: dict = (
        yaml.safe_load(
            (project_root / "charts" / "cogniverse" / "values.yaml").read_text()
        )
        or {}
    )
    for values_file in values_files or []:
        overlay = yaml.safe_load(Path(values_file).read_text()) or {}
        _deep_merge(merged, overlay)
    inference = merged.get("inference") or {}
    return [
        svc
        for svc in SIDECAR_BUILDS
        if isinstance(inference.get(svc), dict)
        and inference[svc].get("enabled") is True
    ]


def build_images(
    project_root: Path,
    torch_backend: str | None = None,
    values_files: list[Path] | None = None,
    version: str | None = None,
) -> list[str]:
    """Build all cogniverse-owned Docker images, tagged with the git-derived
    version (``dev_version``), so every local build is commit-unique. Pass
    ``version`` to override (tests, no git checkout).

    Builds the runtime + dashboard variants matching ``torch_backend``
    (auto-detected when None) plus the backend-agnostic GLiNER sidecar. The
    optional embedder sidecars (videoprism / clap-embed / face-embed) build only
    when their ``inference.<svc>.enabled`` resolves true across ``values_files``
    (the same overlays ``cogniverse up`` hands helm), so a default build stays
    fast while flipping a sidecar on "just works". ColPali, Whisper, and the
    LateOn/DenseOn text embedders are served by vLLM and pulled directly by k3d.
    """
    version = version or dev_version(project_root)
    backend = torch_backend or detect_torch_backend()

    # Runtime + dashboard install the workspace, which triggers hatch-vcs; the
    # docker context excludes .git, so pass the derived version in explicitly.
    workspace_arg = [
        "--build-arg",
        f"TORCH_BACKEND={backend}",
        "--build-arg",
        f"SETUPTOOLS_SCM_PRETEND_VERSION={version}",
    ]
    builds = [
        (
            _dev_tag(RUNTIME_REPOS_BY_BACKEND[backend], version),
            "libs/runtime/Dockerfile",
            ".",
            workspace_arg,
        ),
        (
            _dev_tag(DASHBOARD_REPOS_BY_BACKEND[backend], version),
            "libs/dashboard/Dockerfile",
            ".",
            workspace_arg,
        ),
        # GLiNER takes no TORCH_BACKEND arg and builds from its own context.
        (
            _dev_tag(GLINER_REPO, version),
            "deploy/gliner/Dockerfile",
            "deploy/gliner",
            [],
        ),
    ]
    for svc in enabled_sidecars(project_root, values_files):
        repo, dockerfile, context = SIDECAR_BUILDS[svc]
        builds.append((_dev_tag(repo, version), dockerfile, context, []))

    built: list[str] = []
    for tag, dockerfile, context, extra_args in builds:
        subprocess.run(
            ["docker", "build", "-f", dockerfile, *extra_args, "-t", tag, context],
            cwd=str(project_root),
            check=True,
            timeout=3600,
        )
        built.append(tag)
    return built


def dev_image_set_values(
    project_root: Path,
    torch_backend: str | None = None,
    values_files: list[Path] | None = None,
    version: str | None = None,
) -> dict[str, str]:
    """Chart ``--set`` overrides pointing every first-party image at the
    git-derived dev tag ``build_images`` produces, so ``cogniverse up`` deploys
    exactly what it built. ``values.k3s.yaml`` carries a static ``<line>-dev``
    placeholder that these override with the commit-unique tag."""
    backend = torch_backend or detect_torch_backend()
    tag = _docker_tag(version or dev_version(project_root))
    overrides = {
        f"runtime.imagesByBackend.{backend}.tag": tag,
        f"dashboard.imagesByBackend.{backend}.tag": tag,
        "inference.gliner.image.tag": tag,
    }
    for svc in enabled_sidecars(project_root, values_files):
        overrides[f"inference.{svc}.image.tag"] = tag
    return overrides


def import_images(cluster_name: str, tags: list[str]) -> None:
    """Import Docker images into a k3d cluster."""
    subprocess.run(
        ["k3d", "image", "import", *tags, "-c", cluster_name],
        check=True,
        timeout=1800,
    )


def _read_third_party_images(values_file: Path, skip_llm: bool = False) -> list[str]:
    """Read third-party image references from a Helm values file.

    Walks top-level vespa/phoenix/llm.builtin and every enabled
    ``inference.<svc>`` block including device-specific overrides.
    """
    with open(values_file) as f:
        values = yaml.safe_load(f) or {}

    images: list[str] = []

    def _add_image(image_block: object) -> None:
        if not isinstance(image_block, dict):
            return
        # pullPolicy: Never means a locally-built image — skip (it isn't
        # in any registry).
        if image_block.get("pullPolicy") == "Never":
            return
        repo = image_block.get("repository")
        if repo:
            tag = image_block.get("tag", "latest")
            images.append(f"{repo}:{tag}")

    _add_image(values.get("vespa", {}).get("image"))
    _add_image(values.get("phoenix", {}).get("image"))

    # Semantic-router gateway (Envoy + the SR image) — part of the default
    # stack, so its images must be pre-pulled or first boot ErrImagePulls.
    semantic_router = values.get("semanticRouter", {}) or {}
    if semantic_router.get("enabled") is not False:
        _add_image(semantic_router.get("envoy", {}).get("image"))
        _add_image(semantic_router.get("router", {}).get("image"))

    if not skip_llm:
        _add_image(values.get("llm", {}).get("builtin", {}).get("image"))

    # Mirror the chart's image-resolution order:
    # imagesByDevice[device] -> image
    inference = values.get("inference", {}) or {}
    for svc_cfg in inference.values():
        if not isinstance(svc_cfg, dict):
            continue
        if svc_cfg.get("enabled") is False:
            continue
        device = svc_cfg.get("device")
        by_device = svc_cfg.get("imagesByDevice") or {}
        if device and device in by_device:
            _add_image(by_device.get(device))
        _add_image(svc_cfg.get("image"))

    seen: set[str] = set()
    unique: list[str] = []
    for img in images:
        if img in seen:
            continue
        seen.add(img)
        unique.append(img)
    return unique


def pull_and_import_third_party(
    cluster_name: str,
    values_file: Path,
    *,
    skip_llm: bool = False,
) -> None:
    """Pull third-party images locally and import into k3d.

    Reads image references from the Helm values file rather than
    hardcoding them. This avoids slow in-cluster pulls that cause
    pod startup timeouts.
    """
    images = _read_third_party_images(values_file, skip_llm=skip_llm)
    if not images:
        return

    for image in images:
        subprocess.run(
            ["docker", "pull", image],
            check=False,  # Don't fail if pull is slow/offline
            timeout=600,
        )

    subprocess.run(
        ["k3d", "image", "import", *images, "-c", cluster_name],
        check=False,  # Non-fatal if import fails
        timeout=600,
    )

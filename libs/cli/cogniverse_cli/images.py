"""Container image build and import into k3d."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml

# Runtime + dashboard ship one image per torch backend. Each variant
# bakes in the matching torch wheel — cogniverse/runtime-cpu carries
# torch+cpu, -cuda carries torch+cu128, -rocm carries torch+rocm6.4.
RUNTIME_TAGS_BY_BACKEND = {
    "cpu": "cogniverse/runtime-cpu:dev",
    "cuda": "cogniverse/runtime-cuda:dev",
    "rocm": "cogniverse/runtime-rocm:dev",
}
DASHBOARD_TAGS_BY_BACKEND = {
    "cpu": "cogniverse/dashboard-cpu:dev",
    "cuda": "cogniverse/dashboard-cuda:dev",
    "rocm": "cogniverse/dashboard-rocm:dev",
}
PYLATE_TAG = "cogniverse/pylate:dev"
# colpali and whisper are no longer built by us — vLLM serves both:
# TomoroAI/tomoro-colqwen3-embed-4b via inference.vllm_colpali (vllm/vllm-openai-cpu)
# openai/whisper-large-v3-turbo via inference.vllm_asr (vllm/vllm-openai-cpu)
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


def build_images(
    project_root: Path,
    torch_backend: str | None = None,
) -> list[str]:
    """Build all cogniverse-owned Docker images.

    Builds the runtime + dashboard variants matching ``torch_backend``
    (auto-detected via ``detect_torch_backend()`` when None). The pylate
    inference sidecar is always built. ColPali and Whisper are now
    served via vLLM (vllm/vllm-openai-cpu) and pulled directly by k3d.
    """
    backend = torch_backend or detect_torch_backend()
    runtime_tag = RUNTIME_TAGS_BY_BACKEND[backend]
    dashboard_tag = DASHBOARD_TAGS_BY_BACKEND[backend]

    backend_arg = ["--build-arg", f"TORCH_BACKEND={backend}"]
    workspace_builds = [
        (runtime_tag, "libs/runtime/Dockerfile", ".", backend_arg),
        (dashboard_tag, "libs/dashboard/Dockerfile", ".", backend_arg),
        # Pass TORCH_BACKEND so the pylate sidecar wheel matches the host.
        (PYLATE_TAG, "deploy/pylate/Dockerfile", ".", backend_arg),
    ]
    built: list[str] = []
    for tag, dockerfile, context, extra_args in workspace_builds:
        subprocess.run(
            ["docker", "build", "-f", dockerfile, *extra_args, "-t", tag, context],
            cwd=str(project_root),
            check=True,
            timeout=3600,
        )
        built.append(tag)
    return built


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

    if not skip_llm:
        _add_image(values.get("llm", {}).get("builtin", {}).get("image"))

    # Mirror the chart's image-resolution order:
    # imagesByDevice[device] -> image -> pylate.imagesByDevice -> pylate.image
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
        pylate = svc_cfg.get("pylate") or {}
        if isinstance(pylate, dict):
            pylate_by_device = pylate.get("imagesByDevice") or {}
            if device and device in pylate_by_device:
                _add_image(pylate_by_device.get(device))
            _add_image(pylate.get("image"))

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

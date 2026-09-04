"""Container image build and import into k3d."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path

import yaml
from pathspec import GitIgnoreSpec

# First-party image repositories keyed by torch backend. Runtime + dashboard
# ship one image per backend; each bakes in the matching torch wheel —
# runtime-cpu carries torch+cpu, -cuda torch+cu128, -rocm torch+rocm6.4. Tags
# derive from the latest commit to their own build inputs: dev builds are
# ``<release>.dev<N>+g<sha>`` (``+`` later sanitizes to ``-`` for Docker).
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
# GLiNER sidecar — backend-agnostic CPU-only NER server. Its
# chart image uses pullPolicy: Never, so k3d must have it built+imported or
# the pod ErrImageNeverPulls on a fresh deploy. One image, all backends.
GLINER_REPO = "cogniverse/gliner"
# Optional embedder sidecars — each backs a real opt-in feature (acoustic
# search, face re-ID). Built only when their
# inference.<svc>.enabled resolves true in the deploy values, so a default
# build stays fast but flipping one on "just works". Their canonical servers
# live in the CLI modal-inference package, so every sidecar build uses the
# repository root as its context. Keyed by inference service name.
SIDECAR_BUILDS = {
    "clap_embed": ("cogniverse/clap-embed", "deploy/clap_embed/Dockerfile", "."),
    "face_embed": ("cogniverse/face-embed", "deploy/face_embed/Dockerfile", "."),
    # Both LateOn services run the same PyLate image — LateOn retrieval
    # needs PyLate's exact encode (query expansion over masked padding),
    # which stock vLLM cannot reproduce. build_images dedupes the shared
    # tag so enabling both services builds the image once.
    "colbert_pylate": ("cogniverse/pylate", "deploy/pylate/Dockerfile", "."),
    "code_colbert_pylate": ("cogniverse/pylate", "deploy/pylate/Dockerfile", "."),
}
# The PyLate image bakes the host-matching torch wheel, like runtime/dashboard.
_TORCH_BACKEND_SIDECARS = frozenset({"colbert_pylate", "code_colbert_pylate"})
# Every locally-built image keyed by the inference service that runs it.
# GLiNER is enabled by the chart defaults; the rest are opt-in.
LOCAL_IMAGE_BUILDS = {
    "gliner": (GLINER_REPO, "deploy/gliner/Dockerfile", "."),
    **SIDECAR_BUILDS,
}
# colpali, whisper, and the DenseOn dense embedder are served by vLLM, not
# built here:
# TomoroAI/tomoro-colqwen3-embed-4b via inference.vllm_colpali (vllm/vllm-openai-cpu)
# openai/whisper-large-v3-turbo via inference.vllm_asr (vllm/vllm-openai-cpu)
# lightonai/DenseOn via inference.denseon (vllm_embed engine)
# Operators pull vllm/vllm-openai-cpu (or per-device variants) directly.

_APP_WORKSPACE_INPUTS = (
    "pyproject.toml",
    "uv.lock",
    "libs/sdk",
    "libs/foundation",
    "libs/evaluation",
    "libs/core",
    "libs/synthetic",
    "libs/vespa",
    "libs/agents",
    "libs/telemetry-phoenix",
)

IMAGE_DOCKERFILES = {
    "runtime": "libs/runtime/Dockerfile",
    "dashboard": "libs/dashboard/Dockerfile",
    "pylate": "deploy/pylate/Dockerfile",
    "gliner": "deploy/gliner/Dockerfile",
    "clap_embed": "deploy/clap_embed/Dockerfile",
    "face_embed": "deploy/face_embed/Dockerfile",
}

# Host-side inputs read by each Docker build. Every set includes the Dockerfile
# and the root ignore rules in addition to all local COPY/ADD sources.
IMAGE_INPUT_PATHS = {
    "runtime": (
        IMAGE_DOCKERFILES["runtime"],
        *_APP_WORKSPACE_INPUTS,
        "libs/runtime",
        "configs/schemas",
        "configs/config.json",
        ".dockerignore",
    ),
    "dashboard": (
        IMAGE_DOCKERFILES["dashboard"],
        *_APP_WORKSPACE_INPUTS,
        "libs/dashboard",
        "configs/schemas",
        "configs/config.json",
        "scripts",
        ".dockerignore",
    ),
    "pylate": (
        IMAGE_DOCKERFILES["pylate"],
        "libs/cli/cogniverse_cli/modal_inference/servers/pylate.py",
        ".dockerignore",
    ),
    "gliner": (
        IMAGE_DOCKERFILES["gliner"],
        "libs/cli/cogniverse_cli/modal_inference/servers/gliner.py",
        ".dockerignore",
    ),
    "clap_embed": (
        IMAGE_DOCKERFILES["clap_embed"],
        "deploy/clap_embed/requirements.txt",
        "libs/cli/cogniverse_cli/modal_inference/servers/clap.py",
        ".dockerignore",
    ),
    "face_embed": (
        IMAGE_DOCKERFILES["face_embed"],
        "deploy/face_embed/requirements.txt",
        "libs/cli/cogniverse_cli/modal_inference/servers/face.py",
        ".dockerignore",
    ),
}

# Every path that affects at least one locally built image.
DEPLOY_INPUT_PATHS = tuple(
    dict.fromkeys(path for paths in IMAGE_INPUT_PATHS.values() for path in paths)
)


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


def _read_project_release_line(project_root: Path) -> str:
    pyproject = project_root / "pyproject.toml"
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    version = str(data["project"]["version"])
    parts = version.split(".")
    if len(parts) >= 3 and parts[-1] == "0":
        parts = parts[:-1]
    return ".".join(parts)


def _latest_deploy_input_commit(project_root: Path, image: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "log",
            "--format=%x1e%H",
            "--name-only",
            "--no-renames",
            "--",
            *IMAGE_INPUT_PATHS[image],
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    dockerignore = GitIgnoreSpec.from_lines(
        (project_root / ".dockerignore").read_text().splitlines()
    )
    always_included = {IMAGE_DOCKERFILES[image], ".dockerignore"}
    commit = ""
    for line in result.stdout.split("\n"):
        if line.startswith("\x1e"):
            commit = line.removeprefix("\x1e")
        elif (
            line
            and commit
            and (line in always_included or not dockerignore.match_file(line))
        ):
            return commit
    raise RuntimeError(f"No committed inputs found for image family {image!r}")


def dev_version(project_root: Path, image: str) -> str:
    """Git-derived version for the latest commit to one image's inputs.

    The shape still matches hatch-vcs' ``0.1.devN+g<sha>`` output, but the
    commit behind it is the most recent commit that touched that image's
    declared input set. A commit outside the set therefore keeps its tag.
    """
    release = _read_project_release_line(project_root)
    commit = _latest_deploy_input_commit(project_root, image)
    count = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "rev-list",
            "--count",
            commit,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    short_sha = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "rev-parse",
            "--short=9",
            commit,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{release}.dev{count}+g{short_sha}"


def dev_versions(project_root: Path) -> dict[str, str]:
    """Return the git-derived version for every locally built image family."""
    return {image: dev_version(project_root, image) for image in IMAGE_INPUT_PATHS}


def _docker_tag(version: str) -> str:
    # Docker tags can't contain '+'; the git version already marks a dev build
    # (e.g. 0.1.dev2137-g<sha>), so there is no separate -dev suffix.
    return version.replace("+", "-")


def _dev_tag(repo: str, version: str) -> str:
    return f"{repo}:{_docker_tag(version)}"


@contextmanager
def _image_cache_lock(scope: str):
    lock_id = hashlib.sha256(scope.encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"cogniverse-image-cache-{lock_id}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _deep_merge(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _merged_inference(project_root: Path, values_files: list[Path] | None) -> dict:
    """``inference`` block after merging the chart defaults with the deploy
    overlays helm will apply, in the order helm applies them."""
    merged: dict = (
        yaml.safe_load(
            (project_root / "charts" / "cogniverse" / "values.yaml").read_text()
        )
        or {}
    )
    for values_file in values_files or []:
        overlay = yaml.safe_load(Path(values_file).read_text()) or {}
        _deep_merge(merged, overlay)
    return merged.get("inference") or {}


def enabled_sidecars(project_root: Path, values_files: list[Path] | None) -> list[str]:
    """Sidecar services whose ``inference.<svc>.enabled`` resolves true after
    merging the chart defaults with the deploy overlays helm will apply, in
    ``SIDECAR_BUILDS`` order. Services with a non-empty ``externalUrl`` are
    Modal-hosted — no local pod, so no build."""
    inference = _merged_inference(project_root, values_files)
    return [
        svc
        for svc in SIDECAR_BUILDS
        if isinstance(inference.get(svc), dict)
        and inference[svc].get("enabled") is True
        and not inference[svc].get("externalUrl")
    ]


def first_party_services(
    project_root: Path, values_files: list[Path] | None
) -> dict[str, str]:
    """Enabled inference services whose resolved chart image is a first-party
    ``cogniverse/`` repository, mapped to that repository.

    No registry serves these, so each one the deploy renders must be built and
    imported or its pod never starts (``ErrImageNeverPull`` under
    ``pullPolicy: Never``, ``ErrImagePull`` otherwise). Derived from the same
    values helm receives, so enabling a service in values is the only step
    needed to bring its image into the build.
    """
    inference = _merged_inference(project_root, values_files)
    required: dict[str, str] = {}
    for svc, cfg in inference.items():
        if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
            continue
        if cfg.get("externalUrl"):
            # Modal-hosted: the deploy renders no pod, so no image is needed.
            continue
        # Chart image-resolution order: imagesByDevice[device] -> image.
        device = cfg.get("device")
        by_device = cfg.get("imagesByDevice") or {}
        image = by_device.get(device) if device and device in by_device else None
        if not isinstance(image, dict):
            image = cfg.get("image")
        if not isinstance(image, dict):
            continue
        repo = image.get("repository")
        if isinstance(repo, str) and repo.startswith("cogniverse/"):
            required[svc] = repo
    return required


def _resolved_versions(
    project_root: Path, versions: dict[str, str] | None
) -> dict[str, str]:
    resolved = dev_versions(project_root) if versions is None else dict(versions)
    missing = sorted(set(IMAGE_INPUT_PATHS) - set(resolved))
    if missing:
        raise ValueError(f"Missing versions for image families: {missing}")
    return resolved


def _image_family_for_dockerfile(dockerfile: str) -> str:
    for image, declared_dockerfile in IMAGE_DOCKERFILES.items():
        if declared_dockerfile == dockerfile:
            return image
    raise RuntimeError(
        f"No image input set is defined for Dockerfile {dockerfile!r}. "
        "Add matching IMAGE_DOCKERFILES and IMAGE_INPUT_PATHS entries."
    )


def _resolved_builds(
    project_root: Path,
    torch_backend: str,
    values_files: list[Path] | None,
    versions: dict[str, str],
) -> list[tuple[str, str, str, list[str]]]:
    unbuildable = sorted(
        set(first_party_services(project_root, values_files)) - set(LOCAL_IMAGE_BUILDS)
    )
    if unbuildable:
        raise RuntimeError(
            f"No image build is defined for enabled services: {unbuildable}. "
            "Add a LOCAL_IMAGE_BUILDS entry (repository, dockerfile, context)."
        )

    runtime_args = [
        "--build-arg",
        f"TORCH_BACKEND={torch_backend}",
        "--build-arg",
        f"SETUPTOOLS_SCM_PRETEND_VERSION={versions['runtime']}",
    ]
    dashboard_args = [
        "--build-arg",
        f"TORCH_BACKEND={torch_backend}",
        "--build-arg",
        f"SETUPTOOLS_SCM_PRETEND_VERSION={versions['dashboard']}",
    ]
    builds = [
        (
            _dev_tag(RUNTIME_REPOS_BY_BACKEND[torch_backend], versions["runtime"]),
            IMAGE_DOCKERFILES["runtime"],
            ".",
            runtime_args,
        ),
        (
            _dev_tag(DASHBOARD_REPOS_BY_BACKEND[torch_backend], versions["dashboard"]),
            IMAGE_DOCKERFILES["dashboard"],
            ".",
            dashboard_args,
        ),
        (
            _dev_tag(LOCAL_IMAGE_BUILDS["gliner"][0], versions["gliner"]),
            LOCAL_IMAGE_BUILDS["gliner"][1],
            LOCAL_IMAGE_BUILDS["gliner"][2],
            [],
        ),
    ]
    to_build = list(enabled_sidecars(project_root, values_files))
    to_build += [
        svc
        for svc in first_party_services(project_root, values_files)
        if svc not in to_build and svc != "gliner"
    ]
    for svc in to_build:
        repo, dockerfile, context = LOCAL_IMAGE_BUILDS[svc]
        image = _image_family_for_dockerfile(dockerfile)
        version = versions[image]
        if svc in _TORCH_BACKEND_SIDECARS:
            version = f"{version}-{torch_backend}"
        sidecar_args = (
            ["--build-arg", f"TORCH_BACKEND={torch_backend}"]
            if svc in _TORCH_BACKEND_SIDECARS
            else []
        )
        builds.append((_dev_tag(repo, version), dockerfile, context, sidecar_args))
    return builds


def dev_image_tags(
    project_root: Path,
    torch_backend: str | None = None,
    values_files: list[Path] | None = None,
    versions: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the ordered, deduplicated tags required by one deployment."""
    backend = torch_backend or detect_torch_backend()
    resolved = _resolved_versions(project_root, versions)
    return tuple(
        dict.fromkeys(
            tag
            for tag, _, _, _ in _resolved_builds(
                project_root, backend, values_files, resolved
            )
        )
    )


def dev_deployment_identity(
    project_root: Path,
    *,
    torch_backend: str,
    values_files: list[Path],
    set_overrides: dict[str, str],
    versions: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return the exact image and Helm inputs that identify a dev deploy."""
    chart_root = project_root / "charts" / "cogniverse"
    chart_hasher = hashlib.sha256()
    for path in sorted(
        candidate for candidate in chart_root.rglob("*") if candidate.is_file()
    ):
        chart_hasher.update(path.relative_to(chart_root).as_posix().encode())
        chart_hasher.update(b"\0")
        chart_hasher.update(path.read_bytes())
        chart_hasher.update(b"\0")
    return {
        "backend": torch_backend,
        "values_files": tuple(
            os.path.relpath(path, project_root) for path in values_files
        ),
        "set_overrides": dict(set_overrides),
        "image_tags": dev_image_tags(
            project_root,
            torch_backend=torch_backend,
            values_files=values_files,
            versions=versions,
        ),
        "chart_digest": f"sha256:{chart_hasher.hexdigest()}",
    }


def verify_local_images_cover_deploy(
    project_root: Path,
    values_files: list[Path] | None,
    *,
    built_tags: list[str],
    versions: dict[str, str] | None = None,
    torch_backend: str | None = None,
) -> None:
    """Raise unless every first-party image the deploy renders was built.

    The build set and the helm overlays are two inputs that must agree; when
    they diverge (a caller builds from chart defaults while helm applies an
    overlay that enables more) the pod fails at image pull, minutes later and
    far from the cause. Call this with the values files helm receives, before
    installing.
    """
    resolved = _resolved_versions(project_root, versions)
    backend = torch_backend or detect_torch_backend()
    have = set(built_tags)
    missing: dict[str, str] = {}
    for svc, repo in first_party_services(project_root, values_files).items():
        build_spec = LOCAL_IMAGE_BUILDS.get(svc)
        if build_spec is None:
            missing[svc] = repo
            continue
        _, dockerfile, _ = build_spec
        image = _image_family_for_dockerfile(dockerfile)
        version = resolved[image]
        if svc in _TORCH_BACKEND_SIDECARS:
            version = f"{version}-{backend}"
        expected = _dev_tag(repo, version)
        if expected not in have:
            missing[svc] = expected
    if missing:
        detail = ", ".join(f"{svc} -> {ref}" for svc, ref in sorted(missing.items()))
        raise RuntimeError(
            "Deploy enables first-party images that were not built: "
            f"{detail}. Build with the same values files helm receives."
        )


def build_images(
    project_root: Path,
    torch_backend: str | None = None,
    values_files: list[Path] | None = None,
    versions: dict[str, str] | None = None,
) -> list[str]:
    """Build absent cogniverse-owned images under their input-derived tags.

    Existing host tags are reused. Pass ``versions`` to override each image
    family's version (tests, no git checkout). The return value is the complete
    required tag set, including reused images.

    Builds the runtime + dashboard variants matching ``torch_backend``
    (auto-detected when None) plus the backend-agnostic GLiNER sidecar. Each
    optional embedder sidecar in ``SIDECAR_BUILDS`` is built only when its
    ``inference.<svc>.enabled`` resolves true across ``values_files``, so
    passing the overlays helm receives is what brings an enabled sidecar into
    the build; omitting them builds the chart-default set only. The PyLate
    image is shared by colbert_pylate and code_colbert_pylate and builds once.
    ColPali, Whisper, and DenseOn are served by vLLM and pulled directly.

    Raises when those values enable a first-party (``cogniverse/*``) service
    that has no ``LOCAL_IMAGE_BUILDS`` spec, rather than leaving its pod to
    fail at image pull.
    """
    backend = torch_backend or detect_torch_backend()
    resolved = _resolved_versions(project_root, versions)
    builds = _resolved_builds(project_root, backend, values_files, resolved)
    with _image_cache_lock("build"):
        host_inventory = subprocess.run(
            ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        host_tags = set((host_inventory.stdout or "").splitlines())
        required: list[str] = []
        for tag, dockerfile, context, extra_args in builds:
            if tag in required:
                continue
            required.append(tag)
            if tag in host_tags:
                continue
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-f",
                    dockerfile,
                    *extra_args,
                    "-t",
                    tag,
                    context,
                ],
                cwd=str(project_root),
                check=True,
                timeout=3600,
            )
    return required


def dev_image_set_values(
    project_root: Path,
    torch_backend: str | None = None,
    values_files: list[Path] | None = None,
    versions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Chart ``--set`` overrides for each first-party image's derived dev tag.

    These point Helm at the exact tags ``build_images`` resolves. The static
    ``<line>-dev`` placeholders in ``values.k3s.yaml`` are never deployed.
    """
    backend = torch_backend or detect_torch_backend()
    resolved = _resolved_versions(project_root, versions)
    overrides = {
        f"runtime.imagesByBackend.{backend}.tag": _docker_tag(resolved["runtime"]),
        f"dashboard.imagesByBackend.{backend}.tag": _docker_tag(resolved["dashboard"]),
        "inference.gliner.image.tag": _docker_tag(resolved["gliner"]),
    }
    for svc in enabled_sidecars(project_root, values_files):
        _, dockerfile, _ = LOCAL_IMAGE_BUILDS[svc]
        image = _image_family_for_dockerfile(dockerfile)
        version = resolved[image]
        if svc in _TORCH_BACKEND_SIDECARS:
            version = f"{version}-{backend}"
        overrides[f"inference.{svc}.image.tag"] = _docker_tag(version)
    return overrides


def import_images(cluster_name: str, tags: list[str]) -> None:
    """Import tags that are absent from the k3d server node."""
    with _image_cache_lock(f"import:{cluster_name}"):
        inventory = subprocess.run(
            [
                "docker",
                "exec",
                f"k3d-{cluster_name}-server-0",
                "crictl",
                "images",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        try:
            node_images = json.loads(inventory.stdout or "{}").get("images", [])
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "k3d node image inventory returned invalid JSON"
            ) from exc
        node_tags = {
            tag.removeprefix("docker.io/")
            for image in node_images
            for tag in image.get("repoTags", [])
        }
        for tag in tags:
            if tag in node_tags:
                continue
            subprocess.run(
                [
                    "k3d",
                    "image",
                    "import",
                    "--mode",
                    "direct",
                    tag,
                    "-c",
                    cluster_name,
                ],
                check=True,
                timeout=1800,
            )


_DEV_NUM_RE = re.compile(r"dev(\d+)")


def _dev_number(tag: str) -> int | None:
    m = _DEV_NUM_RE.search(tag)
    return int(m.group(1)) if m else None


def prune_superseded_images(
    current_tags: list[str],
    *,
    node_container: str | None = None,
    runner=subprocess.run,
) -> list[str]:
    """Remove old generations for each current ``cogniverse/*`` repository.

    Each repository keeps its current tag and the generation immediately
    preceding that tag, so independently versioned images retain one rollback.

    On the host, superseded tags are untagged with ``docker rmi`` (which only
    frees the image when its last tag goes). When ``node_container`` is set,
    the node's containerd is pruned via ``crictl rmi <id>`` — but only for IDs
    whose every tag is superseded, because crictl removes all of an ID's tags
    at once (the gliner image reuses one ID across builds).

    ``runner`` is injectable for tests. Returns the removed host tags.
    """
    current_by_repo = {
        tag.rsplit(":", 1)[0]: _dev_number(tag)
        for tag in current_tags
        if tag.startswith("cogniverse/") and ":" in tag
    }

    result = runner(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    host_tags: list[str] = []
    for line in (result.stdout or "").splitlines():
        if "\t" not in line:
            continue
        tag = line.rsplit("\t", 1)[0].strip()
        if tag.startswith("cogniverse/"):
            host_tags.append(tag)

    kept_by_repo: dict[str, set[int]] = {}
    for repo, current_num in current_by_repo.items():
        if current_num is None:
            continue
        kept = {current_num}
        below = [
            number
            for tag in host_tags
            if tag.rsplit(":", 1)[0] == repo
            and (number := _dev_number(tag)) is not None
            and number < current_num
        ]
        if below:
            kept.add(max(below))
        kept_by_repo[repo] = kept

    def _superseded(tag: str) -> bool:
        repo = tag.rsplit(":", 1)[0]
        n = _dev_number(tag)
        current_num = current_by_repo.get(repo)
        return (
            current_num is not None
            and n is not None
            and n < current_num
            and n not in kept_by_repo[repo]
        )

    removed: list[str] = []
    for tag in host_tags:
        if _superseded(tag):
            runner(
                ["docker", "rmi", tag],
                capture_output=True,
                text=True,
                check=False,
            )
            removed.append(tag)

    if node_container:
        node = runner(
            ["docker", "exec", node_container, "crictl", "images", "-o", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            images = json.loads(node.stdout or "{}").get("images", [])
        except (ValueError, TypeError):
            images = []
        for img in images:
            repo_tags = [
                t.split("docker.io/", 1)[-1]
                for t in img.get("repoTags", [])
                if "cogniverse/" in t
            ]
            if repo_tags and all(_superseded(t) for t in repo_tags):
                runner(
                    ["docker", "exec", node_container, "crictl", "rmi", img["id"]],
                    capture_output=True,
                    text=True,
                    check=False,
                )

    return removed


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
            check=True,
            timeout=600,
        )

    for image in images:
        subprocess.run(
            [
                "k3d",
                "image",
                "import",
                "--mode",
                "direct",
                image,
                "-c",
                cluster_name,
            ],
            check=True,
            timeout=600,
        )

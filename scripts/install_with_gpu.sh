#!/usr/bin/env bash
# Sync the project with the PyTorch backend that matches the local hardware.
#
# Why this script exists
# ----------------------
# The same pyproject.toml is consumed by every host (macOS dev box, CUDA
# workstation, AMD/ROCm Linux box, CPU-only CI). uv has no native concept of
# "default torch wheel per machine" that fires during `uv sync` (the
# `torch-backend` config key is honored only by `uv pip install`, not by
# `uv sync` / `uv lock` as of uv 0.11.7). So we use opt-in extras +
# per-extra wheel-index sources in pyproject.toml, and this wrapper picks
# the right extra for the host.
#
# Detection order:
#   1. macOS (Darwin)             → no extra (default PyPI arm64 wheel
#                                    already ships MPS / Metal support).
#   2. NVIDIA — `nvidia-smi` ok    → --extra cuda  (torch+cu128)
#   3. AMD ROCm — `rocminfo` finds → --extra rocm  (torch+rocm6.4 +
#      a HSA agent                    pytorch-triton-rocm)
#   4. amdgpu kernel module loaded → error: rocminfo missing, won't silently
#      but rocminfo missing           fall back to CPU
#   5. fallback                   → --extra cpu   (torch+cpu)
#
# Override with COGNIVERSE_TORCH_BACKEND=cpu|cuda|rocm|mac to skip detection.
# Extra arguments are forwarded to `uv sync` unchanged.
#
# Linux + ROCm gotchas
# --------------------
# 1. `/dev/kfd` and `/dev/dri/renderD*` are owned by the `render` group
#    (and sometimes `video`). Without group membership, every GPU call
#    returns hipErrorNoDevice — `torch.cuda.is_available()` is False and
#    you get silent CPU fallback. This script warns when that's the case.
#    Fix: `sudo usermod -aG render,video $USER` then re-login or `newgrp render`.
# 2. `uv run python ...` re-syncs on every call by default and would clobber
#    the rocm wheels. Add `export UV_NO_SYNC=1` to ~/.bashrc on the ROCm box
#    (do NOT set this on macOS — there's nothing to preserve there).
#
# Usage:
#   scripts/install_with_gpu.sh                  # autodetect
#   scripts/install_with_gpu.sh --frozen         # autodetect + flags
#   COGNIVERSE_TORCH_BACKEND=cpu scripts/install_with_gpu.sh
#   COGNIVERSE_TORCH_BACKEND=rocm scripts/install_with_gpu.sh   # bypass rocminfo check

set -euo pipefail

detect_backend() {
    if [[ -n "${COGNIVERSE_TORCH_BACKEND:-}" ]]; then
        echo "$COGNIVERSE_TORCH_BACKEND"
        return
    fi

    # macOS: default PyPI wheel includes MPS support — no extra needed.
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "mac"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
        return
    fi

    if command -v rocminfo >/dev/null 2>&1 \
        && rocminfo 2>/dev/null | grep -q 'Name:[[:space:]]*gfx'; then
        echo "rocm"
        return
    fi

    # amdgpu kernel module loaded but ROCm userspace missing — tell the user
    # rather than silently falling back to CPU.
    if [[ -d /sys/module/amdgpu ]] \
        && ! command -v rocminfo >/dev/null 2>&1; then
        echo "amdgpu detected but ROCm userspace (rocminfo) not installed." >&2
        echo "Install ROCm packages first, or run with COGNIVERSE_TORCH_BACKEND=cpu." >&2
        exit 2
    fi

    echo "cpu"
}

# When the user picks rocm, /dev/kfd and /dev/dri/renderD* are owned by the
# `render` group with no world access — a user not in that group will hit
# `hipErrorNoDevice` at runtime even though everything installs fine. Warn
# loudly so the failure mode isn't a silent CPU fallback hours into a job.
warn_if_missing_render_group() {
    [[ "$1" == "rocm" ]] || return 0
    [[ "$(uname -s)" == "Linux" ]] || return 0

    local groups
    groups="$(id -nG 2>/dev/null || true)"
    local missing=()
    [[ " $groups " == *" render "* ]] || missing+=("render")
    [[ " $groups " == *" video "* ]] || missing+=("video")
    [[ ${#missing[@]} -eq 0 ]] && return 0

    local joined
    joined="$(IFS=,; echo "${missing[*]}")"
    echo "[install_with_gpu] WARNING: user '$USER' is not in group(s): ${missing[*]}" >&2
    echo "[install_with_gpu]   /dev/kfd and /dev/dri/render* require these groups." >&2
    echo "[install_with_gpu]   torch.cuda.is_available() will return False until you run:" >&2
    echo "[install_with_gpu]     sudo usermod -aG $joined $USER" >&2
    echo "[install_with_gpu]   then log out and back in (or 'newgrp render')." >&2
}

main() {
    local backend
    backend="$(detect_backend)"
    case "$backend" in
        mac)
            echo "[install_with_gpu] platform=macOS — using default PyPI torch (MPS)" >&2
            exec uv sync "$@"
            ;;
        cpu|cuda|rocm)
            echo "[install_with_gpu] backend=$backend" >&2
            warn_if_missing_render_group "$backend"
            exec uv sync --extra "$backend" "$@"
            ;;
        *)
            echo "Unknown backend '$backend' (expected mac|cpu|cuda|rocm)" >&2
            exit 2
            ;;
    esac
}

main "$@"

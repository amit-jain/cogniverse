# Base image is selected per torch backend. ``cogniverse up`` passes
# ``--build-arg TORCH_BACKEND=cpu|cuda|rocm`` matching the host's
# detected GPU. The slim ``python:3.12-slim`` base + a torch+rocm
# wheel was insufficient for gfx1151 — torch wheel ships its own HIP
# runtime libraries but they need a full ROCm userspace (libdrm,
# /opt/amdgpu/share/libdrm/amdgpu.ids, hipblas, etc.) installed at
# the system level to talk to the kernel module. Even with privileged
# mode and /dev/kfd passthrough, ``torch.randn(2,2).cuda()``
# segfaulted with SIGSEGV. The official rocm/pytorch image bundles
# the entire ROCm 7.2.2 userspace alongside python+torch and works
# out of the box on gfx1151.
ARG TORCH_BACKEND=cpu
ARG ROCM_BASE=rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
ARG CUDA_BASE=nvidia/cuda:12.8.0-runtime-ubuntu24.04
ARG CPU_BASE=python:3.12-slim

FROM ${ROCM_BASE} AS rocm-base
FROM ${CUDA_BASE} AS cuda-base
FROM ${CPU_BASE} AS cpu-base
# Hop to the variant matching the build arg.
FROM ${TORCH_BACKEND}-base AS final

# python:3.12-slim has no curl; rocm/pytorch and nvidia/cuda do.
# apt-get may already be unavailable on slim if it's been stripped —
# the install only runs when needed.
RUN apt-get update 2>/dev/null && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates 2>/dev/null || true

WORKDIR /app

# rocm/pytorch already has torch+rocm 2.9.1 pre-installed so the case
# below is a no-op. cuda/cpu bases need the wheel install. ``pip
# install --upgrade`` keeps the existing torch when it matches PKG.
ARG TORCH_BACKEND
RUN case "$TORCH_BACKEND" in \
        rocm) echo "rocm/pytorch base ships torch+rocm; skipping install" ;; \
        cuda) pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
              --extra-index-url https://pypi.org/simple "torch==2.5.1" ;; \
        cpu|*) pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
              --extra-index-url https://pypi.org/simple "torch==2.5.1" ;; \
    esac

COPY deploy/pylate/requirements.txt ./
# fast-plaid (a pylate transitive dep used for indexing, not inference) has
# no aarch64 wheel. Install pylate --no-deps and add the minimum deps the
# encode path actually needs.
RUN pip install --no-cache-dir \
        fastapi==0.120.1 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        sentence-transformers==5.1.1 \
        transformers==4.57.1 \
        datasets \
        numpy \
    && pip install --no-cache-dir --no-deps pylate==1.4.0

# The module is import-free of cogniverse packages by design, so the
# image carries just this one file plus requirements.txt. Build from
# the REPO ROOT: docker build -f deploy/pylate/Dockerfile .
COPY libs/runtime/cogniverse_runtime/sidecars/colbert_pylate.py ./server.py

ENV MODEL_NAME=lightonai/LateOn \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]

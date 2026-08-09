# Base image is selected per torch backend. ``cogniverse up`` passes
# ``--build-arg TORCH_BACKEND=cpu|cuda|rocm`` matching the host's
# detected GPU. The slim ``python:3.12-slim`` base + a torch+rocm
# wheel was insufficient for gfx1151 — torch wheel ships its own HIP
# runtime libraries but they need a full ROCm userspace (libdrm,
# /opt/amdgpu/share/libdrm/amdgpu.ids, hipblas, etc.) installed at
# the system level to talk to the kernel module. The official
# rocm/pytorch image bundles the entire ROCm userspace alongside
# python+torch and works out of the box on gfx1151.
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

# rocm/pytorch ships its own torch+rocm build, so that case is a no-op.
# cpu/cuda install the workspace-pinned torch so the served encode
# reproduces the in-process PyLate oracle.
ARG TORCH_BACKEND
RUN case "$TORCH_BACKEND" in \
        rocm) echo "rocm/pytorch base ships torch+rocm; skipping install" ;; \
        cuda) pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
              --extra-index-url https://pypi.org/simple "torch==2.8.0" ;; \
        cpu|*) pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
              --extra-index-url https://pypi.org/simple "torch==2.8.0" ;; \
    esac

# fast-plaid (a pylate transitive dep used for indexing, not inference) has
# no aarch64 wheel. Install pylate --no-deps and add the pinned deps the
# encode path needs, matching the workspace lock so the container output
# matches the in-process oracle.
RUN pip install --no-cache-dir \
        fastapi==0.135.3 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        sentence-transformers==5.1.1 \
        transformers==4.56.2 \
        datasets==4.8.4 \
        numpy==2.4.4 \
    && pip install --no-cache-dir --no-deps pylate==1.4.0

# The module is import-free of cogniverse packages by design, so the
# image carries just this one file. Build from the REPO ROOT:
# docker build -f deploy/pylate/Dockerfile .
COPY libs/cli/cogniverse_cli/modal_inference/servers/pylate.py ./server.py

# Defaults match the pinned colbert_pylate contract; the chart and test
# fixtures always set MODEL_NAME + MODEL_REVISION explicitly.
ENV MODEL_NAME=lightonai/LateOn \
    MODEL_REVISION=c01907b70557ee5c7753680d4819a5cce1674b83 \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]

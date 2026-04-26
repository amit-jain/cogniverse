# Cogniverse Runtime Dockerfile
# Multi-stage build for optimized production image

FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy workspace configuration and lockfile
COPY pyproject.toml uv.lock ./

# Copy all workspace packages in the dependency chain
COPY libs/sdk ./libs/sdk
COPY libs/foundation ./libs/foundation
COPY libs/evaluation ./libs/evaluation
COPY libs/core ./libs/core
COPY libs/synthetic ./libs/synthetic
COPY libs/vespa ./libs/vespa
COPY libs/agents ./libs/agents
COPY libs/telemetry-phoenix ./libs/telemetry-phoenix
COPY libs/runtime ./libs/runtime

# Install dependencies with uv using the lockfile (default torch wheel
# is the cu128 build on Linux, replaced below per TORCH_BACKEND).
RUN uv sync --package cogniverse-runtime --extra all --no-dev --frozen

# Install pylate for ColBERT model loading without pulling its full dep tree
# (fast-plaid has no aarch64 wheel, and we must keep existing transformers).
# Install missing runtime deps that --no-deps skips.
RUN uv pip install pylate --no-deps && \
    uv pip install datasets

# Also install telemetry-phoenix into the same venv — agents discover it via entry points
# Pin evals<3 before installing telemetry-phoenix to prevent arize-phoenix
# from pulling evals 3.0+ which removes phoenix.evals.models namespace.
# TODO: remove constraint when upgrading to arize-phoenix 14.x
RUN uv pip install "arize-phoenix-evals<3" && \
    uv pip install -e ./libs/telemetry-phoenix

# Pick the torch wheel variant that matches the deployment target. Each
# backend produces a separate image (cogniverse/runtime-cpu /-cuda /-rocm)
# so callers don't ship NVIDIA runtime to a CPU pod or vice versa.
# Mirrors the whisper engine-per-image pattern (whisper-fw / -wx / -cpp).
#
#   docker build --build-arg TORCH_BACKEND=cpu  ...   →  torch+cpu
#   docker build --build-arg TORCH_BACKEND=cuda ...   →  torch+cu128 (default)
#   docker build --build-arg TORCH_BACKEND=rocm ...   →  torch+rocm6.4 + pytorch-triton-rocm
#
# Default cuda keeps the historical wheel choice for unspecified builds;
# the CLI (libs/cli/cogniverse_cli/images.py) detects the host backend
# and builds the matching tag.
ARG TORCH_BACKEND=cuda
RUN case "$TORCH_BACKEND" in \
        cpu) \
            uv pip install --reinstall \
                --index-url https://download.pytorch.org/whl/cpu \
                torch==2.8.0 torchvision==0.23.0 && \
            rm -rf /app/.venv/lib/python3.12/site-packages/nvidia ;; \
        rocm) \
            uv pip install --reinstall \
                --index-url https://download.pytorch.org/whl/rocm6.4 \
                torch==2.8.0+rocm6.4 torchvision==0.23.0+rocm6.4 \
                pytorch-triton-rocm==3.4.0 && \
            rm -rf /app/.venv/lib/python3.12/site-packages/nvidia ;; \
        cuda) \
            : ;; \
        *) \
            echo "Unknown TORCH_BACKEND=$TORCH_BACKEND" >&2 && exit 1 ;; \
    esac && \
    rm -rf /root/.cache/uv /root/.cache/pip

# Pre-download ML models so they don't need downloading at runtime.
# This eliminates startup delays and liveness probe timeouts in production
# and in tests that exercise the agent pipeline.
#
# Adds ~1.8 GB (GLiNER + spaCy) to the image. CI paths that only verify
# health/lifecycle (no agent queries) can skip this with
#   docker build --build-arg PREDOWNLOAD_MODELS=false
# to get a slim ~7 GB image instead of ~9 GB.
ARG PREDOWNLOAD_MODELS=true
ENV HF_HOME=/app/.cache/huggingface
# Ensure /app/.cache exists so the runtime-stage ``COPY --from=builder
# /app/.cache`` line works even when PREDOWNLOAD_MODELS=false (otherwise
# BuildKit fails with "calculate checksum of ref ...: /app/.cache: not
# found" because no step created the directory).
RUN mkdir -p /app/.cache && \
    if [ "$PREDOWNLOAD_MODELS" = "true" ]; then \
        .venv/bin/python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_large-v2.1')" && \
        uv pip install "en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"; \
    else \
        echo "Skipping model pre-download (PREDOWNLOAD_MODELS=false); models will fetch on first use"; \
    fi

# Runtime stage
FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno for DSPy RLM sandboxed code execution
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_DIR=/app/.deno
ENV PATH="/root/.deno/bin:${PATH}"

# Create non-root user
RUN useradd -m -u 1000 cogniverse && \
    mkdir -p /app /data /logs && \
    chown -R cogniverse:cogniverse /app /data /logs

# Set working directory
WORKDIR /app

# Copy installed virtual environment and pre-downloaded models from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=cogniverse:cogniverse /app/.cache /home/cogniverse/.cache

# Copy application code
COPY --chown=cogniverse:cogniverse libs/sdk ./libs/sdk
COPY --chown=cogniverse:cogniverse libs/foundation ./libs/foundation
COPY --chown=cogniverse:cogniverse libs/evaluation ./libs/evaluation
COPY --chown=cogniverse:cogniverse libs/core ./libs/core
COPY --chown=cogniverse:cogniverse libs/synthetic ./libs/synthetic
COPY --chown=cogniverse:cogniverse libs/vespa ./libs/vespa
COPY --chown=cogniverse:cogniverse libs/agents ./libs/agents
COPY --chown=cogniverse:cogniverse libs/telemetry-phoenix ./libs/telemetry-phoenix
COPY --chown=cogniverse:cogniverse libs/runtime ./libs/runtime
COPY --chown=cogniverse:cogniverse configs/schemas ./configs/schemas
COPY --chown=cogniverse:cogniverse configs/config.json ./configs/config.json

# Switch to non-root user
USER cogniverse

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV RUNTIME_HOST=0.0.0.0
ENV RUNTIME_PORT=8000

# Run the application
CMD ["python", "-m", "uvicorn", "cogniverse_runtime.main:app", "--host", "0.0.0.0", "--port", "8000"]

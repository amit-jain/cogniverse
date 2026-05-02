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

# Single install layer: uv sync (pulls default cu128 torch into a
# transient working tree), then the torch-swap + nvidia/triton cleanup
# happen BEFORE the layer is committed so the cu128 wheel bytes never
# land in the image. fast-plaid has no aarch64 wheel so pylate goes in
# with --no-deps + manual datasets pull. arize-phoenix-evals pinned
# <3 to keep phoenix.evals.models reachable until arize-phoenix 14.x.
ARG TORCH_BACKEND
RUN if [ -z "$TORCH_BACKEND" ]; then \
        echo "ERROR: TORCH_BACKEND build-arg is required (cpu|cuda|rocm)." >&2; \
        exit 1; \
    fi && \
    uv sync --package cogniverse-runtime --extra all --no-dev --frozen && \
    uv pip install pylate --no-deps && \
    uv pip install datasets && \
    uv pip install "arize-phoenix-evals<3" && \
    uv pip install -e ./libs/telemetry-phoenix && \
    case "$TORCH_BACKEND" in \
        cpu) \
            uv pip install --reinstall \
                --index-url https://download.pytorch.org/whl/cpu \
                torch==2.8.0 torchvision==0.23.0 ;; \
        rocm) \
            uv pip install --reinstall \
                --index-url https://download.pytorch.org/whl/rocm6.4 \
                torch==2.8.0+rocm6.4 torchvision==0.23.0+rocm6.4 \
                pytorch-triton-rocm==3.4.0 ;; \
        cuda) \
            : ;; \
        *) \
            echo "Unknown TORCH_BACKEND=$TORCH_BACKEND (expected cpu|cuda|rocm)" >&2 && exit 1 ;; \
    esac && \
    if [ "$TORCH_BACKEND" != "cuda" ]; then \
        rm -rf /app/.venv/lib/python3.12/site-packages/nvidia \
               /app/.venv/lib/python3.12/site-packages/nvidia_*-dist-info \
               /app/.venv/lib/python3.12/site-packages/triton \
               /app/.venv/lib/python3.12/site-packages/triton-*-dist-info; \
    fi && \
    rm -rf /root/.cache/uv /root/.cache/pip

# spaCy model wheel — venv-side install, not HF cache (so the chart's
# model-cache volume mount can't shadow it). HF Hub models like GLiNER
# load lazily into the chart-mounted HF cache instead.
RUN uv pip install "en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

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

COPY --from=builder /app/.venv /app/.venv

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

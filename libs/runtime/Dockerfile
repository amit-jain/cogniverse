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

# Bound uv's network behaviour: the default concurrency equals the CPU count, and
# that many simultaneous DNS lookups exhaust small upstream resolvers, surfacing as
# `dns error: Try again` (EAI_AGAIN) mid-download.
ENV UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_TIMEOUT=300

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

# hatch-vcs derives versions from git, but .dockerignore excludes .git, so the
# host passes the git-derived version in (build_images / release-images.yml).
ARG SETUPTOOLS_SCM_PRETEND_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --package cogniverse-runtime --extra all --no-dev --frozen && \
    uv pip install pylate --no-deps && \
    uv pip install datasets && \
    uv pip install "arize-phoenix-evals<3" && \
    uv pip install -e ./libs/telemetry-phoenix && \
    rm -rf /root/.cache/pip

# Runtime stage
FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Deno for DSPy RLM sandboxed code execution: a pinned release fetched with
# retries (the upstream install script downloads unpinned and unretried), into
# /usr/local/bin so the non-root cogniverse user can execute it. DENO_DIR is the
# runtime cache and points at /app (chown'd to cogniverse below).
ARG DENO_VERSION=2.9.6
RUN curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors \
        "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip" \
        -o /tmp/deno.zip \
    && unzip -q /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && deno --version
ENV DENO_DIR=/app/.deno

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
CMD ["python", "-m", "cogniverse_runtime.runtime_cli", "--host", "0.0.0.0", "--port", "8000"]

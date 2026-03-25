# Cogniverse Dashboard Dockerfile
# Multi-stage build for Streamlit dashboard

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
COPY libs/dashboard ./libs/dashboard

# Install dependencies with uv using the lockfile
RUN uv sync --package cogniverse-dashboard --no-dev --frozen

# Runtime stage
FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cogniverse && \
    mkdir -p /app /data /logs && \
    chown -R cogniverse:cogniverse /app /data /logs

# Set working directory
WORKDIR /app

# Copy installed virtual environment from builder
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
COPY --chown=cogniverse:cogniverse libs/dashboard ./libs/dashboard
COPY --chown=cogniverse:cogniverse configs/schemas ./configs/schemas
COPY --chown=cogniverse:cogniverse configs/config.json ./configs/config.json

# Switch to non-root user
USER cogniverse

# Create Streamlit config
RUN mkdir -p /home/cogniverse/.streamlit && \
    echo '[server]\nport = 8501\naddress = "0.0.0.0"\nheadless = true\n\n[theme]\nbase = "dark"\nprimaryColor = "#FF6B6B"\n' > /home/cogniverse/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run Streamlit dashboard
CMD ["streamlit", "run", "libs/dashboard/cogniverse_dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

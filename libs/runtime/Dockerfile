# Cogniverse Runtime Dockerfile
# Multi-stage build for optimized production image

FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY pyproject.toml ./
COPY libs/core ./libs/core
COPY libs/runtime ./libs/runtime

# Install dependencies with uv
RUN uv pip install --system -e ./libs/core
RUN uv pip install --system -e "./libs/runtime[all]"

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cogniverse && \
    mkdir -p /app /data /logs && \
    chown -R cogniverse:cogniverse /app /data /logs

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=cogniverse:cogniverse libs/core ./libs/core
COPY --chown=cogniverse:cogniverse libs/runtime ./libs/runtime
COPY --chown=cogniverse:cogniverse configs/config.json ./configs/config.json

# Switch to non-root user
USER cogniverse

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV RUNTIME_HOST=0.0.0.0
ENV RUNTIME_PORT=8000

# Run the application
CMD ["python", "-m", "uvicorn", "cogniverse_runtime.main:app", "--host", "0.0.0.0", "--port", "8000"]
